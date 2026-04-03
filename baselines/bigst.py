"""
Adapted from: https://github.com/usail-hkust/BigST
Reference: BigST: Linear Complexity Spatio-Temporal Graph Neural Network
for Traffic Forecasting on Large-Scale Road Networks (VLDB 2024)

Node/time/week embeddings are removed — they are handled by the dgl-spt
feature pipeline (FeaturesPreparatorForDeepModels). The core linearized
spatial convolution (Performer-style random feature attention) is preserved.
"""

import math

import numpy as np
import torch
import torch.nn as nn

from .base import BaseModel


def _create_products_of_givens_rotations(dim, seed):
    nb_givens_rotations = dim * int(math.ceil(math.log(float(dim))))
    q = np.eye(dim, dim)
    np.random.seed(seed)
    for _ in range(nb_givens_rotations):
        random_angle = math.pi * np.random.uniform()
        random_indices = np.random.choice(dim, 2)
        index_i = min(random_indices[0], random_indices[1])
        index_j = max(random_indices[0], random_indices[1])
        slice_i = q[index_i]
        slice_j = q[index_j]
        new_slice_i = math.cos(random_angle) * slice_i + math.cos(random_angle) * slice_j
        new_slice_j = -math.sin(random_angle) * slice_i + math.cos(random_angle) * slice_j
        q[index_i] = new_slice_i
        q[index_j] = new_slice_j
    return torch.tensor(q, dtype=torch.float32)


def _create_random_matrix(m, d, seed=0, scaling=0, struct_mode=False):
    nb_full_blocks = int(m / d)
    block_list = []
    current_seed = seed
    for _ in range(nb_full_blocks):
        torch.manual_seed(current_seed)
        if struct_mode:
            q = _create_products_of_givens_rotations(d, current_seed)
        else:
            unstructured_block = torch.randn((d, d))
            q, _ = torch.linalg.qr(unstructured_block)
            q = torch.t(q)
        block_list.append(q)
        current_seed += 1
    remaining_rows = m - nb_full_blocks * d
    if remaining_rows > 0:
        torch.manual_seed(current_seed)
        if struct_mode:
            q = _create_products_of_givens_rotations(d, current_seed)
        else:
            unstructured_block = torch.randn((d, d))
            q, _ = torch.linalg.qr(unstructured_block)
            q = torch.t(q)
        block_list.append(q[0:remaining_rows])
    final_matrix = torch.vstack(block_list)

    current_seed += 1
    torch.manual_seed(current_seed)
    if scaling == 0:
        multiplier = torch.norm(torch.randn((m, d)), dim=1)
    elif scaling == 1:
        multiplier = torch.sqrt(torch.tensor(float(d))) * torch.ones(m)
    else:
        raise ValueError(f"Scaling must be one of {{0, 1}}. Was {scaling}")

    return torch.matmul(torch.diag(multiplier), final_matrix)


def _random_feature_map(data, is_query, projection_matrix, numerical_stabilizer=1e-6):
    data_normalizer = 1.0 / torch.sqrt(torch.sqrt(torch.tensor(data.shape[-1], dtype=torch.float32)))
    data = data_normalizer * data
    ratio = 1.0 / torch.sqrt(torch.tensor(projection_matrix.shape[0], dtype=torch.float32))
    data_dash = torch.einsum("bnhd,md->bnhm", data, projection_matrix)
    diag_data = torch.sum(torch.square(data), dim=-1, keepdim=True) / 2.0

    if is_query:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.max(data_dash, dim=-1, keepdim=True).values)
            + numerical_stabilizer
        )
    else:
        last_dims_t = len(data_dash.shape) - 1
        attention_dims_t = len(data_dash.shape) - 3
        data_dash = ratio * (
            torch.exp(
                data_dash - diag_data
                - torch.max(
                    torch.max(data_dash, dim=last_dims_t, keepdim=True).values,
                    dim=attention_dims_t, keepdim=True
                ).values
            )
            + numerical_stabilizer
        )
    return data_dash


def _linear_kernel(x, node_vec1, node_vec2):
    """O(N) linearized attention: approximates full softmax attention between all node pairs."""
    # x: [B, N, 1, nhid], node_vec1/node_vec2: [B, N, 1, r]
    node_vec1 = node_vec1.permute(1, 0, 2, 3)  # [N, B, 1, r]
    node_vec2 = node_vec2.permute(1, 0, 2, 3)
    x = x.permute(1, 0, 2, 3)  # [N, B, 1, nhid]

    v2x = torch.einsum("nbhm,nbhd->bhmd", node_vec2, x)
    out1 = torch.einsum("nbhm,bhmd->nbhd", node_vec1, v2x)

    one_matrix = torch.ones(node_vec2.shape[0], device=node_vec1.device)
    node_vec2_sum = torch.einsum("nbhm,n->bhm", node_vec2, one_matrix)
    out2 = torch.einsum("nbhm,bhm->nbh", node_vec1, node_vec2_sum)

    out1 = out1.permute(1, 0, 2, 3)  # [B, N, 1, nhid]
    out2 = out2.permute(1, 0, 2).unsqueeze(-1)
    return out1 / out2


class _ConvApproximation(nn.Module):
    def __init__(self, tau, random_feature_dim):
        super().__init__()
        self.tau = tau
        self.random_feature_dim = random_feature_dim

    def forward(self, x, node_vec1, node_vec2):
        dim = node_vec1.shape[-1]

        random_seed = torch.ceil(torch.abs(torch.sum(node_vec1) * 1e8)).to(torch.int32)
        random_matrix = _create_random_matrix(
            self.random_feature_dim, dim, seed=random_seed
        ).to(node_vec1.device)

        nv1 = node_vec1 / math.sqrt(self.tau)
        nv2 = node_vec2 / math.sqrt(self.tau)
        nv1_prime = _random_feature_map(nv1, True, random_matrix)
        nv2_prime = _random_feature_map(nv2, False, random_matrix)

        x = _linear_kernel(x, nv1_prime, nv2_prime)
        return x


class _LinearizedConv(nn.Module):
    def __init__(self, dim, dropout, tau, random_feature_dim):
        super().__init__()
        self.input_fc = nn.Conv2d(dim, dim, kernel_size=(1, 1), bias=True)
        self.output_fc = nn.Conv2d(dim, dim, kernel_size=(1, 1), bias=True)
        self.activation = nn.Sigmoid()
        self.dropout_layer = nn.Dropout(p=dropout)
        self.conv_app = _ConvApproximation(tau, random_feature_dim)

    def forward(self, input_data, node_vec1, node_vec2):
        x = self.activation(self.input_fc(input_data)) * self.output_fc(input_data)
        x = self.dropout_layer(x)

        x = x.permute(0, 2, 3, 1)  # (B, N, 1, dim)
        x = self.conv_app(x, node_vec1, node_vec2)
        x = x.permute(0, 3, 1, 2)  # (B, dim, N, 1)
        return x


class BigST(BaseModel):
    """
    BigST backbone adapted for the dgl-spt pipeline.

    Differences from the original:
    - No node_emb_layer, time_emb_layer, week_emb_layer (handled by dgl-spt features).
    - input_fc projects from seq_len * input_dim (pre-embedded features) instead of
      output_length * in_dim (raw speed + time + weekday).
    - W_1 / W_2 query/key projections operate on the projected input features.
    - No regression layer (output projection is in the adapter).
    """

    def __init__(self, num_layers, hid_dim, tau, random_feature_dim, dropout,
                 use_residual=True, use_bn=True, **args):
        super().__init__(**args)
        self.num_layers = num_layers
        self.hid_dim = hid_dim
        self.use_residual = use_residual
        self.use_bn = use_bn

        self.input_fc = nn.Conv2d(self.seq_len * self.input_dim, hid_dim, kernel_size=(1, 1), bias=True)

        self.W_1 = nn.Conv2d(hid_dim, hid_dim, kernel_size=(1, 1), bias=True)
        self.W_2 = nn.Conv2d(hid_dim, hid_dim, kernel_size=(1, 1), bias=True)

        self.linear_conv = nn.ModuleList()
        self.bn = nn.ModuleList()
        for _ in range(num_layers):
            self.linear_conv.append(_LinearizedConv(hid_dim, dropout, tau, random_feature_dim))
            self.bn.append(nn.LayerNorm(hid_dim))

        self.activation = nn.ReLU()

    def forward(self, x):
        # x: (B, N, T, D), D = input_dim (already projected by adapter's input_linear)
        B, N, T, D = x.size()

        x = x.contiguous().view(B, N, -1).transpose(1, 2).unsqueeze(-1)  # (B, T*D, N, 1)
        x = self.input_fc(x)  # (B, hid_dim, N, 1)

        node_vec1 = self.W_1(x).permute(0, 2, 3, 1)  # (B, N, 1, hid_dim)
        node_vec2 = self.W_2(x).permute(0, 2, 3, 1)

        x_pool = [x]
        for i in range(self.num_layers):
            if self.use_residual:
                residual = x
            x = self.linear_conv[i](x, node_vec1, node_vec2)
            if self.use_residual:
                x = x + residual
            if self.use_bn:
                x = self.bn[i](x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            x_pool.append(x)

        x = torch.cat(x_pool, dim=1)  # (B, hid_dim * (num_layers + 1), N, 1)
        x = self.activation(x)
        return x
