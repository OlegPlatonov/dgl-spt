import torch
from torch import nn
from dgl import ops


class ResidualModuleWrapper(nn.Module):
    def __init__(self, module, normalization, dim, **kwargs):
        super().__init__()
        self.normalization = normalization(dim)
        self.module = module(dim=dim, **kwargs)

    def forward(self, graph, x):
        x_res = self.normalization(x)
        x_res = self.module(graph, x_res)
        x = x + x_res

        return x


class FeedForwardModule(nn.Module):
    def __init__(self, dim, num_inputs=1, dropout=0, **kwargs):
        super().__init__()
        self.linear_1 = nn.Linear(in_features=dim * num_inputs, out_features=dim)
        self.dropout_1 = nn.Dropout(p=dropout)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(in_features=dim, out_features=dim)
        self.dropout_2 = nn.Dropout(p=dropout)

    def forward(self, graph, x):
        x = self.linear_1(x)
        x = self.dropout_1(x)
        x = self.act(x)
        x = self.linear_2(x)
        x = self.dropout_2(x)

        return x


class GraphAggregationAndFeedForwardModule(nn.Module):
    def __init__(self, graph_aggregation_module, dim, dropout, **kwargs):
        super().__init__()
        self.graph_aggregation_module = graph_aggregation_module(dim=dim, dropout=dropout, **kwargs)
        self.feed_forward_module = FeedForwardModule(dim=dim, num_inputs=2, dropout=dropout)

    def forward(self, graph, x):
        x_aggregated = self.graph_aggregation_module(graph, x)
        x = torch.cat([x, x_aggregated], axis=1)
        x = self.feed_forward_module(graph, x)

        return x


class GraphMeanAggregation(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, graph, x):
        x_aggregated = ops.copy_u_mean(graph, x)

        return x_aggregated


class GraphMeanAggregationAndFeedForwardModule(GraphAggregationAndFeedForwardModule):
    def __init__(self, dim, dropout, **kwargs):
        super().__init__(graph_aggregation_module=GraphMeanAggregation, dim=dim, dropout=dropout)


def _check_dim_and_num_heads_consistency(dim, num_heads):
    if dim % num_heads != 0:
        raise ValueError('Dimension mismatch: hidden_dim should be a multiple of num_heads.')


class GraphAttnGATAggregation(nn.Module):
    def __init__(self, dim, num_heads, **kwargs):
        super().__init__()

        _check_dim_and_num_heads_consistency(dim, num_heads)
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.attn_linear_u = nn.Linear(in_features=dim, out_features=num_heads)
        self.attn_linear_v = nn.Linear(in_features=dim, out_features=num_heads, bias=False)
        self.attn_act = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, graph, x):
        attn_scores_u = self.attn_linear_u(x)
        attn_scores_v = self.attn_linear_v(x)
        attn_scores = ops.u_add_v(graph, attn_scores_u, attn_scores_v)
        attn_scores = self.attn_act(attn_scores)
        attn_probs = ops.edge_softmax(graph, attn_scores)

        x = x.reshape(-1, self.head_dim, self.num_heads)
        x_aggregated = ops.u_mul_e_sum(graph, x, attn_probs)
        x_aggregated = x_aggregated.reshape(-1, self.dim)

        return x_aggregated


class GraphAttnGATAggregationAndFeedForwardModule(GraphAggregationAndFeedForwardModule):
    def __init__(self, dim, num_heads, dropout, **kwargs):
        super().__init__(graph_aggregation_module=GraphAttnGATAggregation, dim=dim, num_heads=num_heads,
                         dropout=dropout)


class GraphAttnTrfAggregation(nn.Module):
    def __init__(self, dim, num_heads, dropout, **kwargs):
        super().__init__()

        _check_dim_and_num_heads_consistency(dim, num_heads)
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attn_scores_multiplier = 1 / torch.tensor(self.head_dim).sqrt()

        self.attn_qkv_linear = nn.Linear(in_features=dim, out_features=dim * 3)

        self.output_linear = nn.Linear(in_features=dim, out_features=dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, graph, x):
        qkvs = self.attn_qkv_linear(x)
        qkvs = qkvs.reshape(-1, self.num_heads, self.head_dim * 3)
        queries, keys, values = qkvs.split(split_size=(self.head_dim, self.head_dim, self.head_dim), dim=-1)

        attn_scores = ops.u_dot_v(graph, keys, queries) * self.attn_scores_multiplier
        attn_probs = ops.edge_softmax(graph, attn_scores)

        x_aggregated = ops.u_mul_e_sum(graph, values, attn_probs)
        x_aggregated = x_aggregated.reshape(-1, self.dim)

        x_aggregated = self.output_linear(x_aggregated)
        x_aggregated = self.dropout(x_aggregated)

        return x_aggregated


class GraphAttnTrfAggregationAndFeedForwardModule(GraphAggregationAndFeedForwardModule):
    def __init__(self, dim, num_heads, dropout, **kwargs):
        super().__init__(graph_aggregation_module=GraphAttnTrfAggregation, dim=dim, num_heads=num_heads,
                         dropout=dropout)
