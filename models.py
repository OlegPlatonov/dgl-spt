import torch
from torch import nn
from modules import (ResidualModuleWrapper, FeedForwardModule, GraphMeanAggregationAndFeedForwardModule,
                     GraphAttnGATAggregationAndFeedForwardModule, GraphAttnTrfAggregationAndFeedForwardModule)
from plr_embeddings import PLREmbeddings


MODULES = {
    'ResNet': FeedForwardModule,
    'ResNet-MeanAggr': GraphMeanAggregationAndFeedForwardModule,
    'ResNet-AttnGATAggr': GraphAttnGATAggregationAndFeedForwardModule,
    'ResNet-AttnTrfAggr': GraphAttnTrfAggregationAndFeedForwardModule
}


NORMALIZATION = {
    'None': nn.Identity,
    'LayerNorm': nn.LayerNorm,
    'BatchNorm': nn.BatchNorm1d
}


class LinearModel(nn.Module):
    def __init__(self, features_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(in_features=features_dim, out_features=output_dim)

    def forward(self, graph, x):
        return self.linear(x).squeeze(1)


class NeuralNetworkModel(nn.Module):
    def __init__(self, model_name, num_layers, features_dim, hidden_dim, output_dim, num_heads, normalization, dropout,
                 use_plr, num_features_mask, plr_num_frequencies, plr_frequency_scale, plr_embedding_dim, use_plr_lite):
        super().__init__()

        module = MODULES[model_name]
        normalization = NORMALIZATION[normalization]

        self.use_plr = use_plr
        if use_plr:
            num_features_dim = num_features_mask.sum()
            self.plr_embeddings = PLREmbeddings(num_features=num_features_dim, num_frequencies=plr_num_frequencies,
                                                frequency_scale=plr_frequency_scale, embedding_dim=plr_embedding_dim,
                                                lite=use_plr_lite)
            self.register_buffer('num_features_mask', num_features_mask)
            input_dim = features_dim - num_features_dim + num_features_dim * plr_embedding_dim
        else:
            input_dim = features_dim

        self.input_linear = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.act = nn.GELU()

        self.residual_modules = nn.ModuleList()
        for _ in range(num_layers):
            residual_module = ResidualModuleWrapper(module=module,
                                                    normalization=normalization,
                                                    dim=hidden_dim,
                                                    num_heads=num_heads,
                                                    dropout=dropout)

            self.residual_modules.append(residual_module)

        self.output_normalization = normalization(hidden_dim)
        self.output_linear = nn.Linear(in_features=hidden_dim, out_features=output_dim)

    def forward(self, graph, x):
        if self.use_plr:
            x_num = x[:, self.num_features_mask]
            x_num_embedded = self.plr_embeddings(x_num).flatten(start_dim=1)
            x = torch.cat([x_num_embedded, x[:, ~self.num_features_mask]], axis=1)

        x = self.input_linear(x)
        x = self.dropout(x)
        x = self.act(x)

        for residual_module in self.residual_modules:
            x = residual_module(graph, x)

        x = self.output_normalization(x)
        x = self.output_linear(x).squeeze(1)

        return x
