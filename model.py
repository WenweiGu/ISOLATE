import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as NN
import torch_geometric.nn
from DConv import DCConv
from GAT import GraphAttentionLayer
from CVAE import CVAE


class RTAnomaly(nn.Module):
    def __init__(self, ndim, len_window, gnn_dim, pooling_ratio, threshold, dropout, filters, kernels, dilation,
                 layers, gru_dim, device, recon_filter, hidden_size, latent_size):
        super(RTAnomaly, self).__init__()
        self.device = device

        self.threshold = threshold
        self.dropout = dropout

        # gcn
        self.num_hidden = gnn_dim
        self.pooling_ratio = pooling_ratio

        self.gat = GraphAttentionLayer(ndim, len_window)
        self.graph_conv1 = NN.GCNConv(len_window, self.num_hidden)
        self.graph_conv2 = NN.GCNConv(self.num_hidden, self.num_hidden)
        self.graph_conv3 = NN.GCNConv(self.num_hidden, self.num_hidden)

        self.graph_pool1 = NN.SAGPooling(self.num_hidden, ratio=self.pooling_ratio)
        self.graph_pool2 = NN.SAGPooling(self.num_hidden, ratio=self.pooling_ratio)
        self.graph_pool3 = NN.SAGPooling(self.num_hidden, ratio=self.pooling_ratio)

        # LSTM
        self.channel = ndim
        self.ts_length = len_window

        self.gru_dim = gru_dim
        self.gru = nn.GRU(self.ts_length, self.gru_dim)

        paddings = np.array(dilation) * (np.array(kernels) - 1)

        self.conv_1 = DCConv(self.channel, filters[0], kernel_size=kernels[0], stride=1, padding=paddings[0],
                             dilation=dilation[0])
        self.conv_bn_1 = nn.BatchNorm1d(filters[0])
        self.conv_2 = DCConv(filters[0], filters[1], kernel_size=kernels[1], stride=1, dilation=dilation[1],
                             padding=paddings[1])
        self.conv_bn_2 = nn.BatchNorm1d(filters[1])
        self.conv_3 = DCConv(filters[1], filters[2], kernel_size=kernels[2], stride=1, dilation=dilation[2],
                             padding=paddings[2])
        self.conv_bn_3 = nn.BatchNorm1d(filters[2])

        # recon
        self.recon1 = nn.Conv1d(1, ndim, kernel_size=recon_filter, stride=1, padding='same', dilation=1)
        self.recon1_bn_1 = nn.BatchNorm1d(ndim)
        self.recon2 = nn.Linear(layers[-1], len_window)  # Output size of mapping layer
        self.recon2_bn_2 = nn.BatchNorm1d(ndim)
        self.activation_2 = nn.LeakyReLU()

        # compute the size of input for fully connected layers
        fc_input = filters[2] + self.gru_dim + 2 * self.num_hidden  # 加了graph

        # Representation mapping function
        layers = [fc_input] + layers
        self.mapping = nn.Sequential()
        for i in range(len(layers) - 2):
            self.mapping.add_module("fc_" + str(i), nn.Linear(layers[i], layers[i + 1]))
            self.mapping.add_module("bn_" + str(i), nn.BatchNorm1d(layers[i + 1]))
            self.mapping.add_module("relu_" + str(i), nn.LeakyReLU())

        # add last layer
        self.mapping.add_module("fc_" + str(len(layers) - 2), nn.Linear(layers[-2], layers[-1]))
        if len(layers) == 2:  # if only one layer, add batch normalization
            self.mapping.add_module("bn_" + str(len(layers) - 2), nn.BatchNorm1d(layers[-1]))

        self.cvae = CVAE(feature_size=layers[-1], hidden_size=hidden_size, latent_size=latent_size, class_size=2)

    @staticmethod
    def output_conv_size(in_size, kernel_size, stride, padding):
        output = int((in_size - kernel_size + 2 * padding) / stride) + 1

        return output

    def forward(self, x, label):
        A = self.gat(x)
        threshold = self.threshold
        batch_size, n_feature, len_feature = x.shape[0], x.shape[1], x.shape[2]

        offset, row, col = (A > threshold).nonzero().t()
        row += offset * n_feature
        col += offset * n_feature
        edge_index = torch.stack([row, col], dim=0)
        x_gnn = x.contiguous().view(batch_size * n_feature, len_feature)  # 这里改动了长度
        batch = torch.arange(0, batch_size).view(-1, 1).repeat(1, n_feature).view(-1).to(self.device)

        x_gnn = F.relu(self.graph_conv1(x_gnn, edge_index))
        x_gnn, edge_index, _, batch, _, _ = self.graph_pool1(x_gnn, edge_index, None, batch)
        x_gnn_1 = torch.cat([torch_geometric.nn.global_max_pool(x_gnn, batch), torch_geometric.nn.global_mean_pool(
            x_gnn, batch)], dim=1)

        x_gnn = F.relu(self.graph_conv2(x_gnn, edge_index))
        x_gnn, edge_index, _, batch, _, _ = self.graph_pool2(x_gnn, edge_index, None, batch)
        x_gnn_2 = torch.cat([torch_geometric.nn.global_max_pool(x_gnn, batch), torch_geometric.nn.global_mean_pool(
            x_gnn, batch)], dim=1)

        x_gnn = F.relu(self.graph_conv3(x_gnn, edge_index))
        x_gnn, edge_index, _, batch, _, _ = self.graph_pool3(x_gnn, edge_index, None, batch)
        x_gnn_3 = torch.cat([torch_geometric.nn.global_max_pool(x_gnn, batch), torch_geometric.nn.global_mean_pool(
            x_gnn, batch)], dim=1)

        # (Batch, 2 * hidden)
        x_gnn = x_gnn_1 + x_gnn_2 + x_gnn_3

        # GRU
        x_gru = self.gru(x)[0]
        x_gru = x_gru.mean(1)
        x_gru = x_gru.view(batch_size, -1)

        # DC Conv
        x_conv = x
        x_conv = self.conv_1(x_conv)
        x_conv = self.conv_bn_1(x_conv)
        x_conv = F.leaky_relu(x_conv)

        x_conv = self.conv_2(x_conv)
        x_conv = self.conv_bn_2(x_conv)
        x_conv = F.leaky_relu(x_conv)

        x_conv = self.conv_3(x_conv)
        x_conv = self.conv_bn_3(x_conv)
        x_conv = F.leaky_relu(x_conv)

        x_conv = torch.mean(x_conv, 2)
        x = torch.cat([x_conv, x_gru], dim=1)
        x = torch.cat((x, x_gnn), 1)

        # linear mapping to low-dimensional space
        x_embed = self.mapping(x)

        # reconstruct the input
        x_recon = self.recon1(x_embed.unsqueeze(1))
        x_recon = self.recon1_bn_1(x_recon)
        x_recon = F.leaky_relu(x_recon)

        x_recon = self.recon2(x_recon)
        x_recon = self.recon2_bn_2(x_recon)
        x_recon = F.leaky_relu(x_recon)

        label = self.cvae.one_hot(label, 2, self.device)

        # Using CVAE to reconstruct
        recon_embed, mu, log_var = self.cvae(x_embed, label)

        return x_recon, recon_embed, x_embed, mu, log_var
