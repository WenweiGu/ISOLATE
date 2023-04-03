import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """Graph Attention Layer
    :param n_features: Number of input features/nodes
    :param len_features: length of the input sequence
    :param dropout: percentage of nodes to dropout
    :param alpha: negative slope used in the leaky rely on activation function
    :param embed_dim: embedding dimension (output dimension of linear transformation)
    :param use_gatv2: whether to use the modified attention mechanism of GATv2 instead of standard GAT
    :param use_bias: whether to include a bias term in the attention layer
    """

    def __init__(self, n_features, len_features, dropout=0.5, alpha=0.2, embed_dim=None, use_gatv2=True,
                 use_bias=True):
        super(GraphAttentionLayer, self).__init__()
        self.n_features = n_features
        self.len_features = len_features
        self.dropout = dropout
        self.embed_dim = embed_dim if embed_dim is not None else len_features
        self.use_gatv2 = use_gatv2
        self.num_nodes = n_features
        self.use_bias = use_bias

        # Because linear transformation is done after concatenation in GATv2
        if self.use_gatv2:
            self.embed_dim *= 2
            lin_input_dim = 2 * len_features
            a_input_dim = self.embed_dim
        else:
            lin_input_dim = len_features
            a_input_dim = 2 * self.embed_dim

        self.lin = nn.Linear(lin_input_dim, self.embed_dim)
        self.a = nn.Parameter(torch.empty((a_input_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        if self.use_bias:
            self.bias = nn.Parameter(torch.empty(n_features, n_features))

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x (N, D, L): N - sample size, D - feature dim, L - length of sequence
        # For feature attention we represent a node as the values of a particular feature across all timestamps

        # 'Dynamic' GAT attention
        # Proposed by Brody et. al., 2021 (https://arxiv.org/pdf/2105.14491.pdf)
        # Linear transformation applied after concatenation and attention layer applied after leakyrelu
        if self.use_gatv2:
            a_input = self._make_attention_input(x)                 # (N, D, D, 2*L)
            a_input = self.leakyrelu(self.lin(a_input))             # (N, D, D, embed_dim)
            e = torch.matmul(a_input, self.a).squeeze(3)            # (N, D, D, 1)

        # Original GAT attention
        else:
            Wx = self.lin(x)                                                  # (N, D, D, embed_dim)
            a_input = self._make_attention_input(Wx)                          # (N, D, D, 2*embed_dim)
            e = self.leakyrelu(torch.matmul(a_input, self.a)).squeeze(3)      # (N, D, D, 1)

        if self.use_bias:
            e += self.bias

        # Attention weights
        attention = F.softmax(e, dim=2)
        # attention = torch.dropout(attention, self.dropout, train=self.training)

        # Computing new node features using the attention
        # h = self.sigmoid(torch.matmul(attention, x))

        return attention

    def _make_attention_input(self, v):
        """Preparing the feature attention mechanism.
        Creating matrix with all possible combinations of concatenations of node.
        Each node consists of all values of that node within the window
            v1 || v1,
            ...
            v1 || vD,
            v2 || v1,
            ...
            v2 || vD,
            ...
            ...
            vD || v1,
            ...
            vD || vD,
        """

        D = self.num_nodes
        blocks_repeating = v.repeat_interleave(D, dim=1)  # Left-side of the matrix
        blocks_alternating = v.repeat(1, D, 1)  # Right-side of the matrix
        combined = torch.cat((blocks_repeating, blocks_alternating), dim=2)  # (N, D*D, 2*L)

        if self.use_gatv2:
            return combined.view(v.size(0), D, D, 2 * self.len_features)
        else:
            return combined.view(v.size(0), D, D, 2 * self.embed_dim)
