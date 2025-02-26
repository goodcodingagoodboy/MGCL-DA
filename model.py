import torch
import torch.nn as nn
import numpy as np
import os
from net.tgcn import ConvTemporalGraphical
from net.CrossSampleAugmented import CrossSampleAugmented
from net.SelfAwareAugmented import SelfAwareAugmented
from net.DynamicUpdate import DynamicUpdate


class Model(nn.Module):
    r"""Spatio-Temporal Graph Convolutional Network (ST-GCN)

    Args:
        in_channels (int): Number of input channels
        num_class (int): Number of classes for classification tasks
        edge_importance_weighting (bool): If ``True``, add learnable edge importance weighting
        **kwargs (optional): Additional parameters for other graph convolution units
    """

    def __init__(self, in_channels, num_class, edge_importance_weighting, root_path, **kwargs) -> object:
        super().__init__()
        self.root_path = root_path
        # Load the adjacency matrix (A)
        A = np.load(os.path.join(self.root_path, 'adj_matrix.npy'))

        # Compute the normalized Laplacian matrix for the graph
        Dl = np.sum(A, 0)  # Degree matrix of the nodes
        num_node = A.shape[0]  # Number of nodes
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i] ** (-0.5)  # Standardize the Laplacian matrix
        DAD = np.dot(np.dot(Dn, A), Dn)  # Get the normalized adjacency matrix DAD

        # Convert adjacency matrix to tensor and register it as a buffer
        temp_matrix = np.zeros((1, A.shape[0], A.shape[0]))
        temp_matrix[0] = DAD
        A = torch.tensor(temp_matrix, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)  # Register A as a buffer to prevent gradient update

        # Build the ST-GCN network
        spatial_kernel_size = A.size(0)  # Size of the spatial convolution kernel
        temporal_kernel_size = 11  # Size of the temporal convolution kernel
        kernel_size = (temporal_kernel_size, spatial_kernel_size)  # Convolution kernel size
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))  # Batch normalization
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}  # Extract other parameters

        # Define three sets of networks, each using a different attention mechanism
        self.st_gcn_networks = nn.ModuleList([
            nn.ModuleList([
                st_gcn(in_channels, 16, kernel_size, 1, residual=False, attention_type='cross_sample', **kwargs0),
                st_gcn(16, 16, kernel_size, 1, residual=False, attention_type='cross_sample', **kwargs),
                st_gcn(16, 16, kernel_size, 1, residual=False, attention_type='cross_sample', **kwargs),
            ]),
            nn.ModuleList([
                st_gcn(in_channels, 16, kernel_size, 1, residual=False, attention_type='original', **kwargs0),
                st_gcn(16, 16, kernel_size, 1, residual=False, attention_type='original', **kwargs),
                st_gcn(16, 16, kernel_size, 1, residual=False, attention_type='original', **kwargs),
            ]),
            nn.ModuleList([
                st_gcn(in_channels, 16, kernel_size, 1, residual=False, attention_type='self_aware', **kwargs0),
                st_gcn(16, 16, kernel_size, 1, residual=False, attention_type='self_aware', **kwargs),
                st_gcn(16, 16, kernel_size, 1, residual=False, attention_type='self_aware', **kwargs),
            ])
        ])

        # Initialize edge importance weights
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # Define classification layers
        self.cls_fcn1 = nn.Conv2d(3200, 1024, kernel_size=1)
        self.cls_fcn2 = nn.Conv2d(1024, 512, kernel_size=1)
        self.cls_fcn3 = nn.Conv2d(512, 64, kernel_size=1)
        self.cls_fcn4 = nn.Conv2d(64, num_class, kernel_size=1)
        self.sig = nn.Sigmoid()

        # Initialize dynamic updater
        self.dynamic_updater = DynamicUpdate()
        # Store the current views
        self.current_SA = None
        self.current_CA = None

    def forward(self, source):
        N, C, T, V, M = source.size()
        source = source.permute(0, 4, 3, 1, 2).contiguous()
        source = source.view(N * M, V * C, T)
        source = self.data_bn(source.float())
        source = source.view(N, M, V, C, T)
        source = source.permute(0, 1, 3, 4, 2).contiguous()
        source = source.view(N * M, C, T, V)

        # Pass through the three different GCN network branches
        group_outputs = []
        for gcn_group, importance in zip(self.st_gcn_networks, self.edge_importance):
            group_output = source
            for gcn in gcn_group:
                group_output, _ = gcn(group_output, self.A * importance)
            group_outputs.append(group_output)

        # If saved views exist, use them
        if self.training and self.current_SA is not None and self.current_CA is not None:
            group_outputs[2] = self.current_SA
            group_outputs[0] = self.current_CA

        # Use the output from the original attention branch for classification
        output = group_outputs[1]

        # Process the output
        output = output.mean(dim=3)
        output = output.view(output.size(0), -1, 1, 1)

        # Pass through classification layers
        target_1 = self.cls_fcn1(output)
        target_2 = self.cls_fcn2(target_1)
        target_3 = self.cls_fcn3(target_2)
        target_4 = self.cls_fcn4(target_3)

        return target_4.view(target_4.size(0), -1), group_outputs


class st_gcn(nn.Module):
    r"""Apply spatio-temporal graph convolution on input graph sequences"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0.5,
                 residual=True,
                 attention_type='cross_sample'):
        super().__init__()
        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        # Graph convolution
        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])

        # Temporal convolution
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        # Attention mechanism
        if attention_type == 'cross_sample':
            self.attention = CrossSampleAugmented()
        elif attention_type == 'original':
            self.attention = None
        elif attention_type == 'self_aware':
            self.attention = SelfAwareAugmented(1)

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x)

        if self.attention is not None:
            x = x.permute(0, 2, 1, 3).contiguous()
            x = self.attention(x)
            x = x.permute(0, 2, 1, 3).contiguous()

        x = x + res
        return self.relu(x), A