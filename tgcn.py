import torch
import torch.nn as nn

class ConvTemporalGraphical(nn.Module):
    """
    The basic module for applying a graph convolution.

    Args:
        in_channels (int): Number of input feature channels.
        out_channels (int): Number of output feature channels.
        kernel_size (int): Graph convolution kernel size.
        t_kernel_size (int, optional): Temporal convolution kernel size. Default: 1.
        t_stride (int, optional): Temporal convolution stride. Default: 1.
        t_padding (int, optional): Temporal padding. Default: 0.
        t_dilation (int, optional): Temporal dilation. Default: 1.
        bias (bool, optional): Whether to use bias in convolution. Default: True.

    Shape:
        - Input[0]: (N, in_channels, T_in, V)
        - Input[1]: (K, V, V) adjacency matrix
        - Output[0]: (N, out_channels, T_out, V)
        - Output[1]: (K, V, V) adjacency matrix
    """

    def __init__(self, in_channels, out_channels, kernel_size, t_kernel_size=1, t_stride=1,
                 t_padding=0, t_dilation=1, bias=True):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels, out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            stride=(t_stride, 1),
            padding=(t_padding, 0),
            dilation=(t_dilation, 1),
            bias=bias
        )

    def forward(self, x, A):
        """
        Forward pass of the graph convolutional layer.

        Args:
            x (torch.Tensor): Input feature tensor of shape (N, in_channels, T, V).
            A (torch.Tensor): Adjacency matrix of shape (K, V, V).

        Returns:
            torch.Tensor: Output feature tensor of shape (N, out_channels, T, V).
            torch.Tensor: Adjacency matrix of shape (K, V, V).
        """
        assert A.size(0) == self.kernel_size, "Adjacency matrix size does not match kernel size."

        # 1. Graph convolution
        x = self.conv(x)  # Shape: (N, out_channels * K, T, V)
        N, KC, T, V = x.size()
        x = x.view(N, self.kernel_size, KC // self.kernel_size, T, V)  # Reshape: (N, K, C, T, V)

        # 2. Apply adjacency matrix transformation
        x = torch.einsum('nkctv,kvw->nctw', (x, A))  # Alternative: torch.matmul(x, A.transpose(-2, -1))

        return x.contiguous(), A