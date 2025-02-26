import torch
import torch.nn as nn

# Define the basic convolutional block
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


# Define the ZPool pooling module
class ZPool(nn.Module):
    def forward(self, x):
        a = torch.max(x, 1, keepdim=True)[0]  # Keep channel dimension
        b = torch.mean(x, 1, keepdim=True)  # Keep channel dimension
        c = torch.cat((a, b), dim=1)
        return c


# Define the attention gate module
class AttentionGate(nn.Module):
    def __init__(self, in_channels):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale


# Define the cross-sample augmented module
class CrossSampleAugmented(nn.Module):
    def __init__(self, no_spatial=False, use_learnable_weights=True):
        super(CrossSampleAugmented, self).__init__()
        self.hc = AttentionGate(in_channels=2)
        self.no_spatial = no_spatial
        self.use_learnable_weights = use_learnable_weights
        if self.use_learnable_weights:
            self.weight_hc = nn.Parameter(torch.ones(1))  # Weight for height-channel interaction
            self.weight_hw = nn.Parameter(torch.ones(1))  # Weight for height-width interaction
        if not no_spatial:
            self.hw = AttentionGate(in_channels=2)

    def forward(self, x):
        # (B, C, H, W) -> (B, W, H, C), enabling interaction between height (H) and channels (C)
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        # Pooling, concatenation, convolution, sigmoid activation, and weighting along the width dimension
        x_out2 = self.hc(x_perm2)
        # Restore to original dimensions: (B, W, H, C) -> (B, C, H, W)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()

        # If spatial attention is required, perform height-width interaction
        if not self.no_spatial:
            x_out_hw = self.hw(x)
            # Fuse the two attention results using learnable weights
            if self.use_learnable_weights:
                x_out = self.weight_hc * x_out21 + self.weight_hw * x_out_hw
            else:
                x_out = 1 / 2 * (x_out_hw + x_out21)
        else:
            x_out = x_out21

        return x_out


# Test the model
if __name__ == '__main__':
    # Create a random input tensor with shape (batch_size, channels, height, width)
    input = torch.randn(1, 512, 7, 7)
    # Instantiate the model
    model = CrossSampleAugmented()
    # Forward pass
    output = model(input)
    # Print the output shape
    print(f"CrossSampleAugmented output shape: {output.shape}")  # Expected output: torch.Size([1, 512, 7, 7])
