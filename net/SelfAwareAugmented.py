import torch
import torch.nn as nn

# Define a basic convolutional block
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


# Define the ZPool module for feature compression
class ZPool(nn.Module):
    def forward(self, x):
        a = torch.max(x, 1, keepdim=True)[0]  # Preserve channel dimension
        b = torch.mean(x, 1, keepdim=True)  # Preserve channel dimension
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


# Define the self-aware augmentation module
class SelfAwareAugmented(nn.Module):
    def __init__(self, in_channels, no_spatial=False):
        super(SelfAwareAugmented, self).__init__()
        self.hc = AttentionGate(in_channels=in_channels)

    def forward(self, x):
        # (B, C, H, W) -> (C, B, H, W), establishing interactions between batch B and channel C
        x_perm = x.permute(1, 0, 2, 3).contiguous()
        # Apply pooling, concatenation, convolution, sigmoid activation, and weighting along the B dimension
        x_out = self.hc(x_perm)
        # Restore original dimensions: (C, B, H, W) -> (B, C, H, W)
        x_out = x_out.permute(1, 0, 2, 3).contiguous()
        return x_out


# Test the model
if __name__ == '__main__':
    # Create a random input tensor with shape (batch_size, channels, height, width)
    input = torch.randn(1, 512, 7, 7)
    # Instantiate the model with the input channel count
    model = SelfAwareAugmented(in_channels=512)
    # Forward pass
    output = model(input)
    # Print the output shape
    print(f"SelfAwareAugmented Output Shape: {output.shape}")  # Expected: torch.Size([1, 512, 7, 7])
