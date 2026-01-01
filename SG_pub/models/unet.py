import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups, out_channels)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups, out_channels)
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
    
    def forward(self, x):
        residual = self.shortcut(x)
        x = self.act(self.gn1(self.conv1(x)))
        x = self.gn2(self.conv2(x))
        return self.act(x + residual)

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, padding=1, bias=False)
        self.ps = nn.PixelShuffle(upscale_factor=2)
        self.gn = nn.GroupNorm(num_groups, out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.ps(x)
        return self.act(self.gn(x))


class UNet(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 channel_dims: list = [64, 128, 256, 512], 
                 num_groups=8):

        super().__init__()
        
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        current_in_c = in_channels

        for i, dim in enumerate(channel_dims):
            self.encoders.append(ResBlock(current_in_c, dim, num_groups))
            if i < len(channel_dims) - 1:
                self.downs.append(
                    nn.Conv2d(dim, channel_dims[i+1], kernel_size=4, stride=2, padding=1, bias=False)
                )
                current_in_c = channel_dims[i+1]
            else:
                self.bottleneck = ResBlock(dim, dim, num_groups)
                current_in_c = dim


        rev_dims = list(reversed(channel_dims))
        for i in range(len(rev_dims) - 1):
            in_c = rev_dims[i]
            out_c = rev_dims[i+1] 
            self.ups.append(UpBlock(in_c, out_c, num_groups))
            self.decoders.append(ResBlock(in_channels=out_c * 2, out_channels=out_c, num_groups=num_groups))


        self.out_conv = nn.Conv2d(channel_dims[0], out_channels, kernel_size=1)
        nn.init.zeros_(self.out_conv.weight)
        if self.out_conv.bias is not None: nn.init.zeros_(self.out_conv.bias)

    def forward(self, x):
        skips = [] 
        for i in range(len(self.encoders)):
            x = self.encoders[i](x)
            if i < len(self.encoders) - 1:
                skips.append(x)
                x = self.downs[i](x)

        x = self.bottleneck(x)
        for i in range(len(self.ups)):
            x = self.ups[i](x)
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            x = self.decoders[i](x)
        return self.out_conv(x)