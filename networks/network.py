import torch
import torch.nn.functional as F
from torch import nn


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=False)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=False)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=False)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)
        
class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, nf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            nf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(nf * 8, nf * 8, input_nc=None, submodule=None, innermost=True)  # add the innermost layer
        
        # add intermediate layers with ngf * 8 filters
        unet_block = UnetSkipConnectionBlock(nf * 8, nf * 8, input_nc=None, submodule=unet_block, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(nf * 8, nf * 8, input_nc=None, submodule=unet_block, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(nf * 8, nf * 8, input_nc=None, submodule=unet_block, use_dropout=use_dropout)
        
        # gradually reduce the number of filters from nf * 8 to nf
        unet_block = UnetSkipConnectionBlock(nf * 4, nf * 8, input_nc=None, submodule=unet_block)
        unet_block = UnetSkipConnectionBlock(nf * 2, nf * 4, input_nc=None, submodule=unet_block)
        unet_block = UnetSkipConnectionBlock(nf, nf * 2, input_nc=None, submodule=unet_block)
        self.model = UnetSkipConnectionBlock(output_nc, nf, input_nc=input_nc, submodule=unet_block, outermost=True)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)

class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=32, depth=5):
        super(UNet, self).__init__()
        self.depth = depth  # Variable depth
        self.down_convs = nn.ModuleList()  # Encoder layers
        self.up_convs = nn.ModuleList()  # Decoder layers
        self.trans_convs = nn.ModuleList()  # Transposed convolutions (upsampling)
        i = 3 # Initial exponent of channel size

        # Encoder (Contracting Path)
        for d in range(depth):
            input_channels = in_channels if d == 0 else 2**(i + d - 1)
            output_channels = 2**(i + d)
            self.down_convs.append(self.conv_block(input_channels, output_channels))

        # Bottleneck
        self.bottleneck = self.conv_block(2**(i + depth - 1), 2**(i + depth))

        # Decoder (Expanding Path)
        for d in range(depth, 0, -1):
            input_channels = 2**(i + d)
            output_channels = 2**(i + d - 1)
            self.trans_convs.append(nn.ConvTranspose2d(input_channels, output_channels, kernel_size=2, stride=2))
            self.up_convs.append(self.conv_block(input_channels, output_channels))

        # Output layer
        self.out_conv = nn.Conv2d(2**i, num_classes, kernel_size=1)

    def forward(self, x):
        enc_outputs = []  # To store outputs from each encoder layer

        # Encoder
        for enc in self.down_convs:
            x = enc(x)
            enc_outputs.append(x)
            x = F.max_pool2d(x, 2)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for idx in range(self.depth):
            x = self.trans_convs[idx](x)
            x = self.crop_and_concat(x, enc_outputs[-(idx+1)])
            x = self.up_convs[idx](x)

        # Ensure the final output has the same size as input
        if x.size(2) != enc_outputs[0].size(2):
            x = F.interpolate(x, size=(enc_outputs[0].size(2), enc_outputs[0].size(3)), mode='bilinear', align_corners=False)


        return self.out_conv(x)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def crop_and_concat(self, upsampled, bypass):
        crop_h = (bypass.size(2) - upsampled.size(2)) // 2
        crop_w = (bypass.size(3) - upsampled.size(3)) // 2
        bypass = bypass[:, :, crop_h:crop_h + upsampled.size(2), crop_w:crop_w + upsampled.size(3)]
        return torch.cat((upsampled, bypass), dim=1)