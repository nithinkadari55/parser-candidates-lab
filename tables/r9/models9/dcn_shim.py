"""Drop-in for LORE-TSR's DCNv2/dcn_v2.py on CPU: the same DCN module (parameters weight, bias, conv_offset_mask)
computed with torchvision.ops.deform_conv2d (modulated deformable convolution, DCNv2 offset layout)."""
import math
import torch
from torch import nn
from torch.nn.modules.utils import _pair
from torchvision.ops import deform_conv2d


class DCNv2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, deformable_groups=1):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size, self.stride, self.padding, self.dilation = _pair(kernel_size), _pair(stride), _pair(padding), _pair(dilation)
        self.deformable_groups = deformable_groups
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        n = in_channels
        for k in self.kernel_size: n *= k
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv); self.bias.data.zero_()

    def forward(self, input, offset, mask):
        return deform_conv2d(input, offset, self.weight, self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, mask=mask)


class DCN(DCNv2):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, deformable_groups=1):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, deformable_groups)
        ch = self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1]
        self.conv_offset_mask = nn.Conv2d(self.in_channels, ch, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=True)
        self.conv_offset_mask.weight.data.zero_(); self.conv_offset_mask.bias.data.zero_()

    def forward(self, input):
        out = self.conv_offset_mask(input)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return deform_conv2d(input, offset, self.weight, self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, mask=mask)
