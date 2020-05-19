import math
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F

class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


def drop_connect(inputs, p, training):
    """ Drop connect. """
    if not training: return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1 - p
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
    binary_tensor = torch.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    return output


def get_same_padding_conv2d(image_size=None):
     return partial(Conv2dStaticSamePadding, image_size=image_size)

def get_width_and_height_from_size(x):
    """ Obtains width and height from a int or tuple """
    if isinstance(x, int): return x, x
    if isinstance(x, list) or isinstance(x, tuple): return x
    else: raise TypeError()

def calculate_output_image_size(input_image_size, stride):
    """
    计算出 Conv2dSamePadding with a stride.
    """
    if input_image_size is None: return None
    image_height, image_width = get_width_and_height_from_size(input_image_size)
    stride = stride if isinstance(stride, int) else stride[0]
    image_height = int(math.ceil(image_height / stride))
    image_width = int(math.ceil(image_width / stride))
    return [image_height, image_width]



class Conv2dStaticSamePadding(nn.Conv2d):
    """ 2D Convolutions like TensorFlow, for a fixed image size"""

    def __init__(self, in_channels, out_channels, kernel_size, image_size=None, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

        # Calculate padding based on image size and save it
        assert image_size is not None
        ih, iw = (image_size, image_size) if isinstance(image_size, int) else image_size
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
        else:
            self.static_padding = Identity()

    def forward(self, x):
        x = self.static_padding(x)
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x

class Identity(nn.Module):
    def __init__(self, ):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


# MBConvBlock
class MBConvBlock(nn.Module):
    '''
    层 ksize3*3 输入32 输出16  conv1  stride步长1
    '''
    def __init__(self, ksize, input_filters, output_filters, expand_ratio, stride, image_size=None):
        super().__init__()
        self._bn_mom = 0.1
        self._bn_eps = 0.01
        self._se_ratio = 0.25
        self._input_filters = input_filters
        self._output_filters = output_filters
        self._expand_ratio = expand_ratio
        self._kernel_size = ksize
        self._stride = stride

        inp = self._input_filters
        oup = self._input_filters * self._expand_ratio
        if self._expand_ratio != 1:
            Conv2d = get_same_padding_conv2d(image_size=image_size)
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)


        # Depthwise convolution
        k = self._kernel_size
        s = self._stride
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,
            kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        image_size = calculate_output_image_size(image_size, s)

        # Squeeze and Excitation layer, if desired
        Conv2d = get_same_padding_conv2d(image_size=(1,1))
        num_squeezed_channels = max(1, int(self._input_filters * self._se_ratio))
        self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
        self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Output phase
        final_oup = self._output_filters
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate=None):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self._expand_ratio != 1:
            expand = self._expand_conv(inputs)
            bn0 = self._bn0(expand)
            x = self._swish(bn0)
        depthwise = self._depthwise_conv(x)
        bn1 = self._bn1(depthwise)
        x = self._swish(bn1)

        # Squeeze and Excitation
        x_squeezed = F.adaptive_avg_pool2d(x, 1)
        x_squeezed = self._se_reduce(x_squeezed)
        x_squeezed = self._swish(x_squeezed)
        x_squeezed = self._se_expand(x_squeezed)
        x = torch.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        # Skip connection and drop connect
        input_filters, output_filters = self._input_filters, self._output_filters
        if self._stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x

class EfficientNet(nn.Module):
    def __init__(self, cfgs, num_classes=1000, image_size=224):
        super().__init__()
        bn_mom = 0.01
        bn_eps = 0.001
        self.cfgs = cfgs

        # stem
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._conv_stem = Conv2d(3, 32, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=32, momentum=bn_mom, eps=bn_eps)

        # MBConv
        self._blocks = nn.ModuleList([])
        for expand, ksize, input_filters, output_filters,  stride, image_size in self.cfgs:
            self._blocks.append(MBConvBlock(ksize, input_filters, output_filters, expand, stride, image_size))

        # Head
        in_channels = self.cfgs[-1][3]
        out_channels = in_channels * 4
        image_size = self.cfgs[-1][-1]
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Final linear layer
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(0.2)
        self._fc = nn.Linear(out_channels, num_classes)
        self._swish = MemoryEfficientSwish()

    def _make_MBConv(self, inputs):
        x = self._swish(inputs)
        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = 0.2
            drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
        return x

    def forward(self, inputs):
        # Stem
        x = self._conv_stem(inputs)
        x = self._bn0(x)

        # Convolution layers
        x = self._make_MBConv(x)

        # Head
        x = self._conv_head(x)
        s = self._bn1(x)
        x = self._swish(x)

        # Pooling and final linear layer
        x = self._avg_pooling(x)
        x = x.view(inputs.size(0), -1)
        x = self._dropout(x)
        x = self._fc(x)
        return x

def EfficientNetB0(**kwargs):
    MBConv_cfgs = [
        # ksize, input, output, expand, stride, image_size
        # layers 1
        # MBConv1, k3*3, inputChannels, outputChannels, stride, Resolution
        [1, 3, 32, 16, 1, [112, 112]],

        # layers 2
        # MBConv6, k3*3, inputChannels, outputChannels, stride, Resolution
        [6, 3, 16, 24, 2, [112, 112]],
        [6, 3, 24, 24, 1, [56, 56]],

        # layers 2
        [6, 5, 24, 40, 2, [56, 56]],
        [6, 5, 40, 40, 1, [28, 28]],

        # layers 3
        [6, 3, 40, 80, 2, [28, 28]],
        [6, 3, 80, 80, 1, [14, 14]],
        [6, 3, 80, 80, 1, [14, 14]],

        # layers 3
        [6, 5, 80,  112, 1, [14, 14]],
        [6, 5, 112, 112, 1, [14, 14]],
        [6, 5, 112, 112, 1, [14, 14]],

        # layers 4
        [6, 5, 112, 192, 2, [14, 14]],
        [6, 5, 192, 192, 1, [7, 7]],
        [6, 5, 192, 192, 1, [7, 7]],
        [6, 5, 192, 192, 1, [7, 7]],

        # layers 1
        [6, 3, 192, 320, 1, [7, 7]],
    ]
    return EfficientNet(MBConv_cfgs, **kwargs)