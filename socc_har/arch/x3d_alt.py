import math
from pathlib import Path
from typing import Optional
import torch
from fvcore.nn.weight_init import c2_msra_fill
from fvcore.common.config import CfgNode
from torch import nn

from socc_har.util.fetch import Fetcher

_POOL1 = {
    "x3d": [[1, 1, 1]],
}

# Number of blocks for different stages given the model depth.
_MODEL_STAGE_DEPTH = {50: (3, 4, 6, 3), 101: (3, 4, 23, 3)}

# Basis of temporal kernel sizes for each of the stage.
_TEMPORAL_KERNEL_BASIS = {
    "x3d": [
        [[5]],  # conv1 temporal kernels.
        [[3]],  # res2 temporal kernels.
        [[3]],  # res3 temporal kernels.
        [[3]],  # res4 temporal kernels.
        [[3]],  # res5 temporal kernels.
    ],
}


class X3DStem(nn.Module):
    """
    X3D's 3D stem module.
    Performs a spatial followed by a depthwise temporal Convolution, BN, and Relu following by a
        spatiotemporal pooling.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        kernel,
        stride,
        padding,
        inplace_relu=True,
        eps=1e-5,
        bn_mmt=0.1,
        norm_module=nn.BatchNorm3d,
    ):
        """
        The `__init__` method of any subclass should also contain these arguments.
        Args:
            dim_in (int): the channel dimension of the input. Normally 3 is used
                for rgb input, and 2 or 3 is used for optical flow input.
            dim_out (int): the output dimension of the convolution in the stem
                layer.
            kernel (list): the kernel size of the convolution in the stem layer.
                temporal kernel size, height kernel size, width kernel size in
                order.
            stride (list): the stride size of the convolution in the stem layer.
                temporal kernel stride, height kernel size, width kernel size in
                order.
            padding (int): the padding size of the convolution in the stem
                layer, temporal padding size, height padding size, width
                padding size in order.
            inplace_relu (bool): calculate the relu on the original input
                without allocating new memory.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
        """
        super(X3DStem, self).__init__()
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.inplace_relu = inplace_relu
        self.eps = eps
        self.bn_mmt = bn_mmt
        # Construct the stem layer.
        self._construct_stem(dim_in, dim_out, norm_module)

    def _construct_stem(self, dim_in, dim_out, norm_module):
        self.conv_xy = nn.Conv3d(
            dim_in,
            dim_out,
            kernel_size=(1, self.kernel[1], self.kernel[2]),
            stride=(1, self.stride[1], self.stride[2]),
            padding=(0, self.padding[1], self.padding[2]),
            bias=False,
        )
        self.conv = nn.Conv3d(
            dim_out,
            dim_out,
            kernel_size=(self.kernel[0], 1, 1),
            stride=(self.stride[0], 1, 1),
            padding=(self.padding[0], 0, 0),
            bias=False,
            groups=dim_out,
        )

        self.bn = norm_module(
            num_features=dim_out, eps=self.eps, momentum=self.bn_mmt
        )
        self.relu = nn.ReLU(self.inplace_relu)

    def forward(self, x):
        x = self.conv_xy(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class VideoModelStem(nn.Module):
    """
    Video 3D stem module. Provides stem operations of Conv, BN, ReLU, MaxPool
    on input data tensor for one or multiple pathways.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        kernel,
        stride,
        padding,
        inplace_relu=True,
        eps=1e-5,
        bn_mmt=0.1,
        norm_module=nn.BatchNorm3d,
        stem_func_name="basic_stem",
    ):
        """
        The `__init__` method of any subclass should also contain these
        arguments. List size of 1 for single pathway models (C2D, I3D, Slow
        and etc), list size of 2 for two pathway models (SlowFast).
        Args:
            dim_in (list): the list of channel dimensions of the inputs.
            dim_out (list): the output dimension of the convolution in the stem
                layer.
            kernel (list): the kernels' size of the convolutions in the stem
                layers. Temporal kernel size, height kernel size, width kernel
                size in order.
            stride (list): the stride sizes of the convolutions in the stem
                layer. Temporal kernel stride, height kernel size, width kernel
                size in order.
            padding (list): the paddings' sizes of the convolutions in the stem
                layer. Temporal padding size, height padding size, width padding
                size in order.
            inplace_relu (bool): calculate the relu on the original input
                without allocating new memory.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
            stem_func_name (string): name of the the stem function applied on
                input to the network.
        """
        super(VideoModelStem, self).__init__()

        assert (
            len(
                {
                    len(dim_in),
                    len(dim_out),
                    len(kernel),
                    len(stride),
                    len(padding),
                }
            )
            == 1
        ), "Input pathway dimensions are not consistent."
        self.num_pathways = len(dim_in)
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.inplace_relu = inplace_relu
        self.eps = eps
        self.bn_mmt = bn_mmt
        # Construct the stem layer.
        self._construct_stem(dim_in, dim_out, norm_module, stem_func_name)

    def _construct_stem(self, dim_in, dim_out, norm_module, stem_func_name):
        trans_func = self.get_stem_func(stem_func_name)

        for pathway in range(len(dim_in)):
            stem = trans_func(
                dim_in[pathway],
                dim_out[pathway],
                self.kernel[pathway],
                self.stride[pathway],
                self.padding[pathway],
                self.inplace_relu,
                self.eps,
                self.bn_mmt,
                norm_module,
            )
            self.add_module("pathway{}_stem".format(pathway), stem)

    def forward(self, x):
        assert (
            len(x) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)
        for pathway in range(len(x)):
            m = getattr(self, "pathway{}_stem".format(pathway))
            x[pathway] = m(x[pathway])
        return x

    def get_stem_func(self, name):
        """
        Retrieves the stem module by name.
        """
        trans_funcs = {"x3d_stem": X3DStem}
        assert (
                name in trans_funcs.keys()
        ), "Transformation function '{}' not supported".format(name)
        return trans_funcs[name]


class Swish(nn.Module):
    """Swish activation function: x * sigmoid(x)."""

    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return SwishEfficient.apply(x)


class SwishEfficient(torch.autograd.Function):
    """Swish activation function: x * sigmoid(x)."""

    @staticmethod
    def forward(ctx, x):
        result = x * torch.sigmoid(x)
        ctx.save_for_backward(x)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_variables[0]
        sigmoid_x = torch.sigmoid(x)
        return grad_output * (sigmoid_x * (1 + x * (1 - sigmoid_x)))


class SE(nn.Module):
    """Squeeze-and-Excitation (SE) block w/ Swish: AvgPool, FC, Swish, FC, Sigmoid."""

    def _round_width(self, width, multiplier, min_width=8, divisor=8):
        """
        Round width of filters based on width multiplier
        Args:
            width (int): the channel dimensions of the input.
            multiplier (float): the multiplication factor.
            min_width (int): the minimum width after multiplication.
            divisor (int): the new width should be dividable by divisor.
        """
        if not multiplier:
            return width

        width *= multiplier
        min_width = min_width or divisor
        width_out = max(
            min_width, int(width + divisor / 2) // divisor * divisor
        )
        if width_out < 0.9 * width:
            width_out += divisor
        return int(width_out)

    def __init__(self, dim_in, ratio, relu_act=True):
        """
        Args:
            dim_in (int): the channel dimensions of the input.
            ratio (float): the channel reduction ratio for squeeze.
            relu_act (bool): whether to use ReLU activation instead
                of Swish (default).
            divisor (int): the new width should be dividable by divisor.
        """
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        dim_fc = self._round_width(dim_in, ratio)
        self.fc1 = nn.Conv3d(dim_in, dim_fc, 1, bias=True)
        self.fc1_act = nn.ReLU() if relu_act else Swish()
        self.fc2 = nn.Conv3d(dim_fc, dim_in, 1, bias=True)

        self.fc2_sig = nn.Sigmoid()

    def forward(self, x):
        x_in = x
        for module in self.children():
            x = module(x)
        return x_in * x


class X3DTransform(nn.Module):
    """
    X3D transformation: 1x1x1, Tx3x3 (channelwise, num_groups=dim_in), 1x1x1,
        augmented with (optional) SE (squeeze-excitation) on the 3x3x3 output.
        T is the temporal kernel size (defaulting to 3)
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        temp_kernel_size,
        stride,
        dim_inner,
        num_groups,
        stride_1x1=False,
        inplace_relu=True,
        eps=1e-5,
        bn_mmt=0.1,
        dilation=1,
        norm_module=nn.BatchNorm3d,
        se_ratio=0.0625,
        swish_inner=True,
        block_idx=0,
    ):
        """
        Args:
            dim_in (int): the channel dimensions of the input.
            dim_out (int): the channel dimension of the output.
            temp_kernel_size (int): the temporal kernel sizes of the middle
                convolution in the bottleneck.
            stride (int): the stride of the bottleneck.
            dim_inner (int): the inner dimension of the block.
            num_groups (int): number of groups for the convolution. num_groups=1
                is for standard ResNet like networks, and num_groups>1 is for
                ResNeXt like networks.
            stride_1x1 (bool): if True, apply stride to 1x1 conv, otherwise
                apply stride to the 3x3 conv.
            inplace_relu (bool): if True, calculate the relu on the original
                input without allocating new memory.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            dilation (int): size of dilation.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
            se_ratio (float): if > 0, apply SE to the Tx3x3 conv, with the SE
                channel dimensionality being se_ratio times the Tx3x3 conv dim.
            swish_inner (bool): if True, apply swish to the Tx3x3 conv, otherwise
                apply ReLU to the Tx3x3 conv.
        """
        super(X3DTransform, self).__init__()
        self.temp_kernel_size = temp_kernel_size
        self._inplace_relu = inplace_relu
        self._eps = eps
        self._bn_mmt = bn_mmt
        self._se_ratio = se_ratio
        self._swish_inner = swish_inner
        self._stride_1x1 = stride_1x1
        self._block_idx = block_idx
        self._construct(
            dim_in,
            dim_out,
            stride,
            dim_inner,
            num_groups,
            dilation,
            norm_module,
        )

    def _construct(
        self,
        dim_in,
        dim_out,
        stride,
        dim_inner,
        num_groups,
        dilation,
        norm_module,
    ):
        (str1x1, str3x3) = (stride, 1) if self._stride_1x1 else (1, stride)

        # 1x1x1, BN, ReLU.
        self.a = nn.Conv3d(
            dim_in,
            dim_inner,
            kernel_size=[1, 1, 1],
            stride=[1, str1x1, str1x1],
            padding=[0, 0, 0],
            bias=False,
        )
        self.a_bn = norm_module(
            num_features=dim_inner, eps=self._eps, momentum=self._bn_mmt
        )
        self.a_relu = nn.ReLU(inplace=self._inplace_relu)

        # Tx3x3, BN, ReLU.
        self.b = nn.Conv3d(
            dim_inner,
            dim_inner,
            [self.temp_kernel_size, 3, 3],
            stride=[1, str3x3, str3x3],
            padding=[int(self.temp_kernel_size // 2), dilation, dilation],
            groups=num_groups,
            bias=False,
            dilation=[1, dilation, dilation],
        )
        self.b_bn = norm_module(
            num_features=dim_inner, eps=self._eps, momentum=self._bn_mmt
        )

        # Apply SE attention or not
        use_se = True if (self._block_idx + 1) % 2 else False
        if self._se_ratio > 0.0 and use_se:
            self.se = SE(dim_inner, self._se_ratio)

        if self._swish_inner:
            self.b_relu = Swish()
        else:
            self.b_relu = nn.ReLU(inplace=self._inplace_relu)

        # 1x1x1, BN.
        self.c = nn.Conv3d(
            dim_inner,
            dim_out,
            kernel_size=[1, 1, 1],
            stride=[1, 1, 1],
            padding=[0, 0, 0],
            bias=False,
        )
        self.c_bn = norm_module(
            num_features=dim_out, eps=self._eps, momentum=self._bn_mmt
        )
        self.c_bn.transform_final_bn = True

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x


class ResBlock(nn.Module):
    """
    Residual block.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        temp_kernel_size,
        stride,
        trans_func,
        dim_inner,
        num_groups=1,
        stride_1x1=False,
        inplace_relu=True,
        eps=1e-5,
        bn_mmt=0.1,
        dilation=1,
        norm_module=nn.BatchNorm3d,
        block_idx=0,
        drop_connect_rate=0.0,
    ):
        """
        ResBlock class constructs redisual blocks. More details can be found in:
            Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
            "Deep residual learning for image recognition."
            https://arxiv.org/abs/1512.03385
        Args:
            dim_in (int): the channel dimensions of the input.
            dim_out (int): the channel dimension of the output.
            temp_kernel_size (int): the temporal kernel sizes of the middle
                convolution in the bottleneck.
            stride (int): the stride of the bottleneck.
            trans_func (string): transform function to be used to construct the
                bottleneck.
            dim_inner (int): the inner dimension of the block.
            num_groups (int): number of groups for the convolution. num_groups=1
                is for standard ResNet like networks, and num_groups>1 is for
                ResNeXt like networks.
            stride_1x1 (bool): if True, apply stride to 1x1 conv, otherwise
                apply stride to the 3x3 conv.
            inplace_relu (bool): calculate the relu on the original input
                without allocating new memory.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            dilation (int): size of dilation.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
            drop_connect_rate (float): basic rate at which blocks are dropped,
                linearly increases from input to output blocks.
        """
        super(ResBlock, self).__init__()
        self._inplace_relu = inplace_relu
        self._eps = eps
        self._bn_mmt = bn_mmt
        self._drop_connect_rate = drop_connect_rate
        self._construct(
            dim_in,
            dim_out,
            temp_kernel_size,
            stride,
            trans_func,
            dim_inner,
            num_groups,
            stride_1x1,
            inplace_relu,
            dilation,
            norm_module,
            block_idx,
        )

    def _construct(
        self,
        dim_in,
        dim_out,
        temp_kernel_size,
        stride,
        trans_func,
        dim_inner,
        num_groups,
        stride_1x1,
        inplace_relu,
        dilation,
        norm_module,
        block_idx,
    ):
        # Use skip connection with projection if dim or res change.
        if (dim_in != dim_out) or (stride != 1):
            self.branch1 = nn.Conv3d(
                dim_in,
                dim_out,
                kernel_size=1,
                stride=[1, stride, stride],
                padding=0,
                bias=False,
                dilation=1,
            )
            self.branch1_bn = norm_module(
                num_features=dim_out, eps=self._eps, momentum=self._bn_mmt
            )
        self.branch2 = trans_func(
            dim_in,
            dim_out,
            temp_kernel_size,
            stride,
            dim_inner,
            num_groups,
            stride_1x1=stride_1x1,
            inplace_relu=inplace_relu,
            dilation=dilation,
            norm_module=norm_module,
            block_idx=block_idx,
        )
        self.relu = nn.ReLU(self._inplace_relu)

    def _drop_connect(self, x, drop_ratio):
        """Apply dropconnect to x"""
        keep_ratio = 1.0 - drop_ratio
        mask = torch.empty(
            [x.shape[0], 1, 1, 1, 1], dtype=x.dtype, device=x.device
        )
        mask.bernoulli_(keep_ratio)
        x.div_(keep_ratio)
        x.mul_(mask)
        return x

    def forward(self, x):
        f_x = self.branch2(x)
        if self.training and self._drop_connect_rate > 0.0:
            f_x = self._drop_connect(f_x, self._drop_connect_rate)
        if hasattr(self, "branch1"):
            x = self.branch1_bn(self.branch1(x)) + f_x
        else:
            x = x + f_x
        x = self.relu(x)
        return x


class Nonlocal(nn.Module):
    """
    Builds Non-local Neural Networks as a generic family of building
    blocks for capturing long-range dependencies. Non-local Network
    computes the response at a position as a weighted sum of the
    features at all positions. This building block can be plugged into
    many computer vision architectures.
    More details in the paper: https://arxiv.org/pdf/1711.07971.pdf
    """

    def __init__(
        self,
        dim,
        dim_inner,
        pool_size=None,
        instantiation="softmax",
        zero_init_final_conv=False,
        zero_init_final_norm=True,
        norm_eps=1e-5,
        norm_momentum=0.1,
        norm_module=nn.BatchNorm3d,
    ):
        """
        Args:
            dim (int): number of dimension for the input.
            dim_inner (int): number of dimension inside of the Non-local block.
            pool_size (list): the kernel size of spatial temporal pooling,
                temporal pool kernel size, spatial pool kernel size, spatial
                pool kernel size in order. By default pool_size is None,
                then there would be no pooling used.
            instantiation (string): supports two different instantiation method:
                "dot_product": normalizing correlation matrix with L2.
                "softmax": normalizing correlation matrix with Softmax.
            zero_init_final_conv (bool): If true, zero initializing the final
                convolution of the Non-local block.
            zero_init_final_norm (bool):
                If true, zero initializing the final batch norm of the Non-local
                block.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
        """
        super(Nonlocal, self).__init__()
        self.dim = dim
        self.dim_inner = dim_inner
        self.pool_size = pool_size
        self.instantiation = instantiation
        self.use_pool = (
            False
            if pool_size is None
            else any((size > 1 for size in pool_size))
        )
        self.norm_eps = norm_eps
        self.norm_momentum = norm_momentum
        self._construct_nonlocal(
            zero_init_final_conv, zero_init_final_norm, norm_module
        )

    def _construct_nonlocal(
        self, zero_init_final_conv, zero_init_final_norm, norm_module
    ):
        # Three convolution heads: theta, phi, and g.
        self.conv_theta = nn.Conv3d(
            self.dim, self.dim_inner, kernel_size=1, stride=1, padding=0
        )
        self.conv_phi = nn.Conv3d(
            self.dim, self.dim_inner, kernel_size=1, stride=1, padding=0
        )
        self.conv_g = nn.Conv3d(
            self.dim, self.dim_inner, kernel_size=1, stride=1, padding=0
        )

        # Final convolution output.
        self.conv_out = nn.Conv3d(
            self.dim_inner, self.dim, kernel_size=1, stride=1, padding=0
        )
        # Zero initializing the final convolution output.
        self.conv_out.zero_init = zero_init_final_conv

        # TODO: change the name to `norm`
        self.bn = norm_module(
            num_features=self.dim,
            eps=self.norm_eps,
            momentum=self.norm_momentum,
        )
        # Zero initializing the final bn.
        self.bn.transform_final_bn = zero_init_final_norm

        # Optional to add the spatial-temporal pooling.
        if self.use_pool:
            self.pool = nn.MaxPool3d(
                kernel_size=self.pool_size,
                stride=self.pool_size,
                padding=[0, 0, 0],
            )

    def forward(self, x):
        x_identity = x
        N, C, T, H, W = x.size()

        theta = self.conv_theta(x)

        # Perform temporal-spatial pooling to reduce the computation.
        if self.use_pool:
            x = self.pool(x)

        phi = self.conv_phi(x)
        g = self.conv_g(x)

        theta = theta.view(N, self.dim_inner, -1)
        phi = phi.view(N, self.dim_inner, -1)
        g = g.view(N, self.dim_inner, -1)

        # (N, C, TxHxW) * (N, C, TxHxW) => (N, TxHxW, TxHxW).
        theta_phi = torch.einsum("nct,ncp->ntp", (theta, phi))
        # For original Non-local paper, there are two main ways to normalize
        # the affinity tensor:
        #   1) Softmax normalization (norm on exp).
        #   2) dot_product normalization.
        if self.instantiation == "softmax":
            # Normalizing the affinity tensor theta_phi before softmax.
            theta_phi = theta_phi * (self.dim_inner ** -0.5)
            theta_phi = nn.functional.softmax(theta_phi, dim=2)
        elif self.instantiation == "dot_product":
            spatial_temporal_dim = theta_phi.shape[2]
            theta_phi = theta_phi / spatial_temporal_dim
        else:
            raise NotImplementedError(
                "Unknown norm type {}".format(self.instantiation)
            )

        # (N, TxHxW, TxHxW) * (N, C, TxHxW) => (N, C, TxHxW).
        theta_phi_g = torch.einsum("ntg,ncg->nct", (theta_phi, g))

        # (N, C, TxHxW) => (N, C, T, H, W).
        theta_phi_g = theta_phi_g.view(N, self.dim_inner, T, H, W)

        p = self.conv_out(theta_phi_g)
        p = self.bn(p)
        return x_identity + p


class ResStage(nn.Module):
    """
    Stage of 3D ResNet. It expects to have one or more tensors as input for
        single pathway (C2D, I3D, Slow), and multi-pathway (SlowFast) cases.
        More details can be found here:
        Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
        "SlowFast networks for video recognition."
        https://arxiv.org/pdf/1812.03982.pdf
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        stride,
        temp_kernel_sizes,
        num_blocks,
        dim_inner,
        num_groups,
        num_block_temp_kernel,
        nonlocal_inds,
        nonlocal_group,
        nonlocal_pool,
        dilation,
        instantiation="softmax",
        trans_func_name="bottleneck_transform",
        stride_1x1=False,
        inplace_relu=True,
        norm_module=nn.BatchNorm3d,
        drop_connect_rate=0.0,
    ):
        """
        The `__init__` method of any subclass should also contain these arguments.
        ResStage builds p streams, where p can be greater or equal to one.
        Args:
            dim_in (list): list of p the channel dimensions of the input.
                Different channel dimensions control the input dimension of
                different pathways.
            dim_out (list): list of p the channel dimensions of the output.
                Different channel dimensions control the input dimension of
                different pathways.
            temp_kernel_sizes (list): list of the p temporal kernel sizes of the
                convolution in the bottleneck. Different temp_kernel_sizes
                control different pathway.
            stride (list): list of the p strides of the bottleneck. Different
                stride control different pathway.
            num_blocks (list): list of p numbers of blocks for each of the
                pathway.
            dim_inner (list): list of the p inner channel dimensions of the
                input. Different channel dimensions control the input dimension
                of different pathways.
            num_groups (list): list of number of p groups for the convolution.
                num_groups=1 is for standard ResNet like networks, and
                num_groups>1 is for ResNeXt like networks.
            num_block_temp_kernel (list): extent the temp_kernel_sizes to
                num_block_temp_kernel blocks, then fill temporal kernel size
                of 1 for the rest of the layers.
            nonlocal_inds (list): If the tuple is empty, no nonlocal layer will
                be added. If the tuple is not empty, add nonlocal layers after
                the index-th block.
            dilation (list): size of dilation for each pathway.
            nonlocal_group (list): list of number of p nonlocal groups. Each
                number controls how to fold temporal dimension to batch
                dimension before applying nonlocal transformation.
                https://github.com/facebookresearch/video-nonlocal-net.
            instantiation (string): different instantiation for nonlocal layer.
                Supports two different instantiation method:
                    "dot_product": normalizing correlation matrix with L2.
                    "softmax": normalizing correlation matrix with Softmax.
            trans_func_name (string): name of the the transformation function apply
                on the network.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
            drop_connect_rate (float): basic rate at which blocks are dropped,
                linearly increases from input to output blocks.
        """
        super(ResStage, self).__init__()
        assert all(
            (
                num_block_temp_kernel[i] <= num_blocks[i]
                for i in range(len(temp_kernel_sizes))
            )
        )
        self.num_blocks = num_blocks
        self.nonlocal_group = nonlocal_group
        self._drop_connect_rate = drop_connect_rate
        self.temp_kernel_sizes = [
            (temp_kernel_sizes[i] * num_blocks[i])[: num_block_temp_kernel[i]]
            + [1] * (num_blocks[i] - num_block_temp_kernel[i])
            for i in range(len(temp_kernel_sizes))
        ]
        assert (
            len(
                {
                    len(dim_in),
                    len(dim_out),
                    len(temp_kernel_sizes),
                    len(stride),
                    len(num_blocks),
                    len(dim_inner),
                    len(num_groups),
                    len(num_block_temp_kernel),
                    len(nonlocal_inds),
                    len(nonlocal_group),
                }
            )
            == 1
        )
        self.num_pathways = len(self.num_blocks)
        self._construct(
            dim_in,
            dim_out,
            stride,
            dim_inner,
            num_groups,
            trans_func_name,
            stride_1x1,
            inplace_relu,
            nonlocal_inds,
            nonlocal_pool,
            instantiation,
            dilation,
            norm_module,
        )

    def _construct(
        self,
        dim_in,
        dim_out,
        stride,
        dim_inner,
        num_groups,
        trans_func_name,
        stride_1x1,
        inplace_relu,
        nonlocal_inds,
        nonlocal_pool,
        instantiation,
        dilation,
        norm_module,
    ):
        for pathway in range(self.num_pathways):
            for i in range(self.num_blocks[pathway]):
                # Retrieve the transformation function.
                trans_func = self.get_trans_func(trans_func_name)
                # Construct the block.
                res_block = ResBlock(
                    dim_in[pathway] if i == 0 else dim_out[pathway],
                    dim_out[pathway],
                    self.temp_kernel_sizes[pathway][i],
                    stride[pathway] if i == 0 else 1,
                    trans_func,
                    dim_inner[pathway],
                    num_groups[pathway],
                    stride_1x1=stride_1x1,
                    inplace_relu=inplace_relu,
                    dilation=dilation[pathway],
                    norm_module=norm_module,
                    block_idx=i,
                    drop_connect_rate=self._drop_connect_rate,
                )
                self.add_module("pathway{}_res{}".format(pathway, i), res_block)
                if i in nonlocal_inds[pathway]:
                    nln = Nonlocal(
                        dim_out[pathway],
                        dim_out[pathway] // 2,
                        nonlocal_pool[pathway],
                        instantiation=instantiation,
                        norm_module=norm_module,
                    )
                    self.add_module(
                        "pathway{}_nonlocal{}".format(pathway, i), nln
                    )

    def forward(self, inputs):
        output = []
        for pathway in range(self.num_pathways):
            x = inputs[pathway]
            for i in range(self.num_blocks[pathway]):
                m = getattr(self, "pathway{}_res{}".format(pathway, i))
                x = m(x)
                if hasattr(self, "pathway{}_nonlocal{}".format(pathway, i)):
                    nln = getattr(
                        self, "pathway{}_nonlocal{}".format(pathway, i)
                    )
                    b, c, t, h, w = x.shape
                    if self.nonlocal_group[pathway] > 1:
                        # Fold temporal dimension into batch dimension.
                        x = x.permute(0, 2, 1, 3, 4)
                        x = x.reshape(
                            b * self.nonlocal_group[pathway],
                            t // self.nonlocal_group[pathway],
                            c,
                            h,
                            w,
                        )
                        x = x.permute(0, 2, 1, 3, 4)
                    x = nln(x)
                    if self.nonlocal_group[pathway] > 1:
                        # Fold back to temporal dimension.
                        x = x.permute(0, 2, 1, 3, 4)
                        x = x.reshape(b, t, c, h, w)
                        x = x.permute(0, 2, 1, 3, 4)
            output.append(x)

        return output

    def get_trans_func(self, name):
        """
        Retrieves the transformation module by name.
        """
        trans_funcs = {
            "x3d_transform": X3DTransform,
        }
        assert (
                name in trans_funcs.keys()
        ), "Transformation function '{}' not supported".format(name)
        return trans_funcs[name]


class X3DHead(nn.Module):
    """
    X3D head.
    This layer performs a fully-connected projection during training, when the
    input size is 1x1x1. It performs a convolutional projection during testing
    when the input size is larger than 1x1x1. If the inputs are from multiple
    different pathways, the inputs will be concatenated after pooling.
    """

    def __init__(
        self,
        dim_in,
        dim_inner,
        dim_out,
        num_classes,
        pool_size,
        dropout_rate=0.0,
        act_func="softmax",
        inplace_relu=True,
        eps=1e-5,
        bn_mmt=0.1,
        norm_module=nn.BatchNorm3d,
        bn_lin5_on=False,
    ):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        X3DHead takes a 5-dim feature tensor (BxCxTxHxW) as input.
        Args:
            dim_in (float): the channel dimension C of the input.
            num_classes (int): the channel dimensions of the output.
            pool_size (float): a single entry list of kernel size for
                spatiotemporal pooling for the TxHxW dimensions.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
            inplace_relu (bool): if True, calculate the relu on the original
                input without allocating new memory.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
            bn_lin5_on (bool): if True, perform normalization on the features
                before the classifier.
        """
        super(X3DHead, self).__init__()
        self.pool_size = pool_size
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.act_func = act_func
        self.eps = eps
        self.bn_mmt = bn_mmt
        self.inplace_relu = inplace_relu
        self.bn_lin5_on = bn_lin5_on
        self._construct_head(dim_in, dim_inner, dim_out, norm_module)

    def _construct_head(self, dim_in, dim_inner, dim_out, norm_module):

        self.conv_5 = nn.Conv3d(
            dim_in,
            dim_inner,
            kernel_size=(1, 1, 1),
            stride=(1, 1, 1),
            padding=(0, 0, 0),
            bias=False,
        )
        self.conv_5_bn = norm_module(
            num_features=dim_inner, eps=self.eps, momentum=self.bn_mmt
        )
        self.conv_5_relu = nn.ReLU(self.inplace_relu)

        if self.pool_size is None:
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool = nn.AvgPool3d(self.pool_size, stride=1)

        self.lin_5 = nn.Conv3d(
            dim_inner,
            dim_out,
            kernel_size=(1, 1, 1),
            stride=(1, 1, 1),
            padding=(0, 0, 0),
            bias=False,
        )
        if self.bn_lin5_on:
            self.lin_5_bn = norm_module(
                num_features=dim_out, eps=self.eps, momentum=self.bn_mmt
            )
        self.lin_5_relu = nn.ReLU(self.inplace_relu)

        if self.dropout_rate > 0.0:
            self.dropout = nn.Dropout(self.dropout_rate)
        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        self.projection = nn.Linear(dim_out, self.num_classes, bias=True)

        # Softmax for evaluation and testing.
        if self.act_func == "softmax":
            self.act = nn.Softmax(dim=4)
        elif self.act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(self.act_func)
            )

    def forward(self, inputs):
        # In its current design the X3D head is only useable for a single
        # pathway input.
        assert len(inputs) == 1, "Input tensor does not contain 1 pathway"
        x = self.conv_5(inputs[0])
        x = self.conv_5_bn(x)
        x = self.conv_5_relu(x)
        x = self.avg_pool(x)

        x = self.lin_5(x)
        if self.bn_lin5_on:
            x = self.lin_5_bn(x)
        x = self.lin_5_relu(x)

        # (N, C, T, H, W) -> (N, T, H, W, C).
        x = x.permute((0, 2, 3, 4, 1))
        # Perform dropout.
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        x = self.projection(x)

        # Performs fully convlutional inference.
        if not self.training:
            x = self.act(x)
            x = x.mean([1, 2, 3])

        x = x.view(x.shape[0], -1)
        return x


class X3D(nn.Module):
    """
    X3D model builder. It builds a X3D network backbone, which is a ResNet.
    Christoph Feichtenhofer.
    "X3D: Expanding Architectures for Efficient Video Recognition."
    https://arxiv.org/abs/2004.04730
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(X3D, self).__init__()
        self.norm_module = self.get_norm(cfg)
        self.enable_detection = False
        self.num_pathways = 1

        exp_stage = 2.0
        self.dim_c1 = cfg.X3D.DIM_C1

        self.dim_res2 = (
            self._round_width(self.dim_c1, exp_stage, divisor=8)
            if cfg.X3D.SCALE_RES2
            else self.dim_c1
        )
        self.dim_res3 = self._round_width(self.dim_res2, exp_stage, divisor=8)
        self.dim_res4 = self._round_width(self.dim_res3, exp_stage, divisor=8)
        self.dim_res5 = self._round_width(self.dim_res4, exp_stage, divisor=8)

        self.block_basis = [
            # blocks, c, stride
            [1, self.dim_res2, 2],
            [2, self.dim_res3, 2],
            [5, self.dim_res4, 2],
            [3, self.dim_res5, 2],
        ]
        self._construct_network(cfg)
        self.init_weights(
            self, cfg.MODEL.FC_INIT_STD, cfg.RESNET.ZERO_INIT_FINAL_BN
        )
        setattr(self, 'example_input_array', None)

    def _round_width(self, width, multiplier, min_depth=8, divisor=8):
        """Round width of filters based on width multiplier."""
        if not multiplier:
            return width

        width *= multiplier
        min_depth = min_depth or divisor
        new_filters = max(
            min_depth, int(width + divisor / 2) // divisor * divisor
        )
        if new_filters < 0.9 * width:
            new_filters += divisor
        return int(new_filters)

    def _round_repeats(self, repeats, multiplier):
        """Round number of layers based on depth multiplier."""
        multiplier = multiplier
        if not multiplier:
            return repeats
        return int(math.ceil(multiplier * repeats))

    def _construct_network(self, cfg):
        """
        Builds a single pathway X3D model.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        assert cfg.MODEL.ARCH in _POOL1.keys()
        assert cfg.RESNET.DEPTH in _MODEL_STAGE_DEPTH.keys()

        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[cfg.RESNET.DEPTH]

        num_groups = cfg.RESNET.NUM_GROUPS
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        dim_inner = num_groups * width_per_group

        w_mul = cfg.X3D.WIDTH_FACTOR
        d_mul = cfg.X3D.DEPTH_FACTOR
        dim_res1 = self._round_width(self.dim_c1, w_mul)

        temp_kernel = _TEMPORAL_KERNEL_BASIS[cfg.MODEL.ARCH]

        self.s1 = VideoModelStem(
            dim_in=cfg.DATA.INPUT_CHANNEL_NUM,
            dim_out=[dim_res1],
            kernel=[temp_kernel[0][0] + [3, 3]],
            stride=[[1, 2, 2]],
            padding=[[temp_kernel[0][0][0] // 2, 1, 1]],
            norm_module=self.norm_module,
            stem_func_name="x3d_stem",
        )

        # blob_in = s1
        dim_in = dim_res1
        for stage, block in enumerate(self.block_basis):
            dim_out = self._round_width(block[1], w_mul)
            dim_inner = int(cfg.X3D.BOTTLENECK_FACTOR * dim_out)

            n_rep = self._round_repeats(block[0], d_mul)
            prefix = "s{}".format(
                stage + 2
            )  # start w res2 to follow convention

            s = ResStage(
                dim_in=[dim_in],
                dim_out=[dim_out],
                dim_inner=[dim_inner],
                temp_kernel_sizes=temp_kernel[1],
                stride=[block[2]],
                num_blocks=[n_rep],
                num_groups=[dim_inner]
                if cfg.X3D.CHANNELWISE_3x3x3
                else [num_groups],
                num_block_temp_kernel=[n_rep],
                nonlocal_inds=cfg.NONLOCAL.LOCATION[0],
                nonlocal_group=cfg.NONLOCAL.GROUP[0],
                nonlocal_pool=cfg.NONLOCAL.POOL[0],
                instantiation=cfg.NONLOCAL.INSTANTIATION,
                trans_func_name=cfg.RESNET.TRANS_FUNC,
                stride_1x1=cfg.RESNET.STRIDE_1X1,
                norm_module=self.norm_module,
                dilation=cfg.RESNET.SPATIAL_DILATIONS[stage],
                drop_connect_rate=cfg.MODEL.DROPCONNECT_RATE
                * (stage + 2)
                / (len(self.block_basis) + 1),
            )
            dim_in = dim_out
            self.add_module(prefix, s)

        if self.enable_detection:
            NotImplementedError
        else:
            spat_sz = int(math.ceil(cfg.DATA.TRAIN_CROP_SIZE / 32.0))
            self.head = X3DHead(
                dim_in=dim_out,
                dim_inner=dim_inner,
                dim_out=cfg.X3D.DIM_C5,
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[cfg.DATA.NUM_FRAMES, spat_sz, spat_sz],
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                bn_lin5_on=cfg.X3D.BN_LIN5,
            )

    def forward(self, x, bboxes=None):
        x = [x]  # list with tensor for each pathway!
        for module in self.children():
            x = module(x)
        return x

    def get_norm(self, cfg):
        """
        Args:
            cfg (CfgNode): model building configs, details are in the comments of
                the config file.
        Returns:
            nn.Module: the normalization layer.
        """
        if cfg.BN.NORM_TYPE == "batchnorm":
            return nn.BatchNorm3d
        else:
            raise NotImplementedError(
                "Norm type {} is not supported".format(cfg.BN.NORM_TYPE)
            )

    def init_weights(self, model, fc_init_std=0.01, zero_init_final_bn=True):
        """
        Performs ResNet style weight initialization.
        Args:
            fc_init_std (float): the expected standard deviation for fc layer.
            zero_init_final_bn (bool): if True, zero initialize the final bn for
                every bottleneck.
        """
        for m in model.modules():
            if isinstance(m, nn.Conv3d):
                """
                Follow the initialization method proposed in:
                {He, Kaiming, et al.
                "Delving deep into rectifiers: Surpassing human-level
                performance on imagenet classification."
                arXiv preprint arXiv:1502.01852 (2015)}
                """
                c2_msra_fill(m)
            elif isinstance(m, nn.BatchNorm3d):
                if (
                        hasattr(m, "transform_final_bn")
                        and m.transform_final_bn
                        and zero_init_final_bn
                ):
                    batchnorm_weight = 0.0
                else:
                    batchnorm_weight = 1.0
                if m.weight is not None:
                    m.weight.data.fill_(batchnorm_weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0.0, std=fc_init_std)
                if m.bias is not None:
                    m.bias.data.zero_()

    @property
    def groups(self) -> [[nn.Module]]:
        return [[self.s1],
                [self.s2],
                [self.s3],
                [self.s4],
                [self.s5],
                [self.head]]

    @staticmethod
    def provide_pretrained_weights() -> Optional[Path]:
        return None


class X3D_L(X3D):

    @staticmethod
    def provide_pretrained_weights() -> Optional[Path]:
        url = 'https://dl.fbaipublicfiles.com/pyslowfast/x3d_models/x3d_l.pyth'
        return Fetcher().load(url, Path('.'))

    def __init__(self, num_classes: int):
        cfg = CfgNode(dict(
            X3D=dict(
                WIDTH_FACTOR=2.0,
                DEPTH_FACTOR=5.0,
                BOTTLENECK_FACTOR=2.25,
                DIM_C5=2048,
                DIM_C1=12,
                SCALE_RES2=False,
                CHANNELWISE_3x3x3 = True,
                BN_LIN5 = False
            ),
            RESNET=dict(
                ZERO_INIT_FINAL_BN=True,
                TRANS_FUNC='x3d_transform',
                STRIDE_1X1=False,
                DEPTH=50,
                NUM_GROUPS=1,
                WIDTH_PER_GROUP = 64,
                SPATIAL_DILATIONS = [[1], [1], [1], [1]]
            ),
            BN=dict(
                NORM_TYPE="batchnorm",
                USE_PRECISE_STATS=True,
                NUM_BATCHES_PRECISE=200,
                WEIGHT_DECAY=0.0
            ),
            MODEL=dict(
                NUM_CLASSES=num_classes,
                ARCH='x3d',
                MODEL_NAME='X3D',
                LOSS_FUNC='cross_entropy',
                DROPOUT_RATE=0.5,
                DROPCONNECT_RATE = 0.0,
                HEAD_ACT = "sigmoid",
                FC_INIT_STD = 0.01
            ),
            DATA=dict(
                INPUT_CHANNEL_NUM=[3],
                TRAIN_CROP_SIZE=312,
                NUM_FRAMES=16
            ),
            NONLOCAL = dict(
                LOCATION = [[[]], [[]], [[]], [[]]],
                GROUP = [[1], [1], [1], [1]],
                INSTANTIATION = "dot_product",
                POOL = [
                    # Res2
                    [[1, 2, 2], [1, 2, 2]],
                    # Res3
                    [[1, 2, 2], [1, 2, 2]],
                    # Res4
                    [[1, 2, 2], [1, 2, 2]],
                    # Res5
                    [[1, 2, 2], [1, 2, 2]],
                ]
            )
        ))
        super().__init__(cfg)
