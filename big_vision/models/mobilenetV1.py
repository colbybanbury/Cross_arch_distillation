# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""MobileNet V1, from https://arxiv.org/abs/1704.04861.

Modified from https://github.com/deepmind/dm-haiku/blob/main/haiku/_src/nets/mobilenetv1.py
and https://github.com/rwightman/efficientnet-jax/blob/master/jeffnet/linen/blocks_linen.py#L87

"""
from numbers import Number
from typing import Any, Callable, Optional, Sequence, Union, Tuple, Iterable, Optional

from big_vision import utils
from big_vision.models import common
import flax
import flax.linen as nn
import flax.training.checkpoints
import jax.numpy as jnp
import numpy as np
from jax import lax

def to_tuple(v: Union[Tuple[Number, ...], Number, Iterable], n: int):
    """Converts input to tuple."""
    if isinstance(v, tuple):
        return v
    elif isinstance(v, Number):
        return (v,) * n
    else:
        return tuple(v)

# calculate SAME-like symmetric padding for a convolution
def get_like_padding(kernel_size: int, stride: int = 1, dilation: int = 1, **_) -> int:
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding

def create_conv2d(
        features: int,
        kernel_size: int = 3,
        stride: Optional[int] = None,
        padding: Union[str, Tuple[int, int]] = "VALID",
        dilation: Optional[int] = None,
        groups: int = 1,
        bias: bool = False,
        dtype: Any = jnp.float32,
        precision: Any = None,
        conv_name: Optional[str] = None,
        kernel_init: Callable[[Any, Tuple[int], Any], Any] = nn.initializers.kaiming_normal(),
        bias_init: Callable[[Any, Tuple[int], Any], Any] = nn.initializers.zeros,
        depthwise: bool = False):
    """Flax Linen Conv & Depthwise Conv"""
    stride = stride or 1
    dilation = dilation or 1
    if isinstance(padding, str):
        if padding == 'LIKE':
            padding = get_like_padding(kernel_size, stride, dilation)
            padding = to_tuple(padding, 2)
            padding = [padding, padding]
    else:
        padding = to_tuple(padding, 2)
        padding = [padding, padding]

    if depthwise:
        groups = features
        # ^ makes it DepthwiseConv2D
    else:
        groups = 1
    return nn.Conv(
            features=features,
            kernel_size=to_tuple(kernel_size, 2),
            strides=to_tuple(stride, 2),
            padding=padding,
            kernel_dilation=to_tuple(dilation, 2),
            feature_group_count=groups, 
            use_bias=bias,
            dtype=dtype,
            precision=precision,
            name=conv_name,
            kernel_init=kernel_init,
            bias_init=bias_init,
    )



class DepthwiseSeparable(nn.Module):
    """ DepthwiseSeparable block
    Used for DS convs in MobileNet-V1 and in the place of IR blocks that have no expansion
    (factor of 1.0). This is an alternative to having a IR with an optional first pw conv.
    https://github.com/rwightman/efficientnet-jax/blob/master/jeffnet/linen/blocks_linen.py#L87
    """

    in_features: int
    out_features: int
    dw_kernel_size: int = 3
    pw_kernel_size: int = 1
    stride: int = 1
    dilation: int = 1
    pad_type: str = 'LIKE'
    noskip: bool = False
    pw_act: bool = False
    se_ratio: float = 0.
    drop_path_rate: float = 0.
    use_bn: bool = True
    
    norm_layer: Any = nn.BatchNorm
    # se_layer: Any = SqueezeExcite
    act_fn: Callable = nn.relu

    @nn.compact
    def __call__(self, x, training: bool):
        # shortcut = x #no shortcut in v1

        x = create_conv2d(
            self.in_features, self.dw_kernel_size, stride=self.stride, dilation=self.dilation,
            padding=self.pad_type, conv_name='conv_dw')(x)
        if self.use_bn:
            x = self.norm_layer(name='bn_dw')(x, use_running_average=not training)
        x = self.act_fn(x)

        # if self.se_layer is not None and self.se_ratio > 0:
        #     x = self.se_layer(
        #         num_features=self.in_features, se_ratio=self.se_ratio,
        #         conv_layer=self.conv_layer, act_fn=self.act_fn, name='se')(x)

        x = create_conv2d(
            self.out_features, self.pw_kernel_size, padding=self.pad_type,
            conv_name='conv_pw')(x)
        if self.use_bn:
            x = self.norm_layer(name='bn_pw')(x, use_running_average=not training)
        if self.pw_act:
            x = self.act_fn(x)

        # if (self.stride == 1 and self.in_features == self.out_features) and not self.noskip:
        #     x = DropPath(self.drop_path_rate)(x, training=training)
        #     x = x + shortcut
        return x


class Model(nn.Module):
    """MobileNetV1 model."""
    num_classes: int
    first_conv_features: int = 8
    # matches the MLPerf tiny Default
    strides: Sequence[int] = (1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1)
    channels: Sequence[int] = (16, 32, 32, 64, 64, 128, 128, 128, 128, 128, 128, 256, 256)
    use_bn: bool = False

    @nn.compact
    def __call__(self, image, *, train=False):
        out = {}

        initial_conv = nn.Conv(
            features=self.first_conv_features,
            kernel_size=(3, 3),
            strides=2,
            padding="SAME",
            use_bias=not self.use_bn,)

        x = initial_conv(image)
        if self.use_bn:
            x = nn.BatchNorm()(x, use_running_average=not train)
        x = nn.relu(x)
        for i in range(len(self.strides)):
            if i == 0:
                in_feat = self.first_conv_features
            else:
                in_feat = self.channels[i - 1]
            block = DepthwiseSeparable(in_features=in_feat,
                                    out_features=self.channels[i],
                                    stride = self.strides[i],
                                    use_bn=self.use_bn)
            x = block(x, training=train)

        x = jnp.mean(x, axis=(1, 2))

        x = nn.Dense(self.num_classes, name="head",
                      kernel_init=nn.initializers.zeros)(x)

        out["logits"] = x #Need this for distillation
        return x, out

def load(init_params, init_file, model_cfg, dont_load=()):
  """Load init from checkpoint."""
  del model_cfg  # Unused
  params = utils.load_params(None, init_file)
  params = common.merge_params(params, init_params, dont_load)
  return params
