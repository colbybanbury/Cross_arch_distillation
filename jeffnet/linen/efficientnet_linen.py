""" EfficientNet (Flax Linen) Model and Factory

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
import re
from typing import Any, Callable, Sequence, Dict
from functools import partial

from flax import linen as nn
import flax.linen.initializers as initializers
import jax
import jax.numpy as jnp

from jeffnet.common import round_features, get_model_cfg, EfficientNetBuilder
from .helpers import load_pretrained
from .layers import conv2d, linear, get_act_fn, batchnorm2d
from .blocks_linen import ConvBnAct, SqueezeExcite, BlockFactory, Head, EfficientHead

ModuleDef = Any
Dtype = Any
effnet_normal = partial(initializers.variance_scaling, 2.0, "fan_out", "normal")
effnet_uniform = partial(initializers.variance_scaling, 1.0/3, "fan_out", "uniform")


class EfficientNet(nn.Module):
    """ EfficientNet (and other MBConvNets)
      * EfficientNet B0-B8, L2
      * EfficientNet-EdgeTPU
      * EfficientNet-Lite
      * MixNet S, M, L, XL
      * MobileNetV3
      * MobileNetV2
      * MnasNet A1, B1, and small
      * FBNet C
      * Single-Path NAS Pixel1
    """

    # model config
    block_defs: Sequence[Sequence[Dict]]
    stem_size: int = 32
    feat_multiplier: float = 1.0
    feat_divisor: int = 8
    feat_min: int = None
    fix_stem: bool = False
    pad_type: str = 'LIKE'
    output_stride: int = 32

    # classifier / head config
    efficient_head: bool = False
    num_classes: int = 1000
    num_features: int = 1280
    global_pool: str = 'avg'

    # pretrained / data config
    default_cfg: Dict = None

    # regularization
    drop_rate: float = 0.
    drop_path_rate: float = 0.

    dtype: Dtype = jnp.float32
    conv_layer: ModuleDef = conv2d
    norm_layer: ModuleDef = batchnorm2d
    se_layer: ModuleDef = SqueezeExcite
    act_fn: Callable = nn.relu

    #squeeze logits
    squeeze_logits: bool = False

    @nn.compact
    def __call__(self, x, train: bool):
        # add dtype binding to layers
        # FIXME is there better way to handle dtype? Passing dtype to all child Modules also seems messy...
        lkwargs = dict(
            conv_layer=partial(self.conv_layer, dtype=self.dtype, kernel_init=effnet_normal()),
            norm_layer=partial(self.norm_layer, dtype=self.dtype),
            act_fn=self.act_fn)
        se_layer = partial(self.se_layer, dtype=self.dtype)
        linear_layer = partial(linear, dtype=self.dtype, kernel_init=effnet_uniform())

        stem_features = self.stem_size
        if not self.fix_stem:
            stem_features = round_features(self.stem_size, self.feat_multiplier, self.feat_divisor, self.feat_min)
        x = ConvBnAct(
            out_features=stem_features, kernel_size=3, stride=2, pad_type=self.pad_type,
            **lkwargs, name='stem')(x, training=train)

        blocks = EfficientNetBuilder(
            stem_features, self.block_defs, BlockFactory(),
            feat_multiplier=self.feat_multiplier, feat_divisor=self.feat_divisor, feat_min=self.feat_min,
            output_stride=self.output_stride, pad_type=self.pad_type, se_layer=se_layer, **lkwargs,
            drop_path_rate=self.drop_path_rate)()
        for stage in blocks:
            for block in stage:
                x = block(x, training=train)

        head_layer = EfficientHead if self.efficient_head else Head
        x = head_layer(
            num_features=self.num_features, num_classes=self.num_classes, drop_rate=self.drop_rate,
            **lkwargs, dtype=self.dtype, linear_layer=linear_layer, name='head')(x, training=train)

        if self.squeeze_logits:
            #reshape from (bach, 1, 1, num_classes) to (batch, num_classes)
            x = jnp.squeeze(x, axis=(1, 2))

        out = {} #needed for consistency with other models
        return x, out


def _filter(state_dict):
    """ convert state dict keys from pytorch style origins to flax linen """
    out = {}
    p_blocks = re.compile(r'blocks\.(\d)\.(\d)')
    p_bn_scale = re.compile(r'bn(\w*)\.weight')
    for k, v in state_dict.items():
        k = p_blocks.sub(r'blocks_\1_\2', k)
        k = p_bn_scale.sub(r'bn\1.scale', k)
        k = k.replace('running_mean', 'mean')
        k = k.replace('running_var', 'var')
        k = k.replace('.weight', '.kernel')
        out[k] = v
    return out


def create_model(variant, pretrained=False, rng=None, input_shape=None, dtype=jnp.float32, **kwargs):
    model_cfg = get_model_cfg(variant)

    #all scaling from config
    if 'feat_multiplier' in kwargs:
        model_cfg['arch_cfg']['feat_multiplier'] = kwargs.pop('feat_multiplier')
    if 'depth_multiplier' in kwargs:
        model_cfg['arch_cfg']['depth_multiplier'] = kwargs.pop('depth_multiplier')
        model_cfg['arch_cfg']['fix_stem_head'] = True
    model_args = model_cfg['arch_fn'](variant, **model_cfg['arch_cfg'])
    model_args.update(kwargs)
    
    # resolve some special layers and their arguments
    se_args = model_args.pop('se_cfg', {})  # not consumable by model
    if 'se_layer' not in model_args:
        if 'bound_act_fn' in se_args:
            se_args['bound_act_fn'] = get_act_fn(se_args['bound_act_fn'])
        if 'gate_fn' in se_args:
            se_args['gate_fn'] = get_act_fn(se_args['gate_fn'])
        model_args['se_layer'] = partial(SqueezeExcite, **se_args)

    bn_args = model_args.pop('bn_cfg')  # not consumable by model
    if 'norm_layer' not in model_args:
        model_args['norm_layer'] = partial(batchnorm2d, **bn_args)

    model_args['act_fn'] = get_act_fn(model_args.pop('act_fn', 'relu'))  # convert str -> fn

    model = EfficientNet(dtype=dtype, default_cfg=model_cfg['default_cfg'], squeeze_logits= "v3" in variant,  **model_args)

    # rng = jax.random.PRNGKey(0) if rng is None else rng
    # params_rng, dropout_rng = jax.random.split(rng)
    # input_shape = model_cfg['default_cfg']['input_size'] if input_shape is None else input_shape
    # input_shape = (1, input_shape[1], input_shape[2], input_shape[0])   # CHW -> HWC by default

    # # FIXME is jiting the init worthwhile for my usage?
    # #     @jax.jit
    # #     def init(*args):
    # #         return model.init(*args, training=True)

    # variables = model.init(
    #     {'params': params_rng, 'dropout': dropout_rng},
    #     jnp.ones(input_shape, dtype=dtype),
    #     train=False)

    # if pretrained:
    #     variables = load_pretrained(variables, default_cfg=model.default_cfg, filter_fn=_filter)

    variables = None #changing implementation to return model but not variables, because big_vision handles this

    return model, variables
