# Copyright 2022 Big Vision Authors.
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

"""ResNet V1 with GroupNorm."""

from typing import Optional, Sequence, Union

from big_vision import utils
from big_vision.models import common
import flax
import flax.linen as nn
import flax.training.checkpoints
import jax.numpy as jnp
import numpy as np


# def weight_standardize(w, axis, eps):
#   w = w - jnp.mean(w, axis=axis)
#   w = w / (jnp.std(w, axis=axis) + eps)
#   return w


# class StdConv(nn.Conv):

#   def param(self, name, *a, **kw):
#     param = super().param(name, *a, **kw)
#     if name == "kernel":
#       param = weight_standardize(param, axis=[0, 1, 2], eps=1e-5)
#     return param


class Model(nn.Module):
  """cnn"""
  num_classes: int
  
  @nn.compact
  def __call__(self, image, *, train=False):
    out = {}

    x = nn.Conv(features=32, kernel_size=(3, 3))(image)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.Conv(features=64, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dense(features=256)(x)
    x = nn.relu(x)
    x = nn.Dense(features=self.num_classes)(x)

    out["logits"] = x #Need this for distillation

    return x, out



def load(init_params, init_file, model_cfg, dont_load=()):
  """Load init from checkpoint."""
  del model_cfg  # Unused
  params = utils.load_params(None, init_file)
  params = common.merge_params(params, init_params, dont_load)
  return params
