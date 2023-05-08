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

# pylint: disable=line-too-long
r"""Training CNN on VWW

big_vision.train \
    --config big_vision/configs/jeffent-pets.py \
    --workdir gs://imagenet_distill/big_vision/vww/scratch_cnn-96/`date '+%m-%d_%H%M'`
"""

import ml_collections as mlc
import big_vision.configs.common as bvcc

from jeffnet.common.model_cfgs import get_model_cfg

NCLS = dict(flowers=102, pet=37, imagenet=1000)

def get_config(arg=None):
  """Config for sweeping embedded architectures on Pets."""

  arg = bvcc.parse_arg(arg, proj_name="unamed_proj", runlocal=False, data='pet', lr=-1., wd=-1., res=128, epochs=-1, variant="pt_mobilenetv2_035", width=-1.0, depth=-1.0, speed='fast',init_model=False,seed=0)
  config = mlc.ConfigDict()

  config.seed = arg.seed

  config.input = {}
  config.input.data = dict(
      name=dict(flowers='oxford_flowers102', pet='oxford_iiit_pet', imagenet='imagenet2012')[arg.data],
      split=dict(flowers='train', pet='train[:90%]', imagenet='train[:98%]')[arg.data],
  )
  config.input.batch_size = 256 if arg.data != 'imagenet' else 1024
  config.input.cache_raw = True
  config.input.shuffle_buffer_size = 25_000
  config.prefetch_to_device = 4

  config.num_classes = NCLS[arg.data]
  if arg.epochs >= 0:
    config.total_epochs = arg.epochs
  else:
    config.total_epochs = {
        'flowers': {'fast': 25_000, 'medium': 100_000, 'long': 1_000_000},
        'pet': {'fast': 1000, 'medium': 3000, 'long': 30_000},
        'imagenet': {'fast': 600, 'medium': 3000, 'long': 30_000},
    }[arg.data][arg.speed]

  config.log_training_steps = 100
  config.ckpt_steps = 2500


  config.proj_name = arg.proj_name
  config.log_name = f"{arg.variant}"
  if arg.lr >= 0:
    config.log_name += f"-lr{arg.lr}"
  if arg.wd >= 0:
    config.log_name += f"-wd{arg.wd}"
  if arg.width > 0:
    config.log_name += f"-w{arg.width}"
  if arg.depth > 0:
    config.log_name += f"-d{arg.depth}"

  # Model section
  config.model_name = 'efficientnet_jax_wrapper'
  config.model = dict(variant=arg.variant)

  if arg.width > 0:
    config.model.feat_multiplier = arg.width
  if arg.depth > 0:
    config.model.depth_multiplier = arg.depth
  if arg.init_model:
    model_cfg = get_model_cfg(arg.variant)
    config.model_init = model_cfg["default_cfg"]["url"]
    config.model_load = dict(dont_load=["head"])


  # identical pre-preprocessing as distillation
  config.student_res = arg.res

  hres = 256
  lres = 224

  crop='inception_crop(224)'

  pp_common = (
      f'|onehot({config.num_classes}, key="label", key_result="labels")'
      f'|keep("image", "labels")'
  )
  config.input.pp = (
    f'decode|{crop}|flip_lr|'
    f'|value_range(-1, 1)|resize_small({config.student_res})' 
  )+ pp_common
  ppv = f'decode|resize_small({hres})|central_crop({lres})|normalize' + pp_common

  if config.student_res is not None:
    ppv_student = f'decode|resize_small({hres})|central_crop({lres})|resize_small({config.student_res})|value_range(-1, 1)' + pp_common
  else:
    ppv_student = ppv

  config.mixup = dict(p=1.0, n=2)

  # Optimizer section
  config.grad_clip_norm = 1.0
  config.optax_name = 'scale_by_adam'
  config.optax = dict(mu_dtype='bfloat16')

  if arg.lr <= 0:
    config.lr = {
        'flowers':{'fast': 0.001, 'medium': 0.001, 'long': 0.0003}, #{'fast': 5e-5}
        'pet': {'fast': 0.01, 'medium': 0.003, 'long': 0.003},
        'imagenet': {'fast': 0.001, 'medium': 0.0003, 'long': 0.0003},
    }[arg.data][arg.speed]
  else:
    config.lr = arg.lr
  if arg.wd <= 0:
    config.wd = {
        'flowers': {'fast': 3e-4, 'medium': 1e-4, 'long': 1e-5}, # {'fast': 5e-6},
        'pet': {'fast': 1e-3, 'medium': 3e-4, 'long': 1e-5},
        'imagenet': {'fast': 1e-4, 'medium': 3e-5, 'long': 1e-6},
    }[arg.data][arg.speed]
  else:
    config.wd = arg.wd

  config.schedule = dict(warmup_steps=1500, decay_type='cosine')
  config.optim_name = 'adam_hp'

  # Eval section
  minitrain_split = 'train[:512]' if not arg.runlocal else 'train[:16]'
  if arg.data == 'flowers':
    val_split = 'validation' if not arg.runlocal else 'validation[:16]'
    test_split = 'test' if not arg.runlocal else 'test[:16]'
  elif arg.data == 'pet':
    val_split = 'train[90%:]' if not arg.runlocal else 'train[:16]'
    test_split = 'test' if not arg.runlocal else 'test[:16]'
  elif arg.data == 'imagenet':
    minitrain_split = 'train[98%:]' if not arg.runlocal else 'train[:16]'
    val_split = 'validation' if not arg.runlocal else 'validation[:16]'


  def get_eval(split):
    return dict(
        type='classification',
        data=dict(name=config.input.data.name, split=split),
        pp_fn=ppv_student,
        loss_name='softmax_xent',
        log_steps=500,
    )
  config.evals = {}
  config.evals.train = get_eval(minitrain_split)
  config.evals.val = get_eval(val_split)
  if arg.data != 'imagenet':
    config.evals.test = get_eval(test_split)

  if arg.runlocal:
    config.input.shuffle_buffer_size = 10
    config.input.batch_size = 8

  return config