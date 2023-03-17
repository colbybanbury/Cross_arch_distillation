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
r"""Pre-training flexible-seqlen ViT on ImageNet-21k following (internal link).

This config is for reference, we never ran it on public infrastructure.

big_vision.trainers.proj.flexi.train \
  --config big_vision/configs/proj/flexivitvww_sup.py \
  --workdir gs://[your_bucket]/big_vision/`date '+%m-%d_%H%M'` \
  --config.total_epochs 90
"""

import big_vision.configs.common as bvcc


def get_config(arg=None):
  """Config for training."""
  # 240px is nice because it's divisible by
  # [240, 120, 80, 60, 48, 40, 30, 24, 20, 16, 15, 12, 10, 8, 6, 5, 4, 3, 2, 1]
  c = bvcc.parse_arg(arg, runlocal=False, res=240)

  c.seed = 0
  c.total_epochs = 90
  c.num_classes = 2
  c.init_head_bias = -10.0
  c.loss = 'sigmoid_xent'

  c.input = dict()
  c.input.data = dict(
      name='vww',
      split='train[:90%]',
  )

  c.input.batch_size = 96 if not c.runlocal else 8
  c.input.shuffle_buffer_size = 250_000 if not c.runlocal else 25

  crop = f'inception_crop({c.res})'

  pp_common = (
      f'|onehot({c.num_classes}, key="label", key_result="labels")'
      f'|keep("image", "labels")'
  )
  c.input.pp = (
    f'copy("image/encoded","image")|copy("image/class/label","label")|drop("image/encoded","image/class/label")|{crop}|flip_lr|randaug(2, 15)'
    f'|value_range(-1, 1)' 
  )+ pp_common
  
  ppv = f'copy("image/encoded","image")|copy("image/class/label","label")|drop("image/encoded", "image/class/label")|resize_small({c.res})|central_crop({c.res})|value_range(-1, 1)' + pp_common


  # Aggressive pre-fetching because our models here are small, so we not only
  # can afford it, but we also need it for the smallest models to not be
  # bottle-necked by the input pipeline. Play around with it for -L models tho.
  c.input.prefetch = 8
  c.prefetch_to_device = 4

  c.log_training_steps = 50
  c.ckpt_steps = 1000

  # Model section
  c.model_name = 'proj.flexi.vit'
  c.model = dict(
      variant='B',
      pool_type='tok',
      posemb='learn',
      # patch_size=(32, 32),
      patch_size=(8, 8),
      posemb_size=(7, 7),
      seqhw=None,  # Dynamic!
  )
  c.model_init = 'FlexiViT-B i1k'
  c.model_load = dict(dont_load=['head/kernel', 'head/bias'])

  # Define the model parameters which are flexible:
  c.flexi = dict()
  c.flexi.seqhw = dict(
      # The settings to sample from. Corresponding patch-sizes at 240px:
      # 48, 40, 30, 24, 20, 16, 15, 12, 10, 8
      v=(5, 6, 8, 10, 12),
      # The probabilities/weights of them. Default uniform.
      p=(1, 1, 1, 1, 1),
  )

  # Optimizer section
  c.optax_name = 'scale_by_adam'
  c.optax = dict(mu_dtype='bfloat16')
  c.grad_clip_norm = 1.0

  c.lr = (0.001 * c.input.batch_size / 4096)
  c.wd = 0.0001
  c.schedule = dict(warmup_steps=10_000, decay_type='cosine')

  # c.mixup = dict(p=0.2, fold_in=None)

  minitrain_split = 'train[:512]'
  val_split = 'train[90%:]'
  test_split = 'val'

  def get_eval(s, split):
    return dict(
        type='classification',
        pred=f'predict_seqhw={s}',
        data=dict(name=c.input.data.name, split=split),
        pp_fn=ppv,
        loss_name='sigmoid_xent',
        log_steps=1000,
    )
  c.evals = {}
  for s in c.flexi.seqhw.v:
    c.evals[f'train{s:02d}'] = get_eval(s, minitrain_split)
    c.evals[f'val{s:02d}'] = get_eval(s, val_split)
    c.evals[f'test{s:02d}'] = get_eval(s, test_split)

  return c
