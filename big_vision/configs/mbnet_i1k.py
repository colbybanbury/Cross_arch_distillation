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
r"""Pre-training mbnet on ILSVRC-2012 as in https://arxiv.org/abs/1912.11370

Run training of a mbnet variant:

big_vision.train \
    --config big_vision/configs/mbet_i1k.py \
    --workdir gs://[your_bucket]/big_vision/`date '+%m-%d_%H%M'` \
    --config.model.depth 50 --config.model.with 1
"""

# from big_vision.configs.common_fewshot import get_fewshot_lsr
import ml_collections as mlc


def get_config(runlocal=False):
  """Config for training on ImageNet-1k."""
  config = mlc.ConfigDict()

  config.seed = 0
  config.total_epochs = 600 #same as distill but prob stop early #90
  config.num_classes = 1000
  config.loss = 'softmax_xent'

  config.input = dict()
  config.input.data = dict(
      name='imagenet2012',
      split='train[:99%]',
  )
  config.input.batch_size = 512
  config.input.cache_raw = not runlocal  # Needs up to 120GB of RAM!
  config.input.shuffle_buffer_size = 250_000 if not runlocal else 10_000  # Per host.

  res = 96

  pp_common = '|onehot(1000, key="{lbl}", key_result="labels")' + f'|resize_small({res})'
  pp_common += '|value_range(-1, 1)|keep("image", "labels")'
  config.input.pp = 'decode_jpeg_and_inception_crop(224)|flip_lr' + pp_common.format(lbl='label')
  pp_eval = 'decode|resize_small(256)|central_crop(224)' + pp_common

  config.log_training_steps = 50
  config.ckpt_steps = 1000
  config.keep_ckpt_steps = 40000

  # Model section
  config.model_name = 'mobilenetV1'
  config.model = dict()

  # Optimizer section
  config.optax_name = 'big_vision.momentum_hp'
  config.grad_clip_norm = 1.0

  # linear scaling rule. Don't forget to sweep if sweeping batch_size.
  config.lr = (0.045 * config.input.batch_size) / 64
  config.wd = (0.00003 * config.input.batch_size) / 512
  config.schedule = dict(decay_type='cosine', warmup_steps=1000)

  # Eval section
  def get_eval(split, dataset='imagenet2012'):
    return dict(
        type='classification',
        data=dict(name=dataset, split=split),
        pp_fn=pp_eval.format(lbl='label'),
        loss_name=config.loss,
        log_steps=1000,  # Very fast O(seconds) so it's fine to run it often.
        cache_final=not runlocal,
    )
  config.evals = {}
  config.evals.train = get_eval('train[:2%]')
  config.evals.minival = get_eval('train[99%:]')
  config.evals.val = get_eval('validation')
#   config.evals.v2 = get_eval('test', dataset='imagenet_v2')
  config.evals.real = get_eval('validation', dataset='imagenet2012_real')
  config.evals.real.pp_fn = pp_eval.format(lbl='real_label')

  # config.evals.fewshot = get_fewshot_lsr(runlocal=runlocal)
  # config.evals.fewshot.log_steps = 1000

  return config