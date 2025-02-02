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
    --config big_vision/configs/vww-scratch.py \
    --workdir gs://imagenet_distill/big_vision/vww/scratch_cnn-96/`date '+%m-%d_%H%M'`
"""

# from big_vision.configs.common_fewshot import get_fewshot_lsr
import ml_collections as mlc


def get_config(runlocal=False):
  """Config for training on ImageNet-1k."""
  config = mlc.ConfigDict()

  config.seed = 0
  config.total_epochs = 10_000
  config.num_classes = 2
  config.loss = 'softmax_xent'

  config.input = dict()
  config.input.data = dict(
      name='vww',
      split='train[:90%]',
  )
  config.input.batch_size = 96
  config.input.cache_raw = True
  config.input.shuffle_buffer_size = 50_000
  config.prefetch_to_device = 4
  
  config.student_res = 96

  teacher_hres = 256
  teacher_lres = 256
  crop = f'resize_small({teacher_hres})|central_crop({teacher_lres})|resize_small({config.student_res})'#f'inception_crop({teacher_lres})'

  # Preprocessing pipeline for student & teacher.
  pp_common = (
      f'|onehot({config.num_classes}, key="label", key_result="labels")'
      f'|keep("image", "labels")'
  )
  config.input.pp = (
    f'copy("image/encoded","image")|copy("image/class/label","label")|drop("image/encoded","image/class/label")|{crop}|flip_lr'
    f'|value_range(-1, 1)' 
  )+ pp_common
  
  ppv = f'copy("image/encoded","image")|copy("image/class/label","label")|drop("image/encoded", "image/class/label")|resize_small({teacher_hres})|central_crop({teacher_lres})|resize_small({config.student_res})|value_range(-1, 1)' + pp_common


  config.log_training_steps = 50
  config.ckpt_steps = 1000

  # Model section
  # config.model_name = 'mobilenetV1'
  # config.model_init = 'gs://imagenet_distill/big_vision/imgnet1k/mbnet-scratch/lr045_wd00003-bs512/01-09_1819/checkpoint.npz'
  # config.model = dict()
  # config.model_load = dict(dont_load=['head/kernel', 'head/bias'])

  config.model_name = 'efficientnet_jax_wrapper'
  config.model = dict(variant='pt_mobilenetv2_035', num_features=112 ) #tf_mobilenetv3_small_035

  # config.model_name = 'vit' 
  # config.model = dict(variant='B/32', pool_type='tok')

  # config.model_name = 'cnn'
  # config.model = dict()

  # Optimizer section
  config.grad_clip_norm = 1.0
  config.optax_name = 'big_vision.scale_by_rms_momentum'
  config.optax = dict(eps=1e-8, momentum=0.9, decay=0.9)

  steps_per_epoch = 745049 // config.input.batch_size

  config.lr = 0.045#({'fast': 0.001, 'medium': 0.0003, 'long': 0.0001}[arg.variant] / 512) * config.input.batch_size
  # config.wd = 1e-5#({'fast': 3e-5, 'medium': 1e-5, 'long': 1e-6}[arg.variant] / 512) * config.input.batch_size
  config.schedule = dict(decay_type='step', decay_rate=0.98, decay_steps=1.0*steps_per_epoch)
  # warmup_steps=5.0*steps_per_epoch, lr_minimum=1e-6,
  # config.optim_name = 'adam_hp'

  # Eval section
  minitrain_split = 'train[:512]'
  val_split = 'train[90%:]'
  test_split = 'val'

  def get_eval(split):
    return dict(
        type='classification',
        data=dict(name=config.input.data.name, split=split),
        pp_fn=ppv,
        loss_name='softmax_xent',
        log_steps=500,
    )
  config.evals = {}
  config.evals.train = get_eval(minitrain_split)
  config.evals.val = get_eval(val_split)
  config.evals.test = get_eval(test_split)

  # config.evals.fewshot = get_fewshot_lsr(runlocal=runlocal)
  # config.evals.fewshot.log_steps = 1000

  return config