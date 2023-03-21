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
r"""Distilling on Visual Wake Words (https://arxiv.org/pdf/1906.05721.pdf)


big_vision.trainers.proj.flexi.distill-flexi-teacher \
    --config big_vision/configs/proj/distill/vww.py:variant=fast \
    --workdir gs://[your_bucket]/big_vision/`date '+%m-%d_%H%M'` \
"""

import big_vision.configs.common as bvcc
import big_vision.configs.proj.distill.common as cd
import ml_collections as mlc


def get_config(arg=None):
  """Config for massive hypothesis-test on vww"""
  arg = bvcc.parse_arg(arg, runlocal=False, data='vww', variant='fast')
  config = mlc.ConfigDict()

  config.input = {}
  config.input.data = dict(
      name='vww',
      split='train[:90%]',
  )
  # config.input.rebalance = True #for places this is used to balance the dataset
  config.input.batch_size = 256
  config.input.cache_raw = True
  config.input.shuffle_buffer_size = 50_000
  config.prefetch_to_device = 4

  config.num_classes = 2
  config.total_epochs = 2_000#{'fast': 1_000, 'medium': 10_000, 'long': 100_000}[arg.variant]

  config.log_training_steps = 100
  config.ckpt_steps = 2500

  # Model section
  # config.student_name = 'cnn'#'bit_paper'
  # config.student = dict()#dict(depth=26, width=0.5)


  config.student_name = 'efficientnet_jax_wrapper'
  config.student = dict(variant='pt_mobilenetv2_035')

  
  # config.student_name = 'mobilenetV1'
  # # config.student_init = 'gs://imagenet_distill/big_vision/imgnet1k/s-mbnet-bit1k/lr003-wd00003/12-17_0040/checkpoint.npz'
  # # config.student_init = 'gs://imagenet_distill/big_vision/imgnet1k/mbnet-scratch/lr045_wd00003-bs512/01-09_1819/checkpoint.npz'
  # # config.student_init = 'gs://imagenet_distill/big_vision/places_balanced/s-mbnet-vit1k_person/lr000375/02-03_1804/checkpoint.npz'
  # config.student_init = "gs://imagenet_distill/big_vision/imgnet1k/s-mbnet-vit1k/lr003-wd00003/12-22_1957/checkpoint.npz" #best
  # # config.student_init = 'gs://imagenet_distill/big_vision/places/s-mbnet-bit21k/02-07_0409/checkpoint.npz'

  # config.student_init = 'gs://imagenet_distill/big_vision/vww/s-mbnet-pretrain-vit1k/12-27_1422/checkpoint.npz'
  # config.student = dict(use_bn=True)
  # config.student_load = dict(dont_load=['head/kernel', 'head/bias'])
  # config.student_load = dict(dont_load=['head/kernel', 'head/bias', '^(DepthwiseSeparable_)[0-9]*(\/bn_)../[a-z]*$', '^(BatchNorm_0/)[a-z]*$', 'Conv_0/bias'])

  config.teachers = ['prof_m']

  config.prof_m_name = 'proj.flexi.vit'
  config.prof_m = dict(
      variant='B',
      pool_type='tok',
      posemb='learn',
      # patch_size=(32, 32),
      patch_size=(8, 8),
      posemb_size=(7, 7),
      seqhw=None,  # Dynamic!
  )
  config.prof_m_init = 'gs://imagenet_distill/vww/flexivit-b-1kpretrain/03-19_2203/checkpoint.npz'

  # Define the model parameters which are flexible:
  config.flexi = dict()

  num_train_images = 74504
  total_steps = config.total_epochs * (num_train_images // config.input.batch_size)
  evenly_div_steps = total_steps // 6 #len of v

  config.flexi.seqhw = dict(
      # The settings to sample from. Corresponding patch-sizes at 240px:
      # 48, 40, 30, 24, 20, 16, 15, 12, 10, 8
      v=(5, 6, 8, 10, 12, 15),
      # The probabilities/weights of them. Default uniform.
      steps=(0, evenly_div_steps, evenly_div_steps*2, evenly_div_steps*3, 
             evenly_div_steps*4, evenly_div_steps*5),
  )


  # if student res is set then the image is resized to that resolution
  #only for the studnet
  config.student_res = 96

  teacher_hres = 240
  teacher_lres = 240
  crop = f'inception_crop({teacher_lres})'

  # Preprocessing pipeline for student & teacher.
  pp_common = (
      f'|onehot({config.num_classes}, key="label", key_result="labels")'
      f'|keep("image", "labels", "{config.teachers[0]}")'
  )
  config.input.pp = (
    f'copy("image/encoded","image")|copy("image/class/label","label")|drop("image/encoded","image/class/label")|{crop}|flip_lr|randaug|'
    f'|value_range(-1, 1)|copy("image","prof_m")|resize_small({config.student_res})' 
  )+ pp_common
  ppv = f'copy("image/encoded","image")|copy("image/class/label","label")|drop("image/encoded", "image/class/label")|resize_small({teacher_hres})|central_crop({teacher_lres})|value_range(-1, 1)' + pp_common

  if config.student_res is not None:
    ppv_student = f'copy("image/encoded","image")|copy("image/class/label","label")|drop("image/encoded", "image/class/label")|resize_small({teacher_hres})|central_crop({teacher_lres})|resize_small({config.student_res})|value_range(-1, 1)' + pp_common
  else:
    ppv_student = ppv

  # config.mixup = dict(p=0.2, n=2)


  # Distillation settings
  config.distance = 'kl'
  config.distance_kw = dict(t={'fast': 10., 'medium': 1., 'long': 1.}[arg.variant])

  # Optimizer section
  config.grad_clip_norm = 1.0
  config.optax_name = 'scale_by_adam'
  config.optax = dict(mu_dtype='bfloat16')

  config.lr = ({'fast': 0.001, 'medium': 0.0003, 'long': 0.0001}[arg.variant] / 512) * config.input.batch_size
  config.wd = ({'fast': 3e-5, 'medium': 1e-5, 'long': 1e-6}[arg.variant] / 512) * config.input.batch_size
  config.schedule = dict(warmup_steps=1500, decay_type='cosine')
  config.optim_name = 'adam_hp'

  # Eval section
  minitrain_split = 'train[:512]' if not arg.runlocal else 'train[:16]'
  val_split = 'train[90%:]' if not arg.runlocal else 'train[:16]'
  test_split = 'val' if not arg.runlocal else 'test[:16]'

  def get_eval(split):
    return dict(
        type='classification',
        pred='student_fwd',
        data=dict(name=config.input.data.name, split=split),
        pp_fn=ppv_student,
        loss_name='softmax_xent',
        log_steps=500,
    )
  config.evals = {}
  config.evals.student_train = get_eval(minitrain_split)
  config.evals.student_val = get_eval(val_split)
  config.evals.student_test = get_eval(test_split)

  # Teacher is fixed, so rare evals.
  teacher = dict(log_steps=100_000, pp_fn=ppv)
  def get_flexi_eval(s, split):
    return dict(
        type='classification',
        pred=f'prof_m_seqhw={s}',
        data=dict(name=config.input.data.name, split=split),
        pp_fn=ppv,
        loss_name='softmax_xent',
        log_steps=100_000,
    )

  for s in config.flexi.seqhw['v']:
    config.evals[f'teacher_train{s:02d}'] = get_flexi_eval(s, minitrain_split)
    config.evals[f'teacher_val{s:02d}'] = get_flexi_eval(s, val_split)
    config.evals[f'teacher_test{s:02d}'] = get_flexi_eval(s, test_split)

  

  # Make a few things much smaller for quick local debugging testruns.
  if arg.runlocal:
    config.input.shuffle_buffer_size = 10
    config.input.batch_size = 8

  return config