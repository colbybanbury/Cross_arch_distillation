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
r"""Distilling BiT-R152x2 into BiT-R50x1 on Flowers/Pet as in https://arxiv.org/abs/2106.05237

While many epochs are required, this is a small dataset, and thus overall it
is still fast and possible to run on the relatively small v3-8TPUs (or GPUs).

This configuration contains the recommended settings from Fig3/Tab4 of the
paper, which can be selected via the fast/medium/long config argument.
(best settings were selected on a 10% minival)

For Flowers:
- The `fast` variant takes ~1h10m on a v2-8 TPU.
  Example logs at gs://big_vision/distill/bit_flowers_fast_06-18_2008/big_vision_metrics.txt
- The `long` variant takes ~25h on a v3-32 TPU.
  Example logs at gs://big_vision/distill/bit_flowers_long_06-19_0524/big_vision_metrics.txt
For Pet:
- The `fast` variant takes ~28min on a v2-8 TPU.
  Example logs at gs://big_vision/distill/bit_pet_fast_06-16_2338/big_vision_metrics.txt
- The `long` variant takes ~11h on a v2-8 and ~8h on a v3-32.
  Example logs at gs://big_vision/distill/bit_pet_long_06-17_0050/big_vision_metrics.txt

big_vision.trainers.proj.distill.distill \
    --config big_vision/configs/proj/distill/bigsweep_flowers_pet.py:data=flowers,variant=fast \
    --workdir gs://[your_bucket]/big_vision/`date '+%m-%d_%H%M'` \
"""

import big_vision.configs.common as bvcc
import big_vision.configs.proj.distill.common as cd
import ml_collections as mlc

NCLS = dict(flowers=102, pet=37)


def get_config(arg=None):
  """Config for massive hypothesis-test on pet."""
  arg = bvcc.parse_arg(arg, runlocal=False, data='flowers', variant='fast', student='mobilenetv2', teacher='vit', lr=-1.0, wd=-1.0)
  config = mlc.ConfigDict()

  config.input = {}
  config.input.data = dict(
      name=dict(flowers='oxford_flowers102', pet='oxford_iiit_pet')[arg.data],
      split=dict(flowers='train', pet='train[:90%]')[arg.data],
  )
  config.input.batch_size = 256
  config.input.cache_raw = True
  config.input.shuffle_buffer_size = 25_000
  config.prefetch_to_device = 4

  config.num_classes = NCLS[arg.data]
  config.total_epochs = {
      'flowers': {'fast': 25_000, 'medium': 100_000, 'long': 1_000_000},
      'pet': {'fast': 1000, 'medium': 3000, 'long': 30_000},
  }[arg.data][arg.variant]

  config.log_training_steps = 100
  config.ckpt_steps = 2500

  config.log_name = f"{arg.student}-{arg.teacher}"
  if arg.lr != "None":
    config.log_name += f"-lr{arg.lr}"
  if arg.wd != "None":
    config.log_name += f"-wd{arg.wd}"

  # Model section
  if arg.student == 'cnn':
    config.student_name = 'cnn'
    config.student = dict(use_bn=True)
  elif arg.student == 'mobilenetv1':
    config.student_name = 'mobilenetV1'
    config.student = dict(use_bn=True)
  elif arg.student == 'mobilenetv2':
    config.student_name = 'efficientnet_jax_wrapper'
    config.student = dict(variant='pt_mobilenetv2_035')
  elif arg.student == 'mobilenetv3':
    config.student_name = 'efficientnet_jax_wrapper'
    config.student = dict(variant='pt_mobilenetv3_small_035')


  # config.student_name = 'proj.flexi.vit'
  # config.student_init = 'gs://big_vision/flexivit/flexivit_b_i1k.npz'
  # config.student = dict(
  #     variant='B',
  #     pool_type='tok',
  #     posemb='learn',
  #     # patch_size=(32, 32),
  #     patch_size=(8, 8),
  #     posemb_size=(7, 7),
  #     seqhw=None,  # Dynamic!
  # )
  # config.student_load = dict(dont_load=['head/kernel', 'head/bias'])

  # config.flexi = dict()
  # config.flexi.seqhw = dict(
  #     # The settings to sample from. Corresponding patch-sizes at 240px:
  #     # 48, 40, 30, 24, 20, 16, 15, 12, 10, 8
  #     v=(5, 6, 8, 10, 12, 15, 16, 20, 24, 30),
  #     # at 128px:
  #     # 64, 32, 16, 8
  #     # v = (2, 4, 8, 16),
  #     # The probabilities/weights of them. Default uniform.
  #     p=(1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
  # ) 


  config.teachers = ['prof_m']

  if arg.teacher == 'bit':
    config.prof_m_name = 'bit_paper'
    config.prof_m_init = cd.inits[f'BiT-M R152x2 {arg.data} rc128']
    config.prof_m = dict(depth=152, width=2)
  elif arg.teacher == 'vit-b32':
    config.prof_m_name = 'vit' 
    config.prof_m_init = cd.inits[f'ViT B/32 {arg.data}']
    config.prof_m = dict(variant='B/32', pool_type='tok')
  elif arg.teacher == 'vit-b16':
    config.prof_m_name = 'vit' 
    config.prof_m_init = cd.inits[f'ViT B/16 {arg.data}']
    config.prof_m = dict(variant='B/16', pool_type='tok')
  elif arg.teacher == 'bit-224':
    config.prof_m_name = 'bit_paper'
    config.prof_m_init = cd.inits[f'BiT-M R152x2 {arg.data} rc224']
    config.prof_m = dict(depth=152, width=2)
  elif arg.teacher == 'mlpmixer-L16':
    config.prof_m_name = 'mlp_mixer'
    config.prof_m_init = cd.inits[f'MLPMixer L/16 {arg.data}']
    config.prof_m = dict(variant='L/16')
  elif arg.teacher == 'mobilenetv2-120d':
    config.prof_m_name = 'efficientnet_jax_wrapper'
    config.prof_m = dict(variant='pt_mobilenetv2_120d')
    config.prof_m_init = cd.inits[f'MobileNetV2 120d {arg.data}']



  # if student res is set then the image is resized to that resolution
  #only for the studnet
  config.student_res = 96

  hres = 256
  lres = 224

  crop='inception_crop(224)'

  teacher_norm = 'value_range'
  if arg.teacher == "mobilenetv2-120d":
    teacher_norm = 'normalize'

  # Preprocessing pipeline for student & teacher.
  pp_common = (
      f'|onehot({config.num_classes}, key="label", key_result="labels")'
      f'|keep("image", "labels", "{config.teachers[0]}")'
  )

  config.input.pp = (
    f'decode|{crop}|flip_lr|randaug(2,10)'
    f'|{teacher_norm}(inkey="image", outkey="prof_m")|value_range(-1, 1)|resize_small({config.student_res})' 
  )+ pp_common
  ppv = f'decode|resize_small({hres})|central_crop({lres})|{teacher_norm}' + pp_common

  if config.student_res is not None:
    ppv_student = f'decode|resize_small({hres})|central_crop({lres})|resize_small({config.student_res})|value_range(-1, 1)' + pp_common
  else:
    ppv_student = ppv

  config.mixup = dict(p=1.0, n=2)


  # Distillation settings
  config.distance = 'kl'
  config.distance_kw = dict(t={
      'flowers': {'fast': 1., 'medium': 1., 'long': 1.},
      'pet': {'fast': 5., 'medium': 5., 'long': 2.},
  }[arg.data][arg.variant])

  # Optimizer section
  config.grad_clip_norm = 1.0
  config.optax_name = 'scale_by_adam'
  config.optax = dict(mu_dtype='bfloat16')

  if arg.lr <= 0:
    config.lr = {
        'flowers':{'fast': 0.001, 'medium': 0.001, 'long': 0.0003}, #{'fast': 5e-5}
        'pet': {'fast': 0.01, 'medium': 0.003, 'long': 0.003},
    }[arg.data][arg.variant]
  else:
    config.lr = arg.lr
  if arg.wd <= 0:
    config.wd = {
        'flowers': {'fast': 3e-4, 'medium': 1e-4, 'long': 1e-5}, # {'fast': 5e-6},
        'pet': {'fast': 1e-3, 'medium': 3e-4, 'long': 1e-5},
    }[arg.data][arg.variant]
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


  # # flexi evals
  # def get_eval(s, split):
  #   return dict(
  #       type='classification',
  #       pred=f'student_seqhw={s}',
  #       data=dict(name=config.input.data.name, split=split),
  #       pp_fn=ppv_student,
  #       loss_name='sigmoid_xent',
  #       log_steps=1000,
  #   )
  # config.evals = {}
  # for s in config.flexi.seqhw.v:
  #   config.evals[f'train{s:02d}'] = get_eval(s, minitrain_split)
  #   config.evals[f'val{s:02d}'] = get_eval(s, val_split)
  #   config.evals[f'test{s:02d}'] = get_eval(s, test_split)

  # def get_t_eval(split):
  #   return dict(
  #       type='classification',
  #       pred='prof_m',
  #       data=dict(name=config.input.data.name, split=split),
  #       pp_fn=ppv,
  #       loss_name='softmax_xent',
  #       log_steps=100_000,
  #   )
  
  # config.evals.teacher_train = get_t_eval(minitrain_split)
  # config.evals.teacher_val = get_t_eval(val_split)
  # config.evals.teacher_test = get_t_eval(test_split)


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
  teacher = dict(log_steps=100_000, pred='prof_m_fwd', pp_fn=ppv)
  config.evals.teacher_train = {**config.evals.student_train, **teacher}
  config.evals.teacher_val = {**config.evals.student_val, **teacher}
  config.evals.teacher_test = {**config.evals.student_test, **teacher}

  # Could in principle also look at agreement on other datasets!
#disableing for now to get different resolitons for teacher and student
  # def get_dist(split):
  #   return dict(
  #       type='proj.distill.distance',
  #       pred='student_prof_m_fwd',
  #       data=dict(name=config.input.data.name, split=split),
  #       pp_fn=ppv + '|keep("image")',
  #       log_steps=1000,
  #       distances=({'kind': 'kl'}, {'kind': 'euclidean'},
  #                  {'kind': 'agree', 'k': 1}, {'kind': 'agree', 'k': 5}),
  #   )
  # config.evals.dist_train = get_dist(minitrain_split)
  # config.evals.dist_val = get_dist(val_split)
  # config.evals.dist_test = get_dist(test_split)

  # Make a few things much smaller for quick local debugging testruns.
  if arg.runlocal:
    config.input.shuffle_buffer_size = 10
    config.input.batch_size = 8

  return config