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


big_vision.trainers.proj.distill.distill \
    --config big_vision/configs/proj/distill/vww.py:variant=fast \
    --workdir gs://[your_bucket]/big_vision/`date '+%m-%d_%H%M'` \
"""

import big_vision.configs.common as bvcc
import big_vision.configs.proj.distill.common as cd
import ml_collections as mlc


def get_config(arg=None):
  """Config for massive hypothesis-test on vww"""
  arg = bvcc.parse_arg(arg, runlocal=False, data='vww', variant='medium')
  config = mlc.ConfigDict()

  config.input = {}
  config.input.data = dict(
      name='vww',
      split='train[:90%]',
  )
  config.input.batch_size = 512
  config.input.cache_raw = True
  config.input.shuffle_buffer_size = 50_000
  config.prefetch_to_device = 4

  config.num_classes = 2
  config.total_epochs = {'fast': 1_000, 'medium': 10_000, 'long': 100_000}[arg.variant]

  config.log_training_steps = 100
  config.ckpt_steps = 2500

  # Model section
  config.student_name = 'cnn'#'bit_paper'
  config.student = dict()#dict(depth=26, width=0.5)

  config.teachers = ['prof_m']

#TODO change
  # config.prof_m_name = 'bit_paper'
  # config.prof_m_init = cd.inits[f'BiT-M R152x2 {arg.data} rc128']
  # config.prof_m = dict(depth=152, width=2)

  config.prof_m_name = 'vit' 
  config.prof_m_init = cd.inits['vww-vit-i21k-augreg-b']
  config.prof_m = dict(variant='B/32', pool_type='tok')

  # if student res is set then the image is resized to that resolution
  #only for the studnet
  config.student_res = 96

  teacher_hres = 448
  teacher_lres = 384
  crop = f'inception_crop({teacher_lres})'

  # Preprocessing pipeline for student & teacher.
  pp_common = (
      f'|onehot({config.num_classes}, key="label", key_result="labels")'
      f'|keep("image", "labels", "{config.teachers[0]}")'
  )
  config.input.pp = (
    f'copy("image/encoded","image")|copy("image/class/label","label")|drop("image/encoded","image/class/label")|{crop}|flip_lr'
    f'|value_range(-1, 1)|copy("image","prof_m")|resize_small({config.student_res})' 
  )+ pp_common
  ppv = f'copy("image/encoded","image")|copy("image/class/label","label")|drop("image/encoded", "image/class/label")|resize_small({teacher_hres})|central_crop({teacher_lres})|value_range(-1, 1)' + pp_common

  if config.student_res is not None:
    ppv_student = f'copy("image/encoded","image")|copy("image/class/label","label")|drop("image/encoded", "image/class/label")|resize_small({teacher_hres})|central_crop({teacher_lres})|resize_small({config.student_res})|value_range(-1, 1)' + pp_common
  else:
    ppv_student = ppv

  config.mixup = dict(p=0.2, n=2)


  # Distillation settings
  config.distance = 'kl'
  config.distance_kw = dict(t={'fast': 10., 'medium': 1., 'long': 1.}[arg.variant])

  # Optimizer section
  config.grad_clip_norm = 1.0
  config.optax_name = 'scale_by_adam'
  config.optax = dict(mu_dtype='bfloat16')

  config.lr = {'fast': 0.001, 'medium': 0.0003, 'long': 0.0001}[arg.variant]
  config.wd = None #{'fast': 3e-4, 'medium': 1e-4, 'long': 1e-5}[arg.variant]
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