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

"""Most common teachers for distillation."""

# pylint: disable=line-too-long
inits = {  # pylint: disable=duplicate-key Internally, we override some paths for convenience.
    'BiT-M R152x2 imagenet2012 ic224': 'gs://bit_models/distill/R152x2_T_224.npz',
    'BiT-M R152x2 imagenet2012 rc384': 'gs://bit_models/distill/R152x2_T_384.npz',
    'BiT-M R152x2 flowers rc128': 'gs://bit_models/distill/R152x2_T_flowers128.npz',
    'BiT-M R152x2 pet rc128': 'gs://bit_models/distill/R152x2_T_pet128.npz',
    'BiT-M R152x2 food rc128': 'gs://bit_models/distill/R152x2_T_food128.npz',
    'BiT-M R152x2 sun rc128': 'gs://bit_models/distill/R152x2_T_sun128.npz',
    'ViT B/32 flowers': 'gs://imagenet_distill/big_vision/vit_transfer/mixup_inceptioncrop/checkpoint.npz',
    'ViT B/32 pet': 'gs://imagenet_distill/pets/vit-1k-b32-128/checkpoint.npz',
    'ViT B/16 pet': 'gs://vit_models/augreg/B_16-i1k-300ep-lr_0.001-aug_strong2-wd_0.03-do_0.1-sd_0.1--oxford_iiit_pet-steps_0k-lr_0.003-res_224.npz',
    'vww-vit-i21k-augreg-b': 'gs://imagenet_distill/big_vision/vww/vww-vit-i21k-augreg-b/checkpoint.npz',
    'BiT-M R152x2 pet rc224': 'gs://imagenet_distill/pets/bit-m-r152x2-224res/checkpoint.npz',
    'MLPMixer L/16 pet': 'gs://imagenet_distill/pets/mlpmixer-l16-224res/checkpoint.npz',
    'MobileNetV2 120d pet': 'gs://imagenet_distill/pets/mobilenetv2_120d/6000steps/checkpoint.npz',
}
# pylint: enable=line-too-long
