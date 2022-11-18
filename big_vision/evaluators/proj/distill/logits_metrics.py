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

"""Evaluator for the classfication task."""
# pylint: disable=consider-using-from-import
from functools import partial, lru_cache

import big_vision.datasets.core as ds_core
import big_vision.input_pipeline as input_pipeline
import big_vision.pp.builder as pp_builder
import big_vision.utils as u

import jax
import jax.numpy as jnp
import numpy as np


import matplotlib.pyplot as plt

# To avoid re-compiling the function for every new instance of the same
# evaluator on a different dataset!
@lru_cache(None)
def get_eval_fn(predict_fn):
  """Produces eval function, also applies pmap."""
  @partial(jax.pmap, axis_name='batch')
  def _eval_fn(params, batch):

    logits, *_ = predict_fn(params, **batch)

    #mean of logits
    mean_logits = jnp.mean(logits)
    
    # Extracts the confidence at the highest logit index for each image.
    topk_confidence, _ = jax.lax.top_k(logits, k=2)

    top1_confidence = jnp.max(topk_confidence, axis=1)

    #Max, min, mean of confidence
    max_top1_confidence = jnp.max(top1_confidence)
    min_top1_confidence = jnp.min(top1_confidence)
    mean_top1_confidence = jnp.mean(top1_confidence)

    
    top2_confidence = jnp.min(topk_confidence, axis=1)
    #Max, min, mean of second highest confidence
    max_top2_confidence = jnp.max(top2_confidence)
    min_top2_confidence = jnp.min(top2_confidence)
    mean_top2_confidence = jnp.mean(top2_confidence)

    #dict of all the metrics
    metrics = {
        'mean_logits': mean_logits,
        'max_top1_confidence': max_top1_confidence,
        'min_top1_confidence': min_top1_confidence,
        'mean_top1_confidence': mean_top1_confidence,
        'max_top2_confidence': max_top2_confidence,
        'min_top2_confidence': min_top2_confidence,
        'mean_top2_confidence': mean_top2_confidence,
    }
  
    return metrics
  return _eval_fn

#different func for mixup
@lru_cache(None)
def get_eval_fn_mixup(predict_fn, p, n):
  """Produces eval function, also applies pmap."""
  @partial(jax.pmap, axis_name='batch')
  def _eval_fn(params, batch):
    rng = jax.random.PRNGKey(1) #keep fixed seed. 0 is really bad for mixup
    rng, _, batch = u.mixup(rng, p=p, n=n, **batch)
    logits, *_ = predict_fn(params, **batch)

    #mean of logits
    mean_logits = jnp.mean(logits)
    
    # Extracts the confidence at the highest logit index for each image.
    topk_confidence, _ = jax.lax.top_k(logits, k=2)

    top1_confidence = jnp.max(topk_confidence, axis=1)

    #Max, min, mean of confidence
    max_top1_confidence = jnp.max(top1_confidence)
    min_top1_confidence = jnp.min(top1_confidence)
    mean_top1_confidence = jnp.mean(top1_confidence)

    
    top2_confidence = jnp.min(topk_confidence, axis=1)
    #Max, min, mean of second highest confidence
    max_top2_confidence = jnp.max(top2_confidence)
    min_top2_confidence = jnp.min(top2_confidence)
    mean_top2_confidence = jnp.mean(top2_confidence)

    #dict of all the metrics
    metrics = {
        'mean_logits': mean_logits,
        'max_top1_confidence': max_top1_confidence,
        'min_top1_confidence': min_top1_confidence,
        'mean_top1_confidence': mean_top1_confidence,
        'max_top2_confidence': max_top2_confidence,
        'min_top2_confidence': min_top2_confidence,
        'mean_top2_confidence': mean_top2_confidence,
    }
  
    return metrics
  return _eval_fn


class Evaluator:
  """Logits Metrcis evaluator."""

  def __init__(self, predict_fn, data, pp_fn, batch_size,
               cache_final=True, cache_raw=False, prefetch=1, mixup=False, config_mixup=None):
    data = ds_core.get(**data)
    pp_fn = pp_builder.get_preprocess_fn(pp_fn)
    self.ds, self.steps = input_pipeline.make_for_inference(
        data.get_tfdata(ordered=True), pp_fn, batch_size,
        num_ex_per_process=data.num_examples_per_process(),
        cache_final=cache_final, cache_raw=cache_raw)
    self.data_iter = input_pipeline.start_input_pipeline(self.ds, prefetch)
    if mixup:
      self.eval_fn = get_eval_fn_mixup(predict_fn, config_mixup['p'], config_mixup['n'])
    else:
      self.eval_fn = get_eval_fn(predict_fn)
    self.mixup = mixup
    self.config_mixup = config_mixup

  def run(self, params):
    """Computes all metrics."""
    logit_metrics = None
    for _, batch in zip(range(self.steps), self.data_iter):
      mask = batch.pop('_mask') #need to pop mask to avoid error in fwd()

      if logit_metrics is None:
        logit_metrics = self.eval_fn(params, batch) #first batch
      else:
        batch_logit_metrics = self.eval_fn(params, batch)
        #accumulate metrics across batches and devices
        for k, v in batch_logit_metrics.items(): 
          logit_metrics[k] = jnp.concatenate([logit_metrics[k], v])

    #final metric calculation across all data
    for k, v in logit_metrics.items():
      if 'mean' in k: #calculate over devices
        v = jnp.mean(v)
      elif 'max' in k:
        v = jnp.max(v)
      elif 'min' in k:
        v = jnp.min(v)
      yield(k, np.array(v).item())
    
