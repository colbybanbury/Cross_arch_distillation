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

"""Training loop for distillation as in https://arxiv.org/abs/2106.05237.

It works by having a (set of) teacher model(s) defined the same way as student
in the config and, for now, only distilling logits with one of many loss
functions.

We explored distilling intermediate feature maps, extra data, and other tricks
in depth in two interships in a separate prototype codebase but eventually they
are not necessary, and thus not (yet?) implemented in this codebase.

Thus, for now, there are no extra learnable parameters besides the student.
This keeps code relatively simple.
"""
# pylint: disable=consider-using-from-import
from functools import partial
import importlib
import multiprocessing.pool
import os

from absl import app
from absl import flags
from absl import logging
import big_vision.datasets.core as ds_core
import big_vision.evaluators.common as eval_common
import big_vision.evaluators.proj.distill.distance as dd
import big_vision.input_pipeline as input_pipeline
import big_vision.optax as bv_optax
import big_vision.pp.builder as pp_builder
import big_vision.utils as u
from clu import parameter_overview
import flax
import jax
import jax.numpy as jnp
from ml_collections import config_flags
import numpy as np
import optax
import tensorflow as tf
from tensorflow.io import gfile

import big_vision.pp.ops_image as pp_ops_image

# pylint: disable=logging-fstring-interpolation


config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=True)

flags.DEFINE_string("workdir", default=None, help="Work unit directory.")
flags.DEFINE_boolean("cleanup", default=False,
                     help="Delete workdir (only) after successful completion.")

# Adds jax flags to the program.
jax.config.parse_flags_with_absl()


def getfirst(d, *keys, default=None):
  """Returns the first of `keys` that's present in mapping `d`."""
  for k in reversed(keys):
    default = d.get(k, default)
  return default


def main(argv):
  del argv
  tf.config.experimental.set_visible_devices([], "GPU")

  config = flags.FLAGS.config
  workdir = flags.FLAGS.workdir
  logging.info(
      f"\u001b[33mHello from process {jax.process_index()} holding "
      f"{jax.local_device_count()}/{jax.device_count()} devices and "
      f"writing to workdir {workdir}.\u001b[0m")

  save_ckpt_path = None
  if workdir:  # Always create if requested, even if we may not write into it.
    gfile.makedirs(workdir)
    save_ckpt_path = os.path.join(workdir, "checkpoint.npz")

  # The pool is used to perform misc operations such as logging in async way.
  pool = multiprocessing.pool.ThreadPool()

  # Here we register preprocessing ops from modules listed on `pp_modules`.
  for m in config.get("pp_modules", ["ops_general", "ops_image", "ops_text"]):
    importlib.import_module(f"big_vision.pp.{m}")

  # This seed makes the Jax part of things (like model init) deterministic.
  # However, full training still won't be deterministic, for example due to the
  # tf.data pipeline not being deterministic even if we would set TF seed.
  # See (internal link) for a fun read on what it takes.
  rng = jax.random.PRNGKey(config.get("seed", 0))

  # These functions do more stuff internally, for OSS release we mock them by
  # trivial alternatives in order to minize disruptions in the code.
  xid, wid = -1, -1
  fillin = lambda s: s
  def info(s, *a):
    logging.info("\u001b[33mNOTE\u001b[0m: " + s, *a)
  def write_note(note):
    if jax.process_index() == 0:
      info("%s", note)

  write_note("Initializing...")

  batch_size = config.input.batch_size
  if batch_size % jax.device_count() != 0:
    raise ValueError(f"Batch size ({batch_size}) must "
                     f"be divisible by device number ({jax.device_count()})")
  info("Global batch size %d on %d hosts results in %d local batch size. With "
       "%d dev per host (%d dev total), that's a %d per-device batch size.",
       batch_size, jax.process_count(), batch_size // jax.process_count(),
       jax.local_device_count(), jax.device_count(),
       batch_size // jax.device_count())

  # First thing after above sanity checks, so we can log "start" ticks.
  mw = u.BigVisionMetricWriter(xid, wid, workdir, config)
  chrono = u.Chrono()

  write_note("Initializing train dataset...")
  train_data = ds_core.get(**config.input.data)
  train_ds = train_data.get_tfdata(ordered=False)

  #if we are using places and we are doing binary classification (person detection)
  #we need to balance the dataset
  if config.input.data.get("name") == "places365_small" and config.num_classes == 2:
    tfds_ids = np.load("balanced_places_tfds_ids.npy", allow_pickle=True)
    keys = tf.constant(tfds_ids)
    vals = tf.ones_like(keys, dtype=tf.int32)
    ht = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys, vals), default_value=0)

    @tf.function
    def filter_fn(x):
      return ht.lookup(x["tfds_id"]) == 1
    
    train_ds = train_ds.filter(filter_fn)
    write_note("Places dataset balanced for binary classification")


  train_ds = input_pipeline.make_for_train(
      data=train_ds,
      batch_size=batch_size,
      preprocess_fn=pp_builder.get_preprocess_fn(config.input.get("pp")),
      shuffle_buffer_size=config.input.get("shuffle_buffer_size"),
      cache_raw=config.input.get("cache_raw", False),
      filter_fn=config.input.get("filter_fn"),
  )

  # Start prefetching already.
  n_prefetch = config.get("prefetch_to_device", 1)
  train_iter = input_pipeline.start_input_pipeline(train_ds, n_prefetch)
  ntrain_img = train_data.total_examples

  def get_steps(name, default=ValueError):  # partial doesn't work well here.
    return u.steps(name, config, ntrain_img, batch_size, default)
  total_steps = get_steps("total")

  info("Running for %d steps, that means %f epochs",
       total_steps, total_steps * batch_size / ntrain_img)

  # Create student and teacher models
  def get_model_mod(name):  # Used many times.
    mod_name = config[f"{name}_name"]
    return importlib.import_module(f"big_vision.models.{mod_name}")

  write_note("Initializing models...")
  def make_model(name):
    return get_model_mod(name).Model(
        num_classes=config.num_classes, **config.get(name, {}))

  models = {
      "student": make_model("student"),
      **{t: make_model(t) for t in config.teachers}
  }

  # We want all parameters to be created in host RAM, not on any device, they'll
  # be sent there later as needed, otherwise we already encountered two
  # situations where we allocate them twice.
  def get_init(model, train=False):
    #TODO fix this. Had to remove jit in order to have diff res for the teacher and student 
    # @partial(jax.jit, backend="cpu")
    def _init(rng, shape):
      bs = batch_size // jax.device_count()
      dummy_input = jnp.zeros((bs,) + shape, jnp.float32)
      params = flax.core.unfreeze(model.init(rng, dummy_input, train=False))

      # Set bias in the head to a low value, such that loss is small initially.
      if "init_head_bias" in config:
        params["params"]["head"]["bias"] = jnp.full_like(params["params"]["head"]["bias"],
                                               config["init_head_bias"])
      return params
    return _init

  rng, *rng_inits = jax.random.split(rng, len(models) + 1)
  if config.get('student_res'): #student and teacher have different resolutions
    shapes = [tuple(train_ds.element_spec[name].shape[1:]) for name in ["image"] + config.teachers]
  else:
    shapes = [tuple(train_ds.element_spec["image"].shape[1:])] * len(models)
  print(shapes)
  
  params_cpu = {name: get_init(models[name])(rngi, shape)
                for name, rngi, shape in zip(models, rng_inits, shapes)}

  if jax.process_index() == 0:
    for name, params in params_cpu.items():
      parameter_overview.log_parameter_overview(params, msg=f"{name} params")
      mw.measure(f"num_params_{name}",
                 sum(p.size for p in jax.tree_leaves(params)))

  write_note(f"Initializing {config.optax_name} optimizer...")
  # For now, we explicitly only optimize the student parameters as there's
  # nothing else to be optimized. If we ever want to add learnable projections
  # or similar for good (we explored but ditched), need to refactor this a bit.
  tx, sched_fns = bv_optax.make(
      config, params_cpu["student"]["params"], sched_kw=dict(
          total_steps=total_steps, batch_size=batch_size, data_size=ntrain_img))

  # We jit this, such that the arrays are created on the CPU, not device[0].
  opt_cpu = jax.jit(tx.init, backend="cpu")(params_cpu["student"]["params"])
  sched_fns_cpu = [jax.jit(sched_fn, backend="cpu") for sched_fn in sched_fns]

  @jax.named_call
  def loss_fn(student_params, params, data, rngs, reduce=True):
    # Note: need to extract and use `student_params` out of `params` because the
    # first argument of `loss_fn` is what's differentiated wrt.
    params["student"]["params"] = student_params

    def fwd(name, params):
      return jax.named_call(models[name].apply, name=name)(
          params, getfirst(data, name, "image"),
          train=name == "student", rngs=rngs.get(name),
          mutable=["batch_stats"] if name == "student" else False  #allows the use of batch norm in the student
      )
    logits = {name: fwd(name, w) for name, w in params.items()}

    mutated_vars = logits["student"][1] #unpack the mutated vars

    #get rid of the mutated vars, which for some reason all models return
    logits = {name: lg[0] for name, lg in logits.items()}
    
    measurements = {}
    for name, lg in logits.items():
      measurements[f"entropy_{name}"] = -jnp.sum(
          jax.nn.log_softmax(lg) * jax.nn.softmax(lg), axis=-1)
      if "labels" in data:
        measurements[f"task_loss_{name}"] = u.softmax_xent(
            logits=lg, labels=data["labels"], reduction=False)

    # NOTE: xent is linear in labels, so for KL, this is actually the same as
    # using a teacher-ensemble in probs-space!
    measurements["distill_loss"] = 0.0
    for name in config.teachers:
      l = dd.dist(logits["student"], logits[name], config.get("distance", "kl"),
                  **config.get("distance_kw", {}))
      measurements[f"distill_loss_{name}"] = l
      measurements["distill_loss"] += l

    #Measure agreement between student and teacher
    if config.get("measure_agreement"):
      for name in config.teachers:
        measurements[f"agreement_top1_{name}"] = dd.dist(logits["student"], logits[name], "agree",k=1)
        if config.num_classes > 5:
          measurements[f"agreement_top5_{name}"] = dd.dist(logits["student"], logits[name], "agree",k=5)

    outputs = (measurements["distill_loss"], (measurements, mutated_vars))
    return jax.tree_map(jnp.mean, outputs) if reduce else outputs

  @partial(jax.pmap, axis_name="batch", donate_argnums=(0, 1))
  def update_fn(params, opt, rng, data):
    """Update step."""

    # Mixup. Note: overwrites the `data` entries (that's intended).
    if config.get("mixup") and config.mixup.p:
      to_mix = {name: data[name]
                for name in ("image", "labels") + tuple(models) if name in data}
      rng, _, to_mix = u.mixup(rng, **config.mixup, **to_mix)
      data = {**data, **to_mix}

    # Get device-specific loss rng.
    rng, *rng_models = jax.random.split(rng, len(models) + 1)
    rngs_models_local = {
        name: {"dropout": jax.random.fold_in(rngi, jax.lax.axis_index("batch"))}
        for name, rngi in zip(models, rng_models)
    }

    w = params["student"]["params"]  # Need to explicitly pull out the optimized ones.
    (l, (measurements, batch_stats)), grads = jax.lax.pmean(
        jax.value_and_grad(loss_fn, has_aux=True)(
            w, params, data, rngs=rngs_models_local),
        axis_name="batch")
    updates, opt = tx.update(grads, opt, w)
    w = optax.apply_updates(w, updates)
    params["student"] = {"params": w, **batch_stats}

    # Take some logging measurements
    gs = jax.tree_leaves(bv_optax.replace_frozen(config.schedule, grads, 0.))
    measurements["l2_grads"] = jnp.sqrt(sum([jnp.vdot(g, g) for g in gs]))
    ps = jax.tree_leaves(w)
    measurements["l2_params"] = jnp.sqrt(sum([jnp.vdot(p, p) for p in ps]))
    us = jax.tree_leaves(updates)
    measurements["l2_updates"] = jnp.sqrt(sum([jnp.vdot(u, u) for u in us]))

    return params, opt, rng, l, measurements

  # We always load the teachers first, because they NEED to be initialized
  # and since we don't ever modify them, we don't store them in checkpoints.
  for name in config.teachers:
    init_def = config[f"{name}_init"]
    write_note(f"Initializing {name} from {init_def}…")
    params_cpu[name]["params"] = get_model_mod(name).load(
        params_cpu[name]["params"], init_def, config[name],
        **config.get(f"{name}_load", {}))

  # Decide how to initialize training. The order is important.
  # 1. Always resumes from the existing checkpoint, e.g. resumes a finetune job.
  # 2. Resume from a previous checkpoint, e.g. start a cooldown training job.
  # 3. Initialize student from something, e.g. start a fine-tuning job.
  # 4. Train from scratch.
  resume_ckpt_path = None
  if save_ckpt_path and gfile.exists(save_ckpt_path):
    resume_ckpt_path = save_ckpt_path
  elif config.get("resume"):
    resume_ckpt_path = fillin(config.resume)
  if resume_ckpt_path:
    write_note("Resume training from checkpoint...")
    # NOTE: we never change the teachers, so only checkpoint student here.
    checkpoint = {"params": params_cpu["student"]["params"],
                  "opt": opt_cpu, "chrono": chrono.save()}
    checkpoint_tree = jax.tree_structure(checkpoint)
    loaded = u.load_checkpoint(checkpoint_tree, resume_ckpt_path)
    # bfloat16 type gets lost when data is saved to disk, so we recover it.
    checkpoint = jax.tree_map(u.recover_dtype, loaded)
    params_cpu["student"]["params"], opt_cpu = checkpoint["params"], checkpoint["opt"]
    chrono.load(checkpoint["chrono"])
  elif config.get("student_init"):
    write_note(f"Initialize student from {config.student_init}...")
    params_cpu["student"]["params"] = get_model_mod("student").load(
        params_cpu["student"]["params"], config.student_init, config.get("student"),
        **config.get("student_load", {}))
    if jax.process_index() == 0:
      parameter_overview.log_parameter_overview(
          params_cpu["student"]["params"], msg="restored (student) params")

  write_note("Kicking off misc stuff...")
  first_step = bv_optax.get_count(opt_cpu)
  chrono.inform(first_step, total_steps, batch_size, ntrain_img / batch_size)
  prof = None  # Keeps track of start/stop of profiler state.

  write_note(f"Replicating...\n{chrono.note}")
  params_repl = flax.jax_utils.replicate(params_cpu)
  opt_repl = flax.jax_utils.replicate(opt_cpu)

  # Initializing evaluators later when they are first needed, so we can see
  # issues with training faster.
  evaluators = None

  # Define predict functions that the evaluators can use:
  # 1. One per model
  predict_fns = {}
  for name, model in models.items():
    def fwd(params, image, n=name, m=model):
      return m.apply(params[n], image, train=False)
    predict_fns[f"{name}_fwd"] = fwd
  # 2. One for the ensemble of all teachers.
  def teacher_ensemble_fwd(params, image):
    all_teacher_logits = [
        models[name].apply(params[name], image)[0]  # return is `logits, out`
        for name in config.teachers
    ]
    return jnp.mean([jax.nn.softmax(l) for l in all_teacher_logits], axis=0), {}
  predict_fns["teacher_ensemble_fwd"] = teacher_ensemble_fwd
  # 3.One for each (student, teacher) pair, eg for distance eval.
  for name in [*config.teachers, "teacher_ensemble"]:
    def fwd(params, image, n=name):  # pylint: disable=function-redefined
      student_ret = predict_fns["student_fwd"](params, image)
      teacher_ret = predict_fns[f"{n}_fwd"](params, image)
      return student_ret, teacher_ret
    predict_fns[f"student_{name}_fwd"] = fwd

  rng, rng_loop = jax.random.split(rng, 2)
  rngs_loop = flax.jax_utils.replicate(rng_loop)
  ckpt_writer = None

  write_note(f"First step compilations...\n{chrono.note}")
  error = None  # For exiting with an error after cleanup. Avoids indentation.

  # Using a python integer for step here, because opt.state.step is allocated
  # on TPU during replication.
  for step, batch in zip(range(first_step + 1, total_steps + 1), train_iter):
    mw.step_start(step)

    with jax.profiler.StepTraceAnnotation("train_step", step_num=step):
      params_repl, opt_repl, rngs_loop, loss_value, measurements = update_fn(
          params_repl, opt_repl, rngs_loop, batch)

    # On the first host, let's always profile a handful of early steps.
    if jax.process_index() == 0:
      prof = u.startstop_prof(prof, step, first_step, get_steps("log_training"))

    # Report training progress
    if (u.itstime(step, get_steps("log_training"), total_steps, host=0)
        or chrono.warmup and jax.process_index() == 0):
      for i, sched_fn_cpu in enumerate(sched_fns_cpu):
        mw.measure(f"global_schedule{i if i else ''}", sched_fn_cpu(step - 1))
      l = mw.measure("training_loss", loss_value[0])
      for name, value in measurements.items():
        mw.measure(name, value[0])
      chrono.tick(step, mw.measure, write_note)
      if not np.isfinite(l):
        error = (f"The loss became nan or inf somewhere within steps "
                 f"[{step - get_steps('log_training')}, {step}]")
        break

    # Checkpoint saving
    if (save_ckpt_path and
        (u.itstime(step, get_steps("ckpt", None), total_steps, host=0) or
         u.itstime(step, get_steps("keep_ckpt", None), total_steps, host=0))):
      chrono.pause(wait_for=(params_repl["student"], opt_repl))
      u.checkpointing_timeout(ckpt_writer, config.get("ckpt_timeout", 1))
      # We need to transfer the weights over now or else we risk keeping them
      # alive while they'll be updated in a future step, creating hard to debug
      # memory errors (see (internal link)). Also, takes device 0's params only.
      params_cpu["student"]["params"], opt_cpu = jax.tree_map(
          lambda x: np.array(x[0]), (params_repl["student"]["params"], opt_repl))

      # Check whether we want to keep a copy of the current checkpoint.
      copy_step = None
      if u.itstime(step, get_steps("keep_ckpt", None), total_steps):
        copy_step = step

      ckpt = {"params": params_cpu["student"]["params"],
              "opt": opt_cpu,
              "chrono": chrono.save()}
      ckpt_writer = pool.apply_async(
          u.save_checkpoint, (ckpt, save_ckpt_path, copy_step))
      chrono.resume()

    if evaluators is None:
      evaluators = eval_common.from_config(
          config, predict_fns,
          lambda s: write_note(f"Initializing evaluator: {s}...\n{chrono.note}")
      )
    for (name, evaluator, log_steps, prefix) in evaluators:
      if u.itstime(step, log_steps, total_steps):
        chrono.pause(wait_for=params_repl)
        write_note(f"{name} evaluation...\n{chrono.note}")
        for key, value in evaluator.run(params_repl):
          mw.measure(f"{prefix}{key}", value)
        chrono.resume()
    mw.step_end()

  # Always give a chance to stop the profiler, no matter how things ended.
  # TODO: can we also do this when dying of an exception like OOM?
  if jax.process_index() == 0 and prof is not None:
    u.startstop_prof(prof)

  # Last note needs to happen before the pool's closed =)
  if not error:
    write_note(f"Done!\n{chrono.note}")
  else:
    write_note(f"Failed!\n{error}\n{chrono.note}")

  pool.close()
  pool.join()
  mw.close()

  # Make sure all hosts stay up until the end of main.
  u.sync()

  # Before cleanup, as cleanup should only run for successful jobs.
  if error is not None:
    raise RuntimeError(error)

  u.maybe_cleanup_workdir(workdir, flags.FLAGS.cleanup, info)


if __name__ == "__main__":
  app.run(main)
