from jeffnet.linen import create_model


def Model(num_classes, variant, **kwargs):
    """wrapper for efficientnet-jax model"""

    #create model based on variant and config.
    #We ignore the variables created by create_model since they are 
    #    initialized in the big_vision training functions
    model, _ = create_model(
        variant,
        # dtype=model_dtype,
        # drop_rate=drop_rate,
        # drop_path_rate=drop_path_rate, #Not using for now
        # rng=model_create_rng
        num_classes=num_classes, 
        **kwargs)

    return model

        
    
def load(init_params, init_file, model_cfg, dont_load=()):
  """Load init from checkpoint."""
  del model_cfg  # Unused
  params = utils.load_params(None, init_file)
  params = common.merge_params(params, init_params, dont_load)
  return params
