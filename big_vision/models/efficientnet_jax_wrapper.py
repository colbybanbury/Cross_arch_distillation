from jeffnet.linen import create_model
from big_vision import utils
from big_vision.models import common
from  jeffnet.common import load_state_dict_from_url, split_state_dict, load_state_dict
from flax.core import FrozenDict, freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict
import re

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

def _filter(state_dict):
    """ convert state dict keys from pytorch style origins to flax linen """
    out = {}
    p_blocks = re.compile(r'blocks\.(\d)\.(\d)')
    p_bn_scale = re.compile(r'bn(\w*)\.weight')
    for k, v in state_dict.items():
        k = p_blocks.sub(r'blocks_\1_\2', k)
        k = p_bn_scale.sub(r'bn\1.scale', k)
        k = k.replace('running_mean', 'mean')
        k = k.replace('running_var', 'var')
        k = k.replace('.weight', '.kernel')
        out[k] = v
    return out
    
def load_url(variables, url, dont_load=None, filter_fn=_filter):
    state_dict = load_state_dict_from_url(url, transpose=True)

    source_params, source_state = split_state_dict(state_dict)
    if filter_fn is not None:
        # filter after split as we may have modified the split criteria (ie bn running vars)
        source_params = filter_fn(source_params)
        source_state = filter_fn(source_state)

    # FIXME better way to do this?
    var_unfrozen = unfreeze(variables)
    missing_keys = []
    flat_params = flatten_dict(var_unfrozen['params'])
    flat_param_keys = set()
    for k, v in flat_params.items():
        #dontload can be a list of keys or a prefixes (e.g. 'head')
        if '/'.join(k) in dont_load or dont_load[0] in '/'.join(k):
            continue
        flat_k = '.'.join(k)
        if flat_k in source_params:
            assert flat_params[k].shape == v.shape
            flat_params[k] = source_params[flat_k]
        else:
            missing_keys.append(flat_k)
        flat_param_keys.add(flat_k)
    unexpected_keys = list(set(source_params.keys()).difference(flat_param_keys))
    params = unflatten_dict(flat_params)

    flat_state = flatten_dict(var_unfrozen['batch_stats'])
    flat_state_keys = set()
    for k, v in flat_state.items():
        if '/'.join(k) in dont_load:
            continue
        flat_k = '.'.join(k)
        if flat_k in source_state:
            assert flat_state[k].shape == v.shape
            flat_state[k] = source_state[flat_k]
        else:
            missing_keys.append(flat_k)
        flat_state_keys.add(flat_k)
    unexpected_keys.extend(list(set(source_state.keys()).difference(flat_state_keys)))
    batch_stats = unflatten_dict(flat_state)

    if missing_keys:
        print(f' WARNING: {len(missing_keys)} keys missing while loading state_dict. {str(missing_keys)}')
    if unexpected_keys:
        print(f' WARNING: {len(unexpected_keys)} unexpected keys found while loading state_dict. {str(unexpected_keys)}')

    return dict(params=params, batch_stats=batch_stats)

def load(init_params, init_file, model_cfg, dont_load=(), filter=_filter):
    """Load init from checkpoint."""
    del model_cfg  # Unused

    if "http" in init_file:
        params = load_url(init_params, init_file, dont_load=dont_load, filter_fn=filter)
    else:
        params = utils.load_params(None, init_file)
        params = common.merge_params(params, init_params, dont_load)
    return params
