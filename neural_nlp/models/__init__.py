import logging
import sys
from collections import OrderedDict
import torch
import copy
from tqdm import tqdm
import argparse
import numpy as np
from brainio.assemblies import merge_data_arrays
from numpy.random.mtrand import RandomState
from neural_nlp.models.implementations import model_pool, load_model, model_layers
def permute_mat(mat):
    mat_flat = mat.flatten()
    assert(mat_flat.ndim==1)
    shuffle_idx = torch.randperm(mat_flat.shape[0])
    mat_flat_rnd = mat_flat[shuffle_idx]
    mat_perm = torch.reshape(mat_flat_rnd, mat.shape)
    return mat_perm


def initialize_gpt2_weights(model, mu=0.0, sigma=0.02, permute=False, valid_keys=None):
    model_perm = copy.deepcopy(model)
    orig_states = model_perm.state_dict()
    if valid_keys is None:
        valid_keys = ['attn.c_attn.weight', 'attn.c_attn.bias', 'attn.c_proj', 'ln_1', 'ln_2', 'mlp.c_fc', 'mlp.c_proj',
                      'wte', 'wpe', 'lm_head']
    if type(mu) is float:
        # make a dictionalry of mu and sigma for each key
        mu_dict = dict.fromkeys(valid_keys, mu)

    elif type(mu) is dict:
        # add the missing keys to the mu_dict
        remaining_keys = [x for x in valid_keys if x not in mu.keys()]
        mu_dict = {**mu, **dict.fromkeys(remaining_keys, 0.0)}
    else:
        raise ValueError('mu must be either float or dict')
    if type(sigma) is float:
        # make a dictionalry of mu and sigma for each key
        sigma_dict = dict.fromkeys(valid_keys, sigma)
    elif type(sigma) is dict:
        # add the missing keys to the mu_dict
        remaining_keys = [x for x in valid_keys if x not in sigma.keys()]
        sigma_dict = {**sigma, **dict.fromkeys(remaining_keys, 0.0)}
    else:
        raise ValueError('sigma must be either float or dict')

    to_permute = np.sum(
        [np.sum([valid_keys[n] in s for s in list(orig_states.keys())]) for n in range(len(valid_keys))])
    if permute:
        pbar = tqdm(total=to_permute, desc=f'permuting {to_permute} weights in {len(orig_states.keys())}')
    else:
        pbar = tqdm(total=to_permute, desc=f'initializing {to_permute} weights in {len(orig_states.keys())}')
    perm_states = dict.fromkeys(orig_states.keys())
    for key in orig_states.keys():
        if any([x in key for x in valid_keys]):
            a = orig_states[key]
            idx = [x in key for x in valid_keys].index(True)
            mu_key = valid_keys[idx]
            b = torch.normal(mu_dict[mu_key], sigma_dict[mu_key], size=a.shape)
            perm_states[key] = permute_mat(a) if permute else permute_mat(b)
            pbar.update()
        else:
            perm_states[key] = orig_states[key]

    return perm_states


def initialize_layer_norm_uniform(model, valid_keys=None):
    model_perm = copy.deepcopy(model)
    orig_states = model_perm.state_dict()
    if valid_keys is None:
        valid_keys = ['ln_1', 'ln_2']
    to_permute = np.sum(
        [np.sum([valid_keys[n] in s for s in list(orig_states.keys())]) for n in range(len(valid_keys))])
    pbar = tqdm(total=to_permute, desc=f'initializing {to_permute} weights in {len(orig_states.keys())}')
    perm_states = dict.fromkeys(orig_states.keys())
    for key in orig_states.keys():
        if any([x in key for x in valid_keys]):
            a = orig_states[key]
            b = torch.rand(a.shape, requires_grad=False)
            perm_states[key] = permute_mat(b)
            pbar.update()
        else:
            perm_states[key] = orig_states[key]

    return perm_states


_logger = logging.getLogger(__name__)


class SubsamplingHook:
    def __init__(self, activations_extractor, num_features):
        self._activations_extractor = activations_extractor
        self._num_features = num_features
        self._sampling_indices = None

    def __call__(self, activations):
        self._ensure_initialized(activations)
        activations = OrderedDict((layer, layer_activations[:, self._sampling_indices[layer]])
                                  for layer, layer_activations in activations.items())
        return activations

    @classmethod
    def hook(cls, activations_extractor, num_features):
        hook = SubsamplingHook(activations_extractor=activations_extractor, num_features=num_features)
        handle = activations_extractor.register_activations_hook(hook)
        hook.handle = handle
        if hasattr(activations_extractor, 'identifier'):
            activations_extractor.identifier += f'-subsample_{num_features}'
        else:
            activations_extractor._extractor.identifier += f'-subsample_{num_features}'
        return handle

    def _ensure_initialized(self, activations):
        if self._sampling_indices:
            return
        rng = RandomState(0)
        self._sampling_indices = {layer: rng.randint(layer_activations.shape[1], size=self._num_features)
                                  for layer, layer_activations in activations.items()}


def get_activations(model_identifier, layers, stimuli, stimuli_identifier=None, subsample=None):
    _logger.debug(f'Loading model {model_identifier}')
    model = load_model(model_identifier)
    if subsample:
        SubsamplingHook.hook(model, subsample)

    _logger.debug("Retrieving activations")
    activations = model(stimuli, layers=layers, stimuli_identifier=stimuli_identifier)
    return activations


def get_activations_for_sentence(model_name, layers, sentences):
    model = load_model(model_name)
    activations = []
    for sentence in sentences:
        sentence_activations = model.get_activations([sentence], layers=layers)
        activations.append(sentence_activations)
    activations = merge_data_arrays(activations)
    return activations


def main():
    parser = argparse.ArgumentParser('model activations')
    parser.add_argument('--model', type=str, required=True, choices=list(model_pool.keys()))
    parser.add_argument('--layers', type=str, nargs='+', default='default')
    parser.add_argument('--sentence', type=str, required=True)
    parser.add_argument('--log_level', type=str, default='INFO')
    args = parser.parse_args()
    log_level = logging.getLevelName(args.log_level)
    logging.basicConfig(stream=sys.stdout, level=log_level)
    if args.layers == 'default':
        args.layers = model_layers[args.model]
    _logger.info("Running with args %s", vars(args))

    activations = get_activations_for_sentence(args.model, layers=args.layers, sentences=args.sentence.split(' '))
    _logger.info("Activations computed: {}".format(activations))


if __name__ == '__main__':
    main()
