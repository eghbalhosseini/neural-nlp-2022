#from neural_nlp.models.gpt_neox_model.modeling_gpt_neox import GPTNeoXForCausalLM, GPTNeoXModel
#from neural_nlp.models.gpt_neox_model.configuration_gpt_neox import GPTNeoXConfig
from transformers import GPT2Tokenizer
#from transformers import GPT2BPETokenizer
from neural_nlp.models.gpt_neox_model.modeling_gpt_neox import  GPTNeoXModel, GPTNeoXForCausalLM
from neural_nlp.models.gpt_neox_model.configuration_gpt_neox import GPTNeoXConfig
from neural_nlp.models.gpt_neox_model.configuration_gpt_neox import GPTNeoXPosLearnedConfig
from neural_nlp.models.gpt_neox_model.modeling_gpt_neox_learned import GPTNeoXPosLearnedForCausalLM,GPTNeoXPosLearnedModel
import torch
import copy
from tqdm import tqdm
import numpy as np

def permute_mat(mat):
    mat_flat = mat.flatten()
    assert(mat_flat.ndim==1)
    shuffle_idx = torch.randperm(mat_flat.shape[0])
    mat_flat_rnd = mat_flat[shuffle_idx]
    mat_perm = torch.reshape(mat_flat_rnd, mat.shape)
    return mat_perm

def initialize_gpt_neox_weights(model,mu=0,sigma=0.02,permute=False):
    model_perm = copy.deepcopy(model)
    orig_states = model_perm.state_dict()
    valid_keys = ['attn.qkv_proj.weight', 'attn.qkv_proj.bias', 'attn.out_proj', 'ln', 'mlp.fc', 'wte', 'wpe']
    to_permute = np.sum(
        [np.sum([valid_keys[n] in s for s in list(orig_states.keys())]) for n in range(len(valid_keys))])
    if permute:
        pbar = tqdm(total=to_permute, desc=f'permuting {to_permute} weights in {len(orig_states.keys())}')
    else:
        pbar = tqdm(total=to_permute, desc=f'initializing {to_permute} weights in {len(orig_states.keys())}')

    perm_states = dict.fromkeys(orig_states.keys())
    for key in orig_states.keys():
        if 'attn.qkv_proj.weight' in key:
            a = orig_states[key]
            b = torch.normal(mu, sigma, size=a.shape)
            #out_st= permute_mat(a) if permute else permute_mat(b)
            perm_states[key] = permute_mat(a) if permute else permute_mat(b)
            pbar.update()
        elif 'attn.qkv_proj.bias' in key:
            a = orig_states[key]
            b = torch.normal(mu, sigma, size=a.shape)
            perm_states[key] = permute_mat(a) if permute else permute_mat(b)
            pbar.update()
        elif 'attn.out_proj' in key:
            a = orig_states[key]
            b = torch.normal(mu, sigma, size=a.shape)
            perm_states[key] = permute_mat(a) if permute else permute_mat(b)
            pbar.update()
        # modify layer norm
        elif 'ln' in key:
            a = orig_states[key]
            b = torch.normal(mu, sigma, size=a.shape)
            perm_states[key] = permute_mat(a) if permute else permute_mat(b)
            pbar.update()
        # modify feedforward layer
        elif 'mlp.fc' in key:
            a = orig_states[key]
            b = torch.normal(mu, sigma, size=a.shape)
            perm_states[key] = permute_mat(a) if permute else permute_mat(b)
            pbar.update()
        elif 'wte' in key or 'wpe' in key:
            a = orig_states[key]
            b = torch.normal(mu, sigma, size=a.shape)
            perm_states[key] = permute_mat(a) if permute else permute_mat(b)
            pbar.update()
        else:
            perm_states[key] = orig_states[key]
    return perm_states