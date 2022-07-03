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

def permute_mat(mat):
    mat_flat = mat.flatten()
    assert(mat_flat.ndim==1)
    shuffle_idx = torch.randperm(mat_flat.shape[0])
    mat_flat_rnd = mat_flat[shuffle_idx]
    mat_perm = torch.reshape(mat_flat_rnd, mat.shape)
    return mat_perm

def initialize_gpt_neox_weights(model,mu=0,sigma=0.02):
    model_perm = copy.deepcopy(model)
    orig_states = model_perm.state_dict()
    perm_states = dict.fromkeys(orig_states.keys())
    for key in tqdm(orig_states.keys(), desc='permuting weights'):
        if 'attn.qkv_proj.weight' in key:
            a = orig_states[key]
            b = torch.normal(mu, sigma, size=a.shape)
            perm_states[key] = b
        elif 'attn.qkv_proj.bias' in key:
            a = orig_states[key]
            b = torch.normal(mu, sigma, size=a.shape)
            perm_states[key] = permute_mat(b)
        elif 'attn.out_proj' in key:
            a = orig_states[key]
            b = torch.normal(mu, sigma, size=a.shape)
            perm_states[key] = permute_mat(b)
        # modify layer norm
        elif 'ln' in key:
            a = orig_states[key]
            b = torch.normal(mu, sigma, size=a.shape)
            perm_states[key] = permute_mat(b)
        # modify feedforward layer
        elif 'mlp.fc' in key:
            a = orig_states[key]
            b = torch.normal(mu, sigma, size=a.shape)
            perm_states[key] = permute_mat(b)
        elif 'wte' in key or 'wpe' in key:
            a = orig_states[key]
            b = torch.normal(mu, sigma, size=a.shape)
            perm_states[key] = permute_mat(b)
        else:
            perm_states[key] = orig_states[key]
    return perm_states