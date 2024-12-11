from neural_nlp.models.implementations import _PytorchTransformerWrapper, word_last
from neural_nlp import score as score_function
import getpass

user=getpass.getuser()
if user=='eghbalhosseini':
    model_and_config_dir = '/Users/eghbalhosseini/MyData/neural_nlp_bench/'
    LLAMA_path = '/Users/eghbalhosseini/MyData/LLAMA/'
elif user=='ehoseini':
    model_and_config_dir = '/om2/user/ehoseini/MyData/neural_nlp_bench/'
    LLAMA_path= '/rdma/vast-rdma/vast/evlab/ehoseini/MyData/LLAMA/'

from accelerate import init_empty_weights, Accelerator
from accelerate import load_checkpoint_and_dispatch,infer_auto_device_map
from transformers import LlamaForCausalLM, LlamaTokenizer,LlamaConfig

import torch
if torch.cuda.is_available():
    print(f"PyTorch is using {torch.cuda.device_count()} GPU(s):")
    print(f" - GPU Name: {torch.cuda.get_device_name(0)}")
    print(f" - CUDA Version: {torch.version.cuda}")
else:
    print("No GPU found, PyTorch is using CPU.")

def get_gpu_memory_size(gpu_name):
    if "A100" in gpu_name:
        return "70GiB"
    elif "RTX A6000" in gpu_name or "A6000" in gpu_name:
        return "40GiB"
    else:
        return "Unknown Size"

max_memory_declaration = {}
import torch
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
else:
    print ("MPS device not found.")
    num_devices = torch.cuda.device_count()
    # Iterate through each CUDA device
    for i in range(num_devices):
        gpu_name = torch.cuda.get_device_name(i)
        max_memory_declaration[i] = get_gpu_memory_size(gpu_name)

    # get the number of available gpus
    device_count = torch.cuda.device_count()
    device_type = torch.cuda.get_device_name(0)


if __name__ =='__main__':
    benchmark_name = "DsParametricfMRI-first-all-rand-Encoding_sep2024"
    #benchmark_name='ANNSet1fMRI-best-reliable-encoding'
    modelname = '65B'

    weight_path = f'{LLAMA_path}/LLAMA_{modelname}/'
    config_path = f'{weight_path}/config.json'
    tokenizer = LlamaTokenizer.from_pretrained(weight_path)
    tokenizer.pad_token = tokenizer.eos_token
    modelConfig = LlamaConfig.from_json_file(config_path)
    modelConfig.output_hidden_states = True
    with init_empty_weights():
        model = LlamaForCausalLM(modelConfig)

    device_map = infer_auto_device_map(model, no_split_module_classes=['LlamaDecoderLayer'],
                                       max_memory=max_memory_declaration)
    # print device map
    print(device_map)
    model = load_checkpoint_and_dispatch(model, checkpoint=weight_path, device_map=device_map)
    layers=('drop',) + tuple(f'encoder.layers.{i}' for i in range(modelConfig.num_hidden_layers))
    for i in model.named_parameters():
        print(f"{i[0]} -> {i[1].device}")
    # config_file=GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP[model_name]
    # model_file=GPT2_PRETRAINED_MODEL_ARCHIVE_MAP[model_name]
    benchmark_tsk = benchmark_name

    state_dict = None
    tokenizer.special_tokens_map
    model_identifier = modelConfig.model_type+'_'+modelname
    transformer = _PytorchTransformerWrapper(identifier=model_identifier,
                                             tokenizer=tokenizer,
                                             tokenizer_special_tokens=(),
                                             model=model,
                                             layers=layers,
                                             sentence_average=word_last)


    score_results = score_function(benchmark=benchmark_name, model=model_identifier, model_impl=transformer,
                                   layers=list(layers))

    print(score_results)
