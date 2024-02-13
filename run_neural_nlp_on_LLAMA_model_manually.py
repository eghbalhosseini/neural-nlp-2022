from neural_nlp.models.implementations import _PytorchTransformerWrapper, word_last, transformer_configurations,model_pool
from neural_nlp import score as score_function
import getpass

user=getpass.getuser()
if user=='eghbalhosseini':
    model_and_config_dir = '/Users/eghbalhosseini/MyData/neural_nlp_bench/'
    LLAMA_path = '/Users/eghbalhosseini/MyData/LLAMA/'
elif user=='ehoseini':
    model_and_config_dir = '/om2/user/ehoseini/MyData/neural_nlp_bench/'
    LLAMA_path= '/nese/mit/group/evlab/u/ehoseini/MyData/LLAMA/'

from accelerate import init_empty_weights, Accelerator
from accelerate import load_checkpoint_and_dispatch,infer_auto_device_map
from transformers import LlamaForCausalLM, LlamaTokenizer,LlamaConfig

import torch
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print (x)
else:
    print ("MPS device not found.")


if __name__ =='__main__':
    benchmark_name = "ANNSet1fMRI-encoding"

    modelname = '13B'

    weight_path = f'{LLAMA_path}/LLAMA_{modelname}/'
    config_path = f'{weight_path}/config.json'
    tokenizer = LlamaTokenizer.from_pretrained(weight_path)
    tokenizer.pad_token = tokenizer.eos_token
    modelConfig = LlamaConfig.from_json_file(config_path)
    modelConfig.output_hidden_states = True
    with init_empty_weights():
        model = LlamaForCausalLM(modelConfig)


    device_map = infer_auto_device_map(model, no_split_module_classes=['LlamaDecoderLayer'],)
    # print device map
    print(device_map)
    model = load_checkpoint_and_dispatch(model, checkpoint=weight_path, device_map=device_map)
    layers=('drop',) + tuple(f'encoder.layers.{i}' for i in range(modelConfig.num_hidden_layers))
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

    # score_results=score_function(benchmark=benchmark_tsk, model=model_identifier, model_impl=transformer,
    #                  layers=list(brainscore_config['layers']))
    score_results = score_function(benchmark=benchmark_name, model=model_identifier, model_impl=transformer,
                                   layers=list(layers))

    print(score_results)