import argparse
from neural_nlp.models.implementations import _PytorchTransformerWrapper, word_last, transformer_configurations,model_pool
from neural_nlp import score as score_function
import os
import numpy as np
import getpass

from transformers import GPT2Tokenizer
from transformers import GPT2Model
from transformers import GPT2Config
from transformers import AutoConfig, AutoModel, AutoTokenizer

user=getpass.getuser()
if user=='eghbalhosseini':
    model_and_config_dir = '/Users/eghbalhosseini/MyData/neural_nlp_bench/'
elif user=='ehoseini':
    model_and_config_dir = '/om2/user/ehoseini/MyData/neural_nlp_bench/'

#GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP = {"distilgpt2": os.path.join(model_and_config_dir,'CONFIG_ARCHIVE_MAP','distilgpt2-config.json')}
#GPT2_PRETRAINED_MODEL_ARCHIVE_MAP = {"distilgpt2": os.path.join(model_and_config_dir,'MODEL_ARCHIVE_MAP','distilgpt2-pytorch_model.bin')}

if __name__ =='__main__':
    #benchmark_name="Pereira2018-encoding"
    #benchmark_name="Pereira2023aud-sent-sentence-RidgeEncoding"
    #benchmark_name="Pereira2018-RDM"
    #benchmark_name="DsParametricRDM-max-encoding"
    #benchmark_name="Pereira2018-encoding"
    #benchmark_name="LangLocECoG-sentence-encoding"
    #benchmark_name = 'wikitext-2'
    #benchmark_name = "Blank2014fROI-encoding"
    benchmark_name = "LangLocECoG-bip-gamma-RidgeEncoding"
    #benchmark_name ="Fedorenko2016v3-encoding"
    #benchmark_name = 'LangLocECoGv2-encoding'
    #model_name="gpt2"
    model_name="roberta-base"
    #config_file=GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP[model_name]
    #model_file=GPT2_PRETRAINED_MODEL_ARCHIVE_MAP[model_name]
    benchmark_tsk = benchmark_name
    all_model_names=[x['weight_identifier'] for x in transformer_configurations]
    all_model_names.index(model_name)
    model_configs=transformer_configurations[all_model_names.index(model_name)]
    model_identifier=model_name
    my_env = os.environ.copy()
    config = AutoConfig.from_pretrained(model_configs['weight_identifier'])
    config.output_hidden_states = True
    state_dict = None

    model = AutoModel.from_pretrained(model_configs['weight_identifier'], config=config, state_dict=state_dict)
    #model.from_pretrained(model_file, config=config)
    tokenizer = AutoTokenizer.from_pretrained(model_configs['weight_identifier'])
    #model_identifier = config.weight_identifier
    model_identifier=model_name
    # find model index in model_configs
#    config_idx = int(np.argwhere([x['weight_identifier'] == model_identifier for x in transformer_configurations]))
#    brainscore_config = transformer_configurations[config_idx]
    # - tokenizer_ctr: the importable class name of the model's tokenizer class
#    brainscore_config['tokenizer_ctr'] = brainscore_config.get('tokenizer_ctr',brainscore_config['prefix'] + 'Tokenizer')
    # - model_ctr: the importable class name of the model's model class
#    brainscore_config['model_ctr'] = brainscore_config.get('model_ctr', brainscore_config['prefix'] + 'Model')
    # - config_ctr: the importable class name of the model's config class
#    brainscore_config['config_ctr'] = brainscore_config.get('config_ctr', brainscore_config['prefix'] + 'Config')

    transformer = _PytorchTransformerWrapper(identifier=model_identifier,
                                             tokenizer=tokenizer,
                                             tokenizer_special_tokens=model_configs.get('tokenizer_special_tokens',()),
                                             model=model,
                                             layers=model_configs['layers'],
                                             sentence_average=word_last)

    #score_results=score_function(benchmark=benchmark_tsk, model=model_identifier, model_impl=transformer,
    #                  layers=list(brainscore_config['layers']))
    benchmark=benchmark_tsk
    model_impl=transformer
    layers=list(model_configs['layers'])
    score_results=score_function(benchmark=benchmark_tsk, model=model_identifier, model_impl=transformer,
                   layers=list(model_configs['layers']))

    print(score_results)