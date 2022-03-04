import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import argparse
from pathlib import Path
import os
from collections import namedtuple
import pandas as pd
import xarray as xr
import glob as glob
import re
from neural_nlp.models import model_pool, model_layers

def get_args():
    parser = argparse.ArgumentParser(description='model_name')
    parser.add_argument('model_prefix', type=str,default='gpt2-neox-orig')
    parser.add_argument('benchmark', type=str, default='Pereira2018-encoding')
    args = parser.parse_args()
    return args

def mock_get_args():
    mock_args = namedtuple('debug', ['model_prefix', 'benchmark'])
    new_args = mock_args('gpt2-neox-pos_learned', 'Pereira2018-encoding')
    return new_args

debug = True
score_dir='/om5/group/evlab/u/ehoseini/.result_caching/neural_nlp.score/'
save_dir='/om2/user/ehoseini/MyData/neural_nlp_2020/score_plots/'

if __name__ == '__main__':
    if debug:
        args = mock_get_args()
    else:
        args = get_args()

    file_pattern=f"benchmark={args.benchmark},model={args.model_prefix}*"
    print(f"finding {file_pattern} in {score_dir}")
    model_scores=glob.glob(os.path.join(score_dir,file_pattern))
    re_p=re.compile(f"pos_learned-\d+\w+")
    untrained=['untrained' in x for x in model_scores]
    temp = [re.findall(re_p, x) for x in model_scores]
    mdl_d_sz=[x[0].split('-')[-1] for x in temp]


    #
    model_scores=[model_scores[idx] for idx, x in enumerate(untrained) if not x]
    ckpt_ids=[mdl_d_sz[idx] for idx, x in enumerate(untrained) if not x]

    ax_cmap = cm.get_cmap('inferno', len(ckpt_ids) + 3)
    ax_colors = (ax_cmap(np.arange(len(ckpt_ids)) / (len(ckpt_ids) + 1)))
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_axes((.2, .3, .6, .5))
    for idx,fname in enumerate(model_scores):
        score_dat = pd.read_pickle(fname)
        score_xr = score_dat['data']
        #layers = model_layers[args.model_name]
        r3 = np.arange(len(score_xr))
        y=score_xr.values[:,0]
        ax.plot(r3, y, linewidth=3, color=ax_colors[idx, :], label=f'size {ckpt_ids[idx]},trained={not(untrained[idx])}', marker='.', markersize=10)
        #ax.fill_between(r3, y - yerr, y + yerr, facecolor=all_col[1, :], alpha=0.1)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xticks(np.arange(score_xr.shape[0]))
    ax.set_xticklabels(score_xr['layer'].values, fontsize=6, fontweight='normal', rotation=90)
    ax.set_ylabel('score')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #ax.set_title(f"model= {args.model_name}\nbencmarks={args.benchmark}", fontsize=10, fontweight='bold')
    save_fname = Path(save_dir, file_pattern.replace('*', '_result.pdf'))
    save_fname.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_fname), transparent=False)
    fig.show()