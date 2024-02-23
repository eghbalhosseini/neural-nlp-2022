
from neural_nlp.benchmarks import benchmark_pool
import os
import getpass
from neural_nlp.benchmarks.ceiling import (ExtrapolationCeiling,
                                           HoldoutSubjectCeiling,
                                           v,ci_error,manual_merge,_coords_match,
                                           FewSubjectExtrapolation,
                                           CeilingCrossValidation)
import sklearn

# get 3 arguments from command line
# first one is banchmark name,
# second one is number of subsamples
# third one is number of bootstraps
import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--benchmark', type=str, default='LangLocECoG-uni-gaus-Encoding',
                    help='benchmark')
parser.add_argument('--num_subsamples', type=int, default=10,
                    help='number of subsamples')
parser.add_argument('--num_bootstraps', type=int, default=10,
                    help='number of bootstraps')

if __name__ =='__main__':
    # get arguments
    args = parser.parse_args()
    benchmark_name=args.benchmark
    num_subsamples=args.num_subsamples
    num_bootstraps=args.num_bootstraps
    # pull the benchmark and compute the ceiling, it will be saved in result_caching folder
    benchmark=benchmark_pool[benchmark_name]
    bench_metric=benchmark._single_metric
    bench_regression = bench_metric.regression
    bench_correlation = bench_metric.correlation
    bench_cross_val = bench_metric.cross_validation
    split_coord = bench_cross_val._split_coord
    stratification_coord = bench_cross_val._stratification_coord
    num_split = bench_cross_val._split._split.n_splits
    spliter = bench_cross_val._split._split
    if type(spliter)==sklearn.model_selection._split.StratifiedKFold:
            kfold=True
    elif type(spliter)==sklearn.model_selection._split.KFold:
            kfold=True
    elif type(spliter)==sklearn.model_selection._split.StratifiedShuffleSplit:
            kfold=False
    elif type(spliter)==sklearn.model_selection._split.ShuffleSplit:
            kfold=False
    else:
        raise ValueError('Unknown split type')

    bench_metric.cross_validation=CeilingCrossValidation(split_coord=split_coord,
                                                         stratification_coord=stratification_coord,
                                                         splits=num_split,kfold=kfold,show_tqdm=False)

    # number of subsamples is how mant of combination (n choose k) we want to sample. for large number of subjects this can be very large
    benchmark._ceiler=FewSubjectExtrapolation(subject_column='subject',extrapolation_dimension='neuroid',post_process=None,num_subsamples=num_subsamples,num_bootstraps=num_bootstraps)
    ceiling=benchmark._ceiler(benchmark.identifier,assembly=benchmark._target_assembly,metric=bench_metric)
    # print ceiling

