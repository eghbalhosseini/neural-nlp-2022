
from neural_nlp.benchmarks import benchmark_pool
import os
import getpass
from neural_nlp.benchmarks.ceiling import (ExtrapolationCeiling,
                                           HoldoutSubjectCeiling,
                                           v,ci_error,manual_merge,_coords_match,
                                           FewSubjectExtrapolation)

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
    # number of subsamples is how mant of combination (n choose k) we want to sample. for large number of subjects this can be very large
    benchmark._ceiler=FewSubjectExtrapolation(subject_column='subject',extrapolation_dimension='neuroid',post_process=None,num_subsamples=num_subsamples,num_bootstraps=num_bootstraps)
    ceiling=benchmark._ceiler(benchmark.identifier,assembly=benchmark._target_assembly,metric=benchmark._metric)
    # print ceiling

