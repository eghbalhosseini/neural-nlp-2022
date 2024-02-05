from neural_nlp.benchmarks import benchmark_pool
import os
import pandas as pd
import getpass
from neural_nlp.benchmarks.ceiling import (ExtrapolationCeiling,
                                           HoldoutSubjectCeiling,
                                           v,ci_error,manual_merge,_coords_match,
                                           FewSubjectExtrapolation,
                                           CeilingCrossValidation)
import sklearn
from brainscore.metrics.regression import linear_regression, pearsonr_correlation, CrossRegressedCorrelation
# find the  the path of RESULT_CACHING directory as a python variable

variable_value = os.environ.get('RESULTCACHING_HOME')
print(variable_value)


import os

# Set environment variable at the beginning of your script
os.environ["SPLIT_AT_PASSAGE"] = '1'
print(os.getenv('SPLIT_AT_PASSAGE'))

if __name__ =='__main__':

    benchmark_name = "Pereira2023aud-pass-passage-Encoding"
    benchmark=benchmark_pool[benchmark_name]
    # load _target_assembly
    # save_target_assembly as a netcdf file
    # drop attributes for voxel selection
    bench_metric=benchmark._metric
    # number of subsamples is how mant of combination (n choose k) we want to sample. for large number of subjects this can be very large
    bench_regression=bench_metric.regression
    bench_correlation=bench_metric.correlation
    bench_cross_val=bench_metric.cross_validation
    split_coord=bench_cross_val._split_coord
    stratification_coord=bench_cross_val._stratification_coord
    num_split=bench_cross_val._split._split.n_splits
    spliter=bench_cross_val._split._split
    print(f'split coordinate : {split_coord} \n')
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


    benchmark._ceiler=FewSubjectExtrapolation(subject_column='subject',extrapolation_dimension='neuroid',post_process=None,num_subsamples=150,num_bootstraps=200)
    ceiling=benchmark._ceiler(benchmark.identifier,assembly=benchmark._target_assembly,metric=bench_metric)
    # print ceiling
    print(ceiling)