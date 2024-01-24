
from neural_nlp.benchmarks import benchmark_pool
import os
import getpass
from neural_nlp.benchmarks.ceiling import (ExtrapolationCeiling,
                                           HoldoutSubjectCeiling,
                                           v,ci_error,manual_merge,_coords_match,
                                           FewSubjectExtrapolation)


if __name__ =='__main__':

    benchmark_name = "LangLocECoG-uni-gaus-Encoding"

    benchmark=benchmark_pool[benchmark_name]
    bench_metric=benchmark._metric
    bench_metric._show_tqdm=False
    # number of subsamples is how mant of combination (n choose k) we want to sample. for large number of subjects this can be very large
    benchmark._ceiler=FewSubjectExtrapolation(subject_column='subject',extrapolation_dimension='neuroid',post_process=None,num_subsamples=10,num_bootstraps=10)
    ceiling=benchmark._ceiler(benchmark.identifier,assembly=benchmark._target_assembly,metric=bench_metric)
    # print ceiling

