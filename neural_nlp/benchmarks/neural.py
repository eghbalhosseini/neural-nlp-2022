"""
Neural benchmarks to probe match of model internals against human internals.
"""
from scipy.optimize import curve_fit
import warnings
from collections import defaultdict
import itertools
import logging
import numpy as np
from brainio.assemblies import DataAssembly, walk_coords, merge_data_arrays, array_is_element
from numpy.random.mtrand import RandomState
from scipy.stats import median_abs_deviation as median_absolute_deviation
from tqdm import tqdm, trange
from neural_nlp.benchmarks.metric.rsa.metric import XarrayRSA
from brainscore.benchmarks import Benchmark
from brainscore.metrics import Score
from brainscore.metrics.rdm import RDM, RDMSimilarity, RDMCrossValidated
from brainscore.metrics.cka import CKACrossValidated
from brainscore.metrics.regression import linear_regression, pearsonr_correlation, CrossRegressedCorrelation,XarrayRegression,pls_regression
from brainscore.metrics.transformations import CartesianProduct, CrossValidation, apply_aggregate
from brainscore.utils import LazyLoad
from neural_nlp.benchmarks.ceiling import ExtrapolationCeiling, HoldoutSubjectCeiling, v,ci_error,manual_merge,_coords_match, FewSubjectExtrapolation
from neural_nlp.benchmarks.s3 import load_s3
from neural_nlp.neural_data.ecog import load_Fedorenko2016
from neural_nlp.neural_data.mgh_ecog import load_MghMockLang
from neural_nlp.neural_data.fmri import load_voxels, load_rdm_sentences, \
    load_Pereira2018_Blank, load_Pereira2018
from neural_nlp.stimuli import load_stimuli, StimulusSet
from neural_nlp.utils import ordered_set
from result_caching import store
import xarray as xr
import getpass
import pandas as pd
import pickle, pathlib
from pathlib import Path
from brainio.assemblies import NeuroidAssembly
from brainio.fetch import fullname
from sklearn.linear_model import RidgeCV
import scipy.stats as stats
import rsatoolbox.data as rsd
from neural_nlp.benchmarks.metric.rsa.metric import XarrayRSA, rsa_correlation
import os


#########################
### SETUP
#########################


if getpass.getuser() == 'eghbalhosseini':
    ANNfMRI_PARENT = '/Users/eghbalhosseini/MyData/neural_nlp_bench/dataset/'
    ANNECOG_PARENT = '/Users/eghbalhosseini/MyData/neural_nlp_bench/dataset/'
    PEREIRA2018_SAMPLE = '/Users/eghbalhosseini/.result_caching/.neural_nlp/'
    fMRI_PARENT = '/Users/eghbalhosseini/MyData/neural_nlp_bench/dataset/'

elif getpass.getuser() == 'ehoseini':
    ANNfMRI_PARENT = '/om2/user/ehoseini/MyData/brain-score-language/dataset/'
    ANNECOG_PARENT = '/om2/user/ehoseini/MyData/brain-score-language/dataset/'
    PEREIRA2018_SAMPLE='/net/storage001.ib.cluster/om2/group/evlab/u/ehoseini/.result_caching/.neural_nlp/'
    fMRI_PARENT = '/om/weka/evlab/ehoseini/MyData/fmri_DNN/outputs/'

elif getpass.getuser() == 'ckauf':
    fMRI_PARENT = '/om2/vast/evlab/ckauf/contextualization-nnlp/fmri_DNN/outputs'


_logger = logging.getLogger(__name__)



#########################
### BASIC FUNCTIONS
#########################

def rgcv_linear_regression(xarray_kwargs=None):
    regression = RidgeCV(
            alphas=[1e-3, 0.01, 0.1, 1, 10, 100])
    xarray_kwargs = xarray_kwargs or {}
    regression = XarrayRegression(regression, **xarray_kwargs)
    return regression


def rgcv_linear_pearsonr(*args, regression_kwargs=None, correlation_kwargs=None, **kwargs):
    regression = rgcv_linear_regression(regression_kwargs or {})
    correlation = pearsonr_correlation(correlation_kwargs or {})
    return CrossRegressedCorrelation(
            *args, regression=regression, correlation=correlation,
            **kwargs)


def align(source, target, on):
    source_values, target_values = source[on].values.tolist(), target[on].values
    indices = [source_values.index(value) for value in target_values]
    assert len(source[on].dims) == 1, "multi-dimensional coordinates not implemented"
    dim = source[on].dims[0]
    dim_indices = {_dim: slice(None) if _dim != dim else indices for _dim in source.dims}
    aligned = source.isel(**dim_indices)
    return aligned


def listen_to(candidate, stimulus_set, reset_column='story', average_sentence=True):
    """
    Pass a `stimulus_set` through a model `candidate`.
    Operates on a sentence-based `stimulus_set`.
    """
    activations = []
    for story in ordered_set(stimulus_set[reset_column].values):
        story_stimuli = stimulus_set[stimulus_set[reset_column] == story]
        story_stimuli.name = f"{stimulus_set.name}-{story}"
        story_activations = candidate(stimuli=story_stimuli, average_sentence=average_sentence)
        activations.append(story_activations)
    model_activations = merge_data_arrays(activations)
    #model_activations = xr.concat(activations, dim='presentation')
    # merging does not maintain stimulus order. the following orders again
    idx = [model_activations['stimulus_id'].values.tolist().index(stimulus_id) for stimulus_id in
           itertools.chain.from_iterable(s['stimulus_id'].values for s in activations)]
    assert len(set(idx)) == len(idx), "Found duplicate indices to order activations"
    model_activations = model_activations[{'presentation': idx}]
    return model_activations

    
def read_words(candidate, stimulus_set, reset_column='stimulus_id', copy_columns=(), average_sentence=False):
    """
    Pass a `stimulus_set` through a model `candidate`.
    In contrast to the `listen_to` function, this function operates on a word-based `stimulus_set`.
    """
    # Input: stimulus_set = pandas df, col 1 with sentence ID and 2nd col as word.
    activations = []
    for i, reset_id in enumerate(ordered_set(stimulus_set[reset_column].values)):
        part_stimuli = stimulus_set[stimulus_set[reset_column] == reset_id]
        # stimulus_ids = part_stimuli['stimulus_id']
        sentence_stimuli = StimulusSet({'sentence': ' '.join(part_stimuli['word']),
                                        reset_column: list(set(part_stimuli[reset_column]))})
        sentence_stimuli.name = f"{stimulus_set.name}-{reset_id}"
        sentence_activations = candidate(stimuli=sentence_stimuli, average_sentence=average_sentence)
        for column in copy_columns:
            sentence_activations[column] = ('presentation', part_stimuli[column])
        activations.append(sentence_activations)

    #model_activations = merge_data_arrays(activations)
    model_activations = xr.concat(activations, dim='presentation')
    # merging does not maintain stimulus order. the following orders again
    idx = [model_activations['stimulus_id'].values.tolist().index(stimulus_id) for stimulus_id in
           itertools.chain.from_iterable(s['stimulus_id'].values for s in activations)]
    assert len(set(idx)) == len(idx), "Found duplicate indices to order activations"
    model_activations = model_activations[{'presentation': idx}]

    return model_activations


def aggregate(score, combine_layers=True):
    if hasattr(score, 'experiment') and score['experiment'].ndim > 0:
        score = score.mean('experiment')
    if hasattr(score, 'atlas') and score['atlas'].ndim > 0:
        score = score.mean('atlas')
    if hasattr(score, 'layer') and score['layer'].ndim > 0 and combine_layers:
        score_core = score.sel(aggregation='center') if hasattr(score, 'aggregation') else score
        max_score = score_core.max()
        max_score = score[{'layer': (score_core == max_score).values}]
        if len(max_score['layer']) > 1:  # multiple layers achieved exactly the same score
            layer_index = max_score['layer'].values[0].tolist().index(max_score['layer'].values[0])  # choose first one
            max_score = max_score.isel(layer=[layer_index])
        max_score = max_score.squeeze('layer', drop=True)
        max_score.attrs['raw'] = score.copy()
        score = max_score
    return score


def ceil_neuroids(raw_neuroids, ceiling, subject_column='subject'):
    ceiled_neuroids = consistency_neuroids(raw_neuroids, ceiling.raw)
    ceiled_neuroids.attrs['raw'] = raw_neuroids
    ceiled_neuroids.attrs['ceiling'] = ceiling.raw
    score = aggregate_neuroid_scores(ceiled_neuroids, subject_column)
    score.attrs['ceiling'] = ceiling
    score.attrs['description'] = "per-neuroid ceiling-normalized score"
    return score


def aggregate_neuroid_scores(neuroid_scores, subject_column):
    subject_scores = neuroid_scores.groupby(subject_column).median()
    center = subject_scores.median(subject_column)
    subject_values = np.nan_to_num(subject_scores.values, nan=0)  # mad cannot deal with all-nan in one axis, treat as 0
    subject_axis = subject_scores.dims.index(subject_scores[subject_column].dims[0])
    error = median_absolute_deviation(subject_values, axis=subject_axis)
    score = Score([center, error], coords={'aggregation': ['center', 'error']}, dims=['aggregation'])
    score.attrs['raw'] = neuroid_scores
    score.attrs['description'] = "score aggregated by taking median of neuroids per subject, " \
                                 "then median of subject scores"
    return score


def consistency_neuroids(neuroids, ceiling_neuroids):
    if 'neuroid_id' in ceiling_neuroids.dims:
        assert set(neuroids['neuroid_id'].values) == set(ceiling_neuroids['neuroid_id'].values)
    elif 'neuroid' in ceiling_neuroids.dims:
        assert set(neuroids['neuroid'].values) == set(ceiling_neuroids['neuroid'].values)
    ceiling_neuroids = ceiling_neuroids[{'neuroid': [neuroids['neuroid_id'].values.tolist().index(neuroid_id)
                                                     for neuroid_id in neuroids['neuroid_id'].values]}]  # align
    ceiling_neuroids = ceiling_neuroids.sel(aggregation='center')
    values = consistency(neuroids.values, ceiling_neuroids.values)
    neuroids = type(neuroids)(values, coords={coord: (dims, values) for coord, dims, values in walk_coords(neuroids)},
                              dims=neuroids.dims)
    return neuroids


def aggregate_ceiling(neuroid_scores, ceiling, subject_column='subject'):
    aggregate_raw = aggregate_neuroid_scores(neuroid_scores, subject_column=subject_column)
    score = consistency(aggregate_raw, ceiling.sel(aggregation='center'))
    score.attrs['raw'] = aggregate_raw
    score.attrs['ceiling'] = ceiling
    score.attrs['description'] = "ceiling-normalized score"
    return score


def consistency(score, ceiling):
    return score / ceiling


#########################
### BENCHMARKS (NO AUD)
#########################


class _PereiraBenchmark(Benchmark):
    """
    data source:
        Pereira et al., nature communications 2018
        https://www.nature.com/articles/s41467-018-03068-4
    """

    def __init__(self, identifier, metric, data_version='base'):
        self._identifier = identifier
        self._data_version = data_version
        self._target_assembly = LazyLoad(lambda: self._load_assembly(version=self._data_version))
        self._metric = metric
        self._ceiler = self.PereiraExtrapolationCeiling(subject_column='subject', num_bootstraps=100)
        self._cross = CartesianProduct(dividers=['experiment', 'atlas'])

    @property
    def identifier(self):
        return self._identifier

    def _metric(self, source_assembly, target_assembly):
        """ for ceiling compute """
        cross_scores = self._cross(target_assembly, apply=
        lambda cross_assembly: self._apply_cross(source_assembly, cross_assembly))
        score = self._average_cross_scores(cross_scores)
        return score

    def _average_cross_scores(self, cross_scores):
        return cross_scores.mean(['experiment', 'atlas'])

    @load_s3(key='Pereira2018')
    def _load_assembly(self, version='base'):
        assembly = load_Pereira2018_Blank(version=version)
        assembly = assembly.sel(atlas_selection_lower=90)
        assembly = assembly[{'neuroid': [filter_strategy in [np.nan, 'HminusE', 'FIXminusH']
                                         for filter_strategy in assembly['filter_strategy'].values]}]
        return assembly

    def __call__(self, candidate):
        stimulus_set = self._target_assembly.attrs['stimulus_set']
        model_activations = listen_to(candidate, stimulus_set)
        assert set(model_activations['stimulus_id'].values) == set(self._target_assembly['stimulus_id'].values)
        _logger.info('Scoring across experiments & atlases')
        cross_scores = self._cross(self._target_assembly,
                                   apply=lambda cross_assembly: self._apply_cross(model_activations, cross_assembly))
        raw_scores = cross_scores.raw
        raw_neuroids = apply_aggregate(lambda values: values.mean('split').mean('experiment'), raw_scores)

        # normally we would ceil every single neuroid here. To estimate the strongest ceiling possible (i.e. make it as
        # hard as possible on the models), we used experiment-overlapping neuroids from as many subjects as possible
        # which means some neuroids got excluded. Since median(r/c) is the same as median(r)/median(c), we just
        # normalize the neuroid aggregate by the overall ceiling aggregate.
        # Additionally, the Pereira data also has voxels from DMN, visual etc. but we care about language here.
        language_neuroids = raw_neuroids.sel(atlas='language', _apply_raw=False)
        score = aggregate_ceiling(language_neuroids, ceiling=self.ceiling, subject_column='subject')
        return score

    def _apply_cross(self, source_assembly, cross_assembly):
        cross_assembly = cross_assembly.dropna('neuroid')  # some subjects have only done one experiment
        source_assembly = source_assembly.dropna('neuroid')  # only relevant when running audio-visual self as "model"
        #assert len(cross_assembly['presentation']) in [243, 384]
        assert not np.isnan(cross_assembly).any()
        source_assembly = source_assembly[{'presentation': [stimulus_id in cross_assembly['stimulus_id'].values
                                                            for stimulus_id in source_assembly['stimulus_id'].values]}]
        return self._metric(source_assembly, cross_assembly)

    @property
    def ceiling(self):
        return self._ceiler(identifier=self.identifier, assembly=self._target_assembly, metric=self._metric)

    class PereiraExtrapolationCeiling(ExtrapolationCeiling):
        def __init__(self, subject_column, *args, **kwargs):
            super(_PereiraBenchmark.PereiraExtrapolationCeiling, self).__init__(
                subject_column, *args, **kwargs)
            self._num_subsamples = 10
            self.holdout_ceiling = _PereiraBenchmark.PereiraHoldoutSubjectCeiling(subject_column=subject_column)
            self._rng = RandomState(0)

        def iterate_subsets(self, assembly, num_subjects):
            # cross experiment to obtain more subjects to extrapolate.
            # don't worry about atlases here, cross-metric will take care of it.
            experiments = set(assembly['experiment'].values)
            for experiment in sorted(experiments):
                experiment_assembly = assembly[{'presentation': [
                    experiment_value == experiment for experiment_value in assembly['experiment'].values]}]
                experiment_assembly = experiment_assembly.dropna('neuroid')  # drop subjects that haven't done this exp
                if len(set(experiment_assembly[self.subject_column].values)) < num_subjects:
                    continue  # not enough subjects
                for sub_subjects in self._random_combinations(
                        subjects=set(experiment_assembly[self.subject_column].values),
                        num_subjects=num_subjects, choice=self._num_subsamples, rng=self._rng):
                    sub_assembly = assembly[{'neuroid': [subject in sub_subjects
                                                         for subject in assembly[self.subject_column].values]}]
                    yield {self.subject_column: sub_subjects, 'experiment': experiment}, sub_assembly

        def _random_combinations(self, subjects, num_subjects, choice, rng):
            # following https://stackoverflow.com/a/55929159/2225200. Also see similar method in `behavioral.py`.
            subjects = np.array(list(subjects))
            combinations = set()
            while len(combinations) < choice:
                elements = rng.choice(subjects, size=num_subjects, replace=False)
                combinations.add(tuple(elements))
            return combinations

        def extrapolate(self, ceilings):
            ceiling = super(_PereiraBenchmark.PereiraExtrapolationCeiling, self).extrapolate(ceilings)
            # compute aggregate ceiling only for language neuroids
            neuroid_ceilings = ceiling.raw
            language_ceilings = neuroid_ceilings.sel(atlas='language')
            ceiling = self.aggregate_neuroid_ceilings(language_ceilings)
            ceiling.attrs['raw'] = neuroid_ceilings  # reset to all neuroids
            return ceiling

        def fit(self, subject_subsamples, bootstrapped_scores):
            valid = ~np.isnan(bootstrapped_scores)
            if sum(valid) < 1:
                raise RuntimeError("No valid scores in sample")
            return super(_PereiraBenchmark.PereiraExtrapolationCeiling, self).fit(
                np.array(subject_subsamples)[valid], np.array(bootstrapped_scores)[valid])

        def post_process(self, scores):
            scores = apply_aggregate(lambda values: values.mean('sub_experiment').mean('experiment'), scores)
            return scores

    class PereiraHoldoutSubjectCeiling(HoldoutSubjectCeiling):
        def __init__(self, *args, **kwargs):
            super(_PereiraBenchmark.PereiraHoldoutSubjectCeiling, self).__init__(*args, **kwargs)
            self._rng = RandomState(0)
            self._num_bootstraps = 5

        def get_subject_iterations(self, subjects):
            # use only a subset of subjects
            return self._rng.choice(list(subjects), size=self._num_bootstraps)



class PereiraEncoding(_PereiraBenchmark):
    """
    data source:
        Pereira et al., nature communications 2018
        https://www.nature.com/articles/s41467-018-03068-4?fbclid=IwAR0W7EZrnIFFO1kvANgeOEICaoDG5fhmdHipazy6n-APUJ6lMY98PkvuTyU
    """

    def __init__(self, **kwargs):
        metric = CrossRegressedCorrelation(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord='stimulus_id', stratification_coord=None))
        super(PereiraEncoding, self).__init__(metric=metric, **kwargs)

    @property
    @load_s3(key='Pereira2018-encoding-ceiling')
    def ceiling(self):
        return super(PereiraEncoding, self).ceiling


class PereiraSamplerEncoding(_PereiraBenchmark):
    """
    data source:
        Pereira et al., nature communications 2018
        https://www.nature.com/articles/s41467-018-03068-4?fbclid=IwAR0W7EZrnIFFO1kvANgeOEICaoDG5fhmdHipazy6n-APUJ6lMY98PkvuTyU
    """

    def __init__(self, **kwargs):
        metric = CrossRegressedCorrelation(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord='stimulus_id', stratification_coord=None))
        super(PereiraSamplerEncoding, self).__init__(metric=metric, **kwargs)

    def _load_assembly(self,version='max'):
        return  pd.read_pickle(f'{PEREIRA2018_SAMPLE}/pereira_ds_{version}_v2.pkl')
    
    @property
    @load_s3(key='Pereira2018-encoding-ceiling')
    def ceiling(self):
        return super(PereiraSamplerEncoding, self).ceiling
    
    
    def __call__(self, candidate):
        stimulus_set = self._target_assembly.attrs['stimulus_set']
        model_activations = listen_to(candidate, stimulus_set)
        assert set(model_activations['stimulus_id'].values) == set(self._target_assembly['stimulus_id'].values)
        _logger.info('subsampling sentences')
        # filter model_activations based on optim_sentences

        target_assembly = self._target_assembly.sel(presentation=self._target_assembly['optim_sentence'])
        # select rows from model_activations based on target_assembly
        model_activations = model_activations.sel(presentation=model_activations['stimulus_id'].isin(target_assembly['stimulus_id']))
        # make sure the stimulus_id dimension is the same between the two
        assert set(model_activations['stimulus_id'].values) == set(target_assembly['stimulus_id'].values)
        _logger.info('Scoring across experiments & atlases')
        # update self._target_assembly with target_assembly
        #self._target_assembly = target_assembly
        cross_scores = self._cross(target_assembly,
                                   apply=lambda cross_assembly: self._apply_cross(model_activations, cross_assembly))
        raw_scores = cross_scores.raw
        raw_neuroids = apply_aggregate(lambda values: values.mean('split').mean('experiment'), raw_scores)

        # normally we would ceil every single neuroid here. To estimate the strongest ceiling possible (i.e. make it as
        # hard as possible on the models), we used experiment-overlapping neuroids from as many subjects as possible
        # which means some neuroids got excluded. Since median(r/c) is the same as median(r)/median(c), we just
        # normalize the neuroid aggregate by the overall ceiling aggregate.
        # Additionally, the Pereira data also has voxels from DMN, visual etc. but we care about language here.
        language_neuroids = raw_neuroids.sel(atlas='language', _apply_raw=False)
        score = aggregate_ceiling(language_neuroids, ceiling=self.ceiling, subject_column='subject')
        return score


class PereiraSamplerMaxEncoding(PereiraSamplerEncoding):
    def _load_assembly(self,version='max'):
        return super()._load_assembly(version='max')


class PereiraSamplerMinEncoding(PereiraSamplerEncoding):
    def _load_assembly(self,version='min'):
        return super()._load_assembly(version='min')


class PereiraSamplerRandEncoding(PereiraSamplerEncoding):
    def _load_assembly(self,version='rand'):
        return super()._load_assembly(version='rand')


class PereiraSamplerV2Encoding(_PereiraBenchmark):
    """
    data source:
        Pereira et al., nature communications 2018
        https://www.nature.com/articles/s41467-018-03068-4?fbclid=IwAR0W7EZrnIFFO1kvANgeOEICaoDG5fhmdHipazy6n-APUJ6lMY98PkvuTyU
    """
    def __init__(self, **kwargs):
        metric = CrossRegressedCorrelation(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord='stimulus_id', stratification_coord=None))
        super(PereiraSamplerV2Encoding, self).__init__(metric=metric, **kwargs)

    def _load_assembly(self, version='max'):
        assembly=pd.read_pickle(f'{PEREIRA2018_SAMPLE}/pereira_ds_{version}.pkl')
        # change stimulu set name  so that it is not confused with the original pereira dataset
        assembly.attrs['stimulus_set'].name=assembly.attrs['stimulus_set'].name+version+'_v2'
        return assembly

    @property
    @load_s3(key='Pereira2018-encoding-ceiling')
    def ceiling(self):
        return super(PereiraSamplerV2Encoding, self).ceiling


class PereiraSamplerMaxV2Encoding(PereiraSamplerV2Encoding):
    def _load_assembly(self,version='max'):
        return super()._load_assembly(version='max')


class PereiraSamplerMinV2Encoding(PereiraSamplerV2Encoding):
    def _load_assembly(self,version='min'):
        return super()._load_assembly(version='min')


class PereiraSamplerRandV2Encoding(PereiraSamplerV2Encoding):
    def _load_assembly(self,version='rand'):
        return super()._load_assembly(version='rand')


class PereiraNormalizedEncoding(_PereiraBenchmark):
    """
    data source:
        Pereira et al., nature communications 2018
        https://www.nature.com/articles/s41467-018-03068-4?fbclid=IwAR0W7EZrnIFFO1kvANgeOEICaoDG5fhmdHipazy6n-APUJ6lMY98PkvuTyU
    """

    def __init__(self, **kwargs):
        metric = CrossRegressedCorrelation(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord='stimulus_id', stratification_coord=None))
        super(PereiraNormalizedEncoding, self).__init__(metric=metric, **kwargs)

    def _load_assembly(self,version='zscored'):
        assembly=xr.load_dataarray(f'{PEREIRA2018_SAMPLE}/Pereira2018.nc')
        assembly_zs=[]
        for exp_id,exp in assembly.groupby('experiment'):
            # zscore indivdual neuroids in exp across presentation dimension
            a=stats.zscore(exp.values, axis=0)
            exp_zs=exp.copy(data=a)
            assembly_zs.append(exp_zs)
        assembly_zs=xr.concat(assembly_zs,dim='presentation')
        # select only langauge atlas
        language_atlas=assembly_zs.atlas.values=='language'
        assembly_zs=assembly_zs.sel(neuroid=language_atlas)
        # copy over the attributes from assembly
        assembly_zs.attrs=assembly.attrs
        # explicitly load the stimulus set
        stimulus_set_file=assembly_zs.attrs['stimulus_set'].replace('s3:',f'{PEREIRA2018_SAMPLE}/')
        stimulus_set=pd.read_csv(stimulus_set_file)

        assembly_zs.attrs['stimulus_set']=StimulusSet(stimulus_set)
        assembly_zs.attrs['stimulus_set'].name='Pereira2018'
        assembly_zs.attrs['version']='zscored'
        assembly_zs=NeuroidAssembly(assembly_zs)
        return assembly_zs

    @property
    @load_s3(key='Pereira2018-encoding-ceiling')
    def ceiling(self):
        return super(PereiraNormalizedEncoding, self).ceiling


class PereiraNormalizedSentenceEncoding(PereiraNormalizedEncoding):
    def __call__(self, candidate):
        stimulus_set = self._target_assembly.attrs['stimulus_set']
        model_activations = listen_to(candidate, stimulus_set,reset_column='stimulus_id')
        assert set(model_activations['stimulus_id'].values) == set(self._target_assembly['stimulus_id'].values)
        _logger.info('Scoring across experiments & atlases')
        cross_scores = self._cross(self._target_assembly,
                                   apply=lambda cross_assembly: self._apply_cross(model_activations, cross_assembly))
        raw_scores = cross_scores.raw
        raw_neuroids = apply_aggregate(lambda values: values.mean('split').mean('experiment'), raw_scores)

        # normally we would ceil every single neuroid here. To estimate the strongest ceiling possible (i.e. make it as
        # hard as possible on the models), we used experiment-overlapping neuroids from as many subjects as possible
        # which means some neuroids got excluded. Since median(r/c) is the same as median(r)/median(c), we just
        # normalize the neuroid aggregate by the overall ceiling aggregate.
        # Additionally, the Pereira data also has voxels from DMN, visual etc. but we care about language here.
        language_neuroids = raw_neuroids.sel(atlas='language', _apply_raw=False)
        score = aggregate_ceiling(language_neuroids, ceiling=self.ceiling, subject_column='subject')
        return score

class PereiraNormalizedEncoding_V2(PereiraNormalizedEncoding):
    def __call__(self, candidate):
        stimulus_set = self._target_assembly.attrs['stimulus_set']
        model_activations = listen_to(candidate, stimulus_set)
        model_activations_zs=[]
        for exp_id,exp in model_activations.groupby('experiment'):
            # zscore indivdual neuroids in exp across presentation dimension
            a=stats.zscore(exp.values, axis=0)
            exp_zs=exp.copy(data=a)
            model_activations_zs.append(exp_zs)
        model_activations_zs=xr.concat(model_activations_zs,dim='presentation')
        assert set(model_activations_zs['stimulus_id'].values) == set(self._target_assembly['stimulus_id'].values)
        _logger.info('Scoring across experiments & atlases')
        cross_scores = self._cross(self._target_assembly,
                                   apply=lambda cross_assembly: self._apply_cross(model_activations_zs, cross_assembly))
        raw_scores = cross_scores.raw
        raw_neuroids = apply_aggregate(lambda values: values.mean('split').mean('experiment'), raw_scores)

        # normally we would ceil every single neuroid here. To estimate the strongest ceiling possible (i.e. make it as
        # hard as possible on the models), we used experiment-overlapping neuroids from as many subjects as possible
        # which means some neuroids got excluded. Since median(r/c) is the same as median(r)/median(c), we just
        # normalize the neuroid aggregate by the overall ceiling aggregate.
        # Additionally, the Pereira data also has voxels from DMN, visual etc. but we care about language here.
        language_neuroids = raw_neuroids.sel(atlas='language', _apply_raw=False)
        score = aggregate_ceiling(language_neuroids, ceiling=self.ceiling, subject_column='subject')
        return score



#########################
### BENCHMARKS (AUDITORY)
#########################

#specify split coordinate for cross-validation
if os.getenv('SPLIT_AT_PASSAGE', '0') == '1':
    try:
        print("THIS SPLITS AT *ORIGINAL* PASSAGE ID")
        pereira_split_coord = 'old_stimulus_passage_index'
    except:
        print("THIS SPLITS AT *NEW* PASSAGE ID")
        pereira_split_coord = 'stimulus_passage_index'
elif os.getenv('SPLIT_AT_TOPIC', '0') == '1':
    pereira_split_coord = 'stimulus_passage_category'
else:
    pereira_split_coord = 'stimulus_id'

_logger.info(f"\nCross validation split coordinate is {pereira_split_coord}\n")

class _Pereira2023audBenchmark(Benchmark):
    """
    NEW AUDITORY PEREIRA EXPERIMENT
    """

    def __init__(self, identifier, metric, version='sent', reset_column='stimulus_id', threshold=90):
        self._identifier = identifier
        assembly = self._load_assembly(threshold=threshold, version=version)
        self._target_assembly = assembly
        self._metric = metric
        self._reset_column = reset_column
        # self._ceiler = self.PereiraExtrapolationCeiling(subject_column='subject', num_bootstraps=100)
        self._cross = CartesianProduct(dividers=['experiment', 'atlas'])

    @property
    def identifier(self):
        return self._identifier

    def _read_words(self, candidate, stimulus_set, reset_column='stimulus_id', copy_columns=(), average_sentence=False):
        """
        Pass a `stimulus_set` through a model `candidate`.
        In contrast to the `listen_to` function, this function operates on a word-based `stimulus_set`.
        """
        # Input: stimulus_set = pandas df, col 1 with sentence ID and 2nd col as word.
        activations = []
        for i, reset_id in enumerate(ordered_set(stimulus_set[reset_column].values)):
            part_stimuli = stimulus_set[stimulus_set[reset_column] == reset_id]
            # stimulus_ids = part_stimuli['stimulus_id']
            sentence_stimuli = StimulusSet({'sentence': part_stimuli.values,
                                            reset_column: list(set(part_stimuli[reset_column].values))})
            sentence_stimuli.name = f"{self._target_assembly.stimuli_group}-{reset_column}-{reset_id}"
            sentence_activations = candidate(stimuli=sentence_stimuli, average_sentence=average_sentence)[-1, :]
            # for column in copy_columns:
            #    sentence_activations[column] = ('presentation', part_stimuli[column])
            activations.append(sentence_activations)

        # model_activations = merge_data_arrays(activations)
        model_activations = xr.concat(activations, dim='presentation')
        # merging does not maintain stimulus order. the following orders again
        idx = [model_activations[reset_column].values.tolist().index(stimulus_id) for stimulus_id in
               [int(s[reset_column].values) for s in activations]]
        assert len(set(idx)) == len(idx), "Found duplicate indices to order activations"
        model_activations = model_activations[{'presentation': idx}]

        return model_activations
    

    def _metric(self, source_assembly, target_assembly):
        """ for ceiling compute """
        cross_scores = self._cross(target_assembly, apply=
        lambda cross_assembly: self._apply_cross(source_assembly, cross_assembly))
        score = self._average_cross_scores(cross_scores)
        return score

    def _average_cross_scores(self, cross_scores):
        return cross_scores.mean(['experiment', 'atlas'])

    # @load_s3(key='Pereira2018')
    def _load_assembly(self,version='sent', threshold=90):
        try:
            assembly = pd.read_pickle(f'{fMRI_PARENT}/Pereira2023aud_{version}_train_language_top_{threshold}_CK.pkl')
            _logger.warning(f'Using the following assembly file: {fMRI_PARENT}/Pereira2023aud_{version}_train_language_top_{threshold}_CK.pkl')
        except:
            assembly = pd.read_pickle(f'{fMRI_PARENT}/Pereira2023aud_{version}_train_language_top_{threshold}_V2.pkl')
            _logger.warning(f'Using the following assembly file: {fMRI_PARENT}/Pereira2023aud_{version}_train_language_top_{threshold}_V2.pkl')
        # select stimuli that have the stim_group= version
        vox_reliability = {'language': (False, .95), 'auditory': (False, .95), 'visual': (False, .95)}
        vox_corr = {'language': (False, .1), 'auditory': (False, .1), 'visual': (False, .1)}
        if vox_reliability['language'][0]:
            vox_rel_vec = (assembly.repetition_corr_ratio > vox_reliability['language'][1]).values
        else:
            vox_rel_vec = (assembly.repetition_corr_ratio > -np.inf).values

        if vox_corr['language'][0]:
            vox_corr_vec = (assembly.repetition_corr > vox_corr['language'][1]).values
        else:
            vox_corr_vec = (assembly.repetition_corr > -np.inf).values
        vox_selection = np.logical_and(vox_corr_vec, vox_rel_vec)
        assembly = assembly.sel(neuroid=vox_selection)
        assembly.attrs['stimuli_group'] = f'Pereira2023aud_{version}'
        # combine simulus_passage_category and stimulus_passage_index to create a unique stimulus_id using a for loop
        stimulus_passage_category_id = [f'{assembly["stimulus_passage_category"].values[i]}_{assembly["stimulus_passage_index"].values[i]}' for i in range(len(assembly['stimulus_passage_category'].values))]
        # use assign coords to add the new stimulus_passage_category_id to presentation diemsnions to the assembly
        assembly = assembly.assign_coords(stimulus_passage_category_id=('presentation', stimulus_passage_category_id))

        # construct stimulus set from the assembly
        stimulus_set = StimulusSet({'sentence': assembly['stimulus'].values,
                                    'stimulus_num': assembly['stimulus_num'].values,
                                    'stimulus_id': assembly['stimulus_id'].values,
                                    'stimulus_passage_category': assembly['stimulus_passage_category'].values,
                                    'stimulus_passage_index': assembly['stimulus_passage_index'].values,
                                    'stimulus_story': assembly['stimulus_story'].values,
                                    'stim_name': assembly['stim_name'].values,
                                    'experiment': assembly['experiment'].values,
                                    'stumulus': assembly['stimulus'].values,
                                    'stimulus_passage_category_id': assembly['stimulus_passage_category_id'].values})
        # attach stimulus set as an attribute to the assembly
        # add name to stimulus set
        stimulus_set.name = f"{assembly.attrs['stimuli_group']}"
        assembly.attrs['stimulus_set'] = stimulus_set                   
        return assembly

    def __call__(self, candidate):
        stimulus_set = self._target_assembly.attrs['stimulus_set']
        _logger.warning(f'extracting activation on {self._reset_column}')
            # #model_activations = self._read_words(candidate, stimulus_set, copy_columns=['word_id'],
            # #                                     reset_column='stimulus_id')
        model_activations = listen_to(candidate, stimulus_set,reset_column=self._reset_column)
        #model_activations = self._get_model_activations(candidate)

        # make sure model_activations and target_assembly have the same order of stimuli
        assert (model_activations['stimulus_id'].values == self._target_assembly['stimulus_id'].values).all()
        _logger.info('Scoring across experiments & atlases')
        cross_scores = self._cross(self._target_assembly,
                                   apply=lambda cross_assembly: self._apply_cross(model_activations, cross_assembly))
        raw_scores = cross_scores.raw
        raw_neuroids = apply_aggregate(lambda values: values.mean('split'), raw_scores)

        # normally we would ceil every single neuroid here. To estimate the strongest ceiling possible (i.e. make it as
        # hard as possible on the models), we used experiment-overlapping neuroids from as many subjects as possible
        # which means some neuroids got excluded. Since median(r/c) is the same as median(r)/median(c), we just
        # normalize the neuroid aggregate by the overall ceiling aggregate.
        # Additionally, the Pereira data also has voxels from DMN, visual etc. but we care about language here.
        language_neuroids = raw_neuroids.sel(atlas='language', _apply_raw=False)
        # score = self._aggregate_no_ceiling(language_neuroids, ceiling=[], subject_column='subject')
        score = self._aggregate_no_ceiling(language_neuroids, subject_column='subject')
        return score

    def _apply_cross(self, source_assembly, cross_assembly):
        cross_assembly = cross_assembly.dropna('neuroid')  # some subjects have only done one experiment
        source_assembly = source_assembly.dropna('neuroid')  # only relevant when running audio-visual self as "model"
        assert not np.isnan(cross_assembly).any()
        source_assembly = source_assembly[{'presentation': [stimulus_id in cross_assembly['stimulus_id'].values
                                                            for stimulus_id in source_assembly['stimulus_id'].values]}]
        return self._metric(source_assembly, cross_assembly)

    @property
    def ceiling(self):
        return self._ceiler(identifier=self.identifier, assembly=self._target_assembly, metric=self._metric)

    def _aggregate_ceiling(self, neuroid_scores, ceiling, subject_column='subject'):
        aggregate_raw = self._aggregate_neuroid_scores(neuroid_scores, subject_column=subject_column)
        score = consistency(aggregate_raw, ceiling)
        score.attrs['raw'] = aggregate_raw
        score.attrs['ceiling'] = ceiling
        score.attrs['description'] = "ceiling-normalized score"
        return score

    def _aggregate_no_ceiling(self, neuroid_scores, subject_column='subject'):
        aggregate_raw = self._aggregate_neuroid_scores(neuroid_scores, subject_column=subject_column)
        score = aggregate_raw
        score.attrs['description'] = "no_ceiling-normalized score"
        return score

    def _aggregate_neuroid_scores(self, neuroid_scores, subject_column):
        subject_scores = neuroid_scores.groupby(subject_column).median()
        center = subject_scores.median(subject_column).values
        subject_values = np.nan_to_num(subject_scores.values,
                                       nan=0)  # mad cannot deal with all-nan in one axis, treat as 0
        subject_axis = subject_scores.dims.index(subject_scores[subject_column].dims[0])
        error = median_absolute_deviation(subject_values, axis=subject_axis)
        # score = Score([center, error], coords={'aggregation': ['center', 'error']}, dims=['aggregation'])
        score = Score([float(center), float(error)], coords={'aggregation': ['center', 'error']}, dims=['aggregation'])
        score.attrs['raw'] = neuroid_scores
        score.attrs['description'] = "score aggregated by taking median of neuroids per subject, " \
                                     "then median of subject scores"
        return score


class Pereira2023audRidgeEncoding(_Pereira2023audBenchmark):
    def __init__(self, **kwargs):
        metric = CrossRegressedCorrelation(
            regression=rgcv_linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord=pereira_split_coord, stratification_coord=None))
        super(Pereira2023audRidgeEncoding, self).__init__(metric=metric, **kwargs)


class Pereira2023audV2RidgeEncoding(_Pereira2023audBenchmark):
    def __init__(self, **kwargs):
        metric = CrossRegressedCorrelation(
            regression=rgcv_linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord=pereira_split_coord, stratification_coord=None))
        super(Pereira2023audRidgeEncoding, self).__init__(metric=metric, **kwargs)
    def __call__(self, candidate):
        stimulus_set = self._target_assembly.attrs['stimulus_set']
        _logger.warning(f'extracting activation on {self._reset_column}')
            # #model_activations = self._read_words(candidate, stimulus_set, copy_columns=['word_id'],
            # #                                     reset_column='stimulus_id')
        model_activations = read_words(candidate, stimulus_set,reset_column=self._reset_column)
        #model_activations = self._get_model_activations(candidate)

        # make sure model_activations and target_assembly have the same order of stimuli
        assert (model_activations['stimulus_id'].values == self._target_assembly['stimulus_id'].values).all()
        _logger.info('Scoring across experiments & atlases')
        cross_scores = self._cross(self._target_assembly,
                                   apply=lambda cross_assembly: self._apply_cross(model_activations, cross_assembly))
        raw_scores = cross_scores.raw
        raw_neuroids = apply_aggregate(lambda values: values.mean('split'), raw_scores)

        # normally we would ceil every single neuroid here. To estimate the strongest ceiling possible (i.e. make it as
        # hard as possible on the models), we used experiment-overlapping neuroids from as many subjects as possible
        # which means some neuroids got excluded. Since median(r/c) is the same as median(r)/median(c), we just
        # normalize the neuroid aggregate by the overall ceiling aggregate.
        # Additionally, the Pereira data also has voxels from DMN, visual etc. but we care about language here.
        language_neuroids = raw_neuroids.sel(atlas='language', _apply_raw=False)
        # score = self._aggregate_no_ceiling(language_neuroids, ceiling=[], subject_column='subject')
        score = self._aggregate_no_ceiling(language_neuroids, subject_column='subject')
        return score


class Pereira2023audSentRidgeEncoding(Pereira2023audRidgeEncoding):
    def __init__(self, **kwargs):
        super(Pereira2023audSentRidgeEncoding, self).__init__(reset_column='stimulus_id',**kwargs)
    def _load_assembly(self,version='sent',threshold=90):
        return super()._load_assembly(version='sent',threshold=90)

# class Pereira2023audPassRidgeEncoding(Pereira2023audRidgeEncoding):
#     def _load_assembly(self,version='pass',threshold=90):
#         return super()._load_assembly(version='pass',threshold=90)


class Pereira2023audPassPassageRidgeEncoding(Pereira2023audRidgeEncoding):
    def __init__(self, **kwargs):
        super(Pereira2023audPassPassageRidgeEncoding, self).__init__(reset_column='stimulus_passage_category_id',**kwargs)
    def _load_assembly(self,version='pass',threshold=90):
        return super()._load_assembly(version='pass',threshold=90)
    
   
class Pereira2023audPassSentenceRidgeEncoding(Pereira2023audRidgeEncoding):
    def __init__(self, **kwargs):
        super(Pereira2023audPassSentenceRidgeEncoding, self).__init__(reset_column='stimulus_id',**kwargs)
    def _load_assembly(self, version='pass', threshold=90):
        return super()._load_assembly(version='pass', threshold=90)

    def _get_model_activations(self, candidate, reset_column='stimulus_id',
                               copy_columns=['stimulus_id']):
        return super()._get_model_activations(candidate, reset_column='stimulus_id',
                                              copy_columns=['stimulus_id'])

class Pereira2023audSentSentenceRidgeEncoding(Pereira2023audRidgeEncoding):
    def __init__(self, **kwargs):
        super(Pereira2023audSentSentenceRidgeEncoding, self).__init__(reset_column='stimulus_id',**kwargs)
    def _load_assembly(self, version='sent', threshold=90):
        return super()._load_assembly(version='sent', threshold=90)

    # def _get_model_activations(self, candidate, reset_column='stimulus_id',
    #                            copy_columns=['stimulus_id']):
    #     return super()._get_model_activations(candidate, reset_column='stimulus_id',
    #                                           copy_columns=['stimulus_id'])

class Pereira2023audSentPassageRidgeEncoding(Pereira2023audRidgeEncoding):
    def __init__(self, **kwargs):
        super(Pereira2023audSentPassageRidgeEncoding, self).__init__(reset_column='stimulus_passage_index',**kwargs) #CK was stimulus_passage_category_id
    def _load_assembly(self, version='sent', threshold=80):
        return super()._load_assembly(version='sent', threshold=80) #threshold was 90

    # def _get_model_activations(self, candidate, reset_column='stimulus_id',
    #                            copy_columns=['stimulus_id']):
    #     return super()._get_model_activations(candidate, reset_column='stimulus_id',
    #                                           copy_columns=['stimulus_id'])


class Pereira2023audPassPassageSampleRidgeEncoding(Pereira2023audRidgeEncoding):
    def __init__(self, **kwargs):
        super(Pereira2023audPassPassageSampleRidgeEncoding, self).__init__(reset_column='stimulus_passage_category_id',**kwargs)
    def _load_assembly(self,version='pass',threshold=90):
        return super()._load_assembly(version='pass',threshold=90)
    def __call__(self, candidate):
        stimulus_set = self._target_assembly.attrs['stimulus_set']
        _logger.warning(f'extracting activation on {self._reset_column}')
            # #model_activations = self._read_words(candidate, stimulus_set, copy_columns=['word_id'],
            # #                                     reset_column='stimulus_id')
        model_activations = listen_to(candidate, stimulus_set,reset_column=self._reset_column)
        assert (model_activations['stimulus_id'].values == self._target_assembly['stimulus_id'].values).all()
        # randomly select 80 stimulus_ids,
        random_stimulus_ids = np.random.choice(model_activations['stimulus_id'].values, 80, replace=False)
        # find location of random_stimulus_ids in model_activations.stimulus_id
        random_stimulus_ids_idx = np.where(np.isin(model_activations['stimulus_id'].values, random_stimulus_ids))[0]
        model_activation_sample=model_activations[random_stimulus_ids_idx,:]
        target_assembly_sample=self._target_assembly[random_stimulus_ids_idx,:]
        assert (model_activation_sample['stimulus_id'].values == target_assembly_sample['stimulus_id'].values).all()
        

        _logger.info('Scoring across experiments & atlases')
        cross_scores = self._cross(target_assembly_sample,
                                   apply=lambda cross_assembly: self._apply_cross(model_activation_sample, cross_assembly))
        raw_scores = cross_scores.raw
        raw_neuroids = apply_aggregate(lambda values: values.mean('split'), raw_scores)

        # normally we would ceil every single neuroid here. To estimate the strongest ceiling possible (i.e. make it as
        # hard as possible on the models), we used experiment-overlapping neuroids from as many subjects as possible
        # which means some neuroids got excluded. Since median(r/c) is the same as median(r)/median(c), we just
        # normalize the neuroid aggregate by the overall ceiling aggregate.
        # Additionally, the Pereira data also has voxels from DMN, visual etc. but we care about language here.
        language_neuroids = raw_neuroids.sel(atlas='language', _apply_raw=False)
        # score = self._aggregate_no_ceiling(language_neuroids, ceiling=[], subject_column='subject')
        score = self._aggregate_no_ceiling(language_neuroids, subject_column='subject')
        return score


class Pereira2023audEncoding(_Pereira2023audBenchmark):
    def __init__(self, **kwargs):
        metric = CrossRegressedCorrelation(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord=pereira_split_coord, stratification_coord=None))
        super(Pereira2023audEncoding, self).__init__(metric=metric, **kwargs)


class Pereira2023audPassPassageEncoding(Pereira2023audEncoding):
    def __init__(self, **kwargs):
        super(Pereira2023audPassPassageEncoding, self).__init__(reset_column='stimulus_passage_category_id',
                                                                     **kwargs)

    def _load_assembly(self, version='pass', threshold=90):
        return super()._load_assembly(version='pass', threshold=90)


class Pereira2023audPassSentenceEncoding(Pereira2023audEncoding):
    def __init__(self, **kwargs):
        super(Pereira2023audPassSentenceEncoding, self).__init__(reset_column='stimulus_id', **kwargs)

    def _load_assembly(self, version='pass', threshold=90):
        return super()._load_assembly(version='pass', threshold=90)

    def _get_model_activations(self, candidate, reset_column='stimulus_id',
                               copy_columns=['stimulus_id']):
        return super()._get_model_activations(candidate, reset_column='stimulus_id',
                                              copy_columns=['stimulus_id'])


class Pereira2023audSentSentenceEncoding(Pereira2023audEncoding):
    def __init__(self, **kwargs):
        super(Pereira2023audSentSentenceEncoding, self).__init__(reset_column='stim_name', **kwargs)

    def _load_assembly(self, version='sent', threshold=90):
        return super()._load_assembly(version='sent', threshold=90)

    # def _get_model_activations(self, candidate, reset_column='stimulus_id',
    #                            copy_columns=['stimulus_id']):
    #     return super()._get_model_activations(candidate, reset_column='stimulus_id',
    #                                           copy_columns=['stimulus_id'])


class Pereira2023audSentPassageEncoding(Pereira2023audEncoding):
    def __init__(self, **kwargs):
        super(Pereira2023audSentPassageEncoding, self).__init__(reset_column='stimulus_passage_category_id',
                                                                     **kwargs)

    def _load_assembly(self, version='sent', threshold=90):
        return super()._load_assembly(version='sent', threshold=90)

    # def _get_model_activations(self, candidate, reset_column='stimulus_id',
    #                            copy_columns=['stimulus_id']):
    #     return super()._get_model_activations(candidate, reset_column='stimulus_id',
    #                                           copy_columns=['stimulus_id'])



benchmark_pool = [
    # primary benchmarks
    ('Pereira2018-encoding', PereiraEncoding),
    ('Pereira2018-max-encoding', PereiraSamplerMaxEncoding),
    ('Pereira2018-max-V2-encoding', PereiraSamplerMaxV2Encoding),
    ('Pereira2018-min-V2-encoding', PereiraSamplerMinV2Encoding),
    ('Pereira2018-rand-V2-encoding', PereiraSamplerRandV2Encoding),
    ('Pereira2018-norm-encoding',PereiraNormalizedEncoding),
    ('Pereira2018-norm-v2-encoding',PereiraNormalizedEncoding_V2),
    ('Pereira2018-norm-sentence-encoding',PereiraNormalizedSentenceEncoding),
    ('Pereira2023aud-sent-RidgeEncoding', Pereira2023audSentRidgeEncoding),
    ('Pereira2023aud-pass-passage-RidgeEncoding', Pereira2023audPassPassageRidgeEncoding),
    ('Pereira2023aud-pass-passage-Encoding', Pereira2023audPassPassageEncoding),
    ('Pereira2023aud-pass-passage-sample-RidgeEncoding', Pereira2023audPassPassageSampleRidgeEncoding),
    ('Pereira2023aud-pass-sentence-RidgeEncoding', Pereira2023audPassSentenceRidgeEncoding),
    ('Pereira2023aud-pass-sentence-Encoding', Pereira2023audPassSentenceEncoding),
    ('Pereira2023aud-sent-sentence-RidgeEncoding', Pereira2023audSentSentenceRidgeEncoding),
    ('Pereira2023aud-sent-sentence-Encoding', Pereira2023audSentSentenceEncoding),
    ('Pereira2023aud-sent-passage-RidgeEncoding', Pereira2023audSentPassageRidgeEncoding),
    ('Pereira2023aud-sent-passage-Encoding', Pereira2023audSentPassageEncoding),
    ('Pereira2018-min-encoding', PereiraSamplerMinEncoding),
    ('Pereira2018-rand-encoding', PereiraSamplerRandEncoding)
]

benchmark_pool = {identifier: LazyLoad(lambda identifier=identifier, ctr=ctr: ctr(identifier=identifier))
                  for identifier, ctr in benchmark_pool}
