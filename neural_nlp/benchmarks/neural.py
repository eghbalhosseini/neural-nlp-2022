"""
Neural benchmarks to probe match of model internals against human internals.
"""
from scipy.optimize import curve_fit
import warnings
from collections import defaultdict
import itertools
import logging
import numpy as np
import xarray as xr
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

if getpass.getuser() == 'eghbalhosseini':
    ANNfMRI_PARENT = '/Users/eghbalhosseini/MyData/neural_nlp_bench/dataset/'
    ANNECOG_PARENT = '/Users/eghbalhosseini/MyData/neural_nlp_bench/dataset/'
    PEREIRA2018_SAMPLE = '/Users/eghbalhosseini/MyData/neural_nlp_bench/dataset//'
    DsParametricfMRI_PARENT = '/Users/eghbalhosseini/MyData/neural_nlp_bench/dataset/'


elif getpass.getuser() == 'ehoseini':
    ANNfMRI_PARENT = '/om2/user/ehoseini/MyData/neural_nlp_bench/dataset/'
    ANNECOG_PARENT = '/om2/user/ehoseini/MyData/neural_nlp_bench/dataset/'
    PEREIRA2018_SAMPLE='/om2/user/ehoseini/MyData/neural_nlp_bench/dataset/'
    DsParametricfMRI_PARENT = '/nese/mit/group/evlab/u/ehoseini/MyData/fmri_DNN/outputs/'


_logger = logging.getLogger(__name__)


class Invert:
    def __init__(self, metric):
        self._metric = metric

    def __call__(self, source, target):
        source, target = target, source
        return self._metric(source, target)


class StoriesRDMBenchmark:
    def __init__(self, bold_shift=4):
        assemblies = self._load_rdms(bold_shift_seconds=bold_shift)
        assemblies = {story: rdm for story, rdm in assemblies.items() if story != 'Elvis'}
        self._target_assemblies = assemblies
        self._metric = RDMSimilarityCrossValidated()
        self._cross_region = CartesianProduct(dividers=['region'])

    def _load_rdms(self, roi_filter='from90to100', bold_shift_seconds=4):
        assemblies = {}
        for story in ['Boar', 'KingOfBirds', 'Elvis', 'HighSchool', 'MatchstickSeller']:
            assembly = load_rdm_sentences(story=story, roi_filter=roi_filter, bold_shift_seconds=bold_shift_seconds)
            assembly = assembly.mean(dim='subject')
            stimulus_set_identifier = f'naturalistic-neural-reduced.{story}'
            stimulus_set = load_stimuli(stimulus_set_identifier)
            stimulus_set = StimulusSet({'sentence': stimulus_set})
            stimulus_set.name = stimulus_set_identifier
            assembly.attrs['stimulus_set'] = stimulus_set
            assemblies[story] = assembly
        return assemblies

    def __call__(self, candidate):
        scores = []
        for story, story_assembly in self._target_assemblies.items():
            source_assembly = candidate(stimuli=story_assembly.stimulus_set)
            score = self._cross_region(story_assembly,
                                       apply=lambda region_assembly: self._metric(source_assembly, region_assembly))
            score = score.expand_dims('story')
            score['story'] = [story]
            scores.append(score)
        score = Score.merge(*scores)
        score = apply_aggregate(lambda score: score.mean('story'), score)
        score = apply_aggregate(lambda score: score.mean('region'), score)
        return score

class RDMSimilarityCrossValidated:
    # adapted from
    # https://github.com/brain-score/brain-score/blob/3d59d7a841fca63a5d346e599143f547560b5082/brainscore/metrics/rdm.py#L8

    class LeaveOneOutWrapper:
        def __init__(self, metric):
            self._metric = metric

        def __call__(self, train_source, train_target, test_source, test_target):
            # compare assemblies for a single split. we ignore the 10% train ("leave-one-out") and only use test.
            score = self._metric(test_source, test_target)
            return DataAssembly(score)

    def __init__(self, stimulus_coord='stimulus_sentence'):
        self._rdm = RDM()
        self._similarity = RDMSimilarity(comparison_coord=stimulus_coord)
        self._cross_validation = CrossValidation(test_size=.9,  # leave 10% out
                                                 split_coord=stimulus_coord, stratification_coord=None)

    def __call__(self, model_activations, target_rdm):
        model_activations = align(model_activations, target_rdm, on='stimulus_sentence')
        model_rdm = self._rdm(model_activations)
        values = model_rdm.values
        if np.isnan(values.flatten()).any():
            warnings.warn(f"{np.isnan(values.flatten()).sum()} nan values found in model rdm - setting to 0")
            values[np.isnan(values)] = 0
            model_rdm = type(model_rdm)(values, coords={coord: (dims, vals) for coord, dims, vals in
                                                        walk_coords(model_rdm)}, dims=model_rdm.dims)
        leave_one_out = self.LeaveOneOutWrapper(self._similarity)
        # multi-dimensional coords with repeated dimensions not yet supported in CrossValidation
        drop_coords = [coord for coord, dims, value in walk_coords(target_rdm) if dims == ('stimulus', 'stimulus')]
        target_rdm = target_rdm.drop(drop_coords)
        return self._cross_validation(model_rdm, target_rdm, apply=leave_one_out)

def align(source, target, on):
    source_values, target_values = source[on].values.tolist(), target[on].values
    indices = [source_values.index(value) for value in target_values]
    assert len(source[on].dims) == 1, "multi-dimensional coordinates not implemented"
    dim = source[on].dims[0]
    dim_indices = {_dim: slice(None) if _dim != dim else indices for _dim in source.dims}
    aligned = source.isel(**dim_indices)
    return aligned

class Blank2014VoxelEncoding(Benchmark):
    """
    data source:
        Blank et al., Journal of Neurophysiology 2014
        https://journals.physiology.org/doi/full/10.1152/jn.00884.2013
    """

    def __init__(self, identifier, bold_shift=4):
        self._identifier = identifier
        assembly = LazyLoad(lambda: self._load_assembly(bold_shift))
        self._target_assembly = assembly
        regression = linear_regression(xarray_kwargs=dict(
            stimulus_coord='stimulus_id', neuroid_coord='neuroid_id'))
        correlation = pearsonr_correlation(xarray_kwargs=dict(
            correlation_coord='stimulus_id', neuroid_coord='neuroid_id'))
        self._metric = CrossRegressedCorrelation(
            regression=regression, correlation=correlation,
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord='stimulus_id', stratification_coord='story'))

        self._ceiler = ExtrapolationCeiling(subject_column='subject_UID', post_process=self.post_process_ceilings)

    @property
    def identifier(self):
        return self._identifier

    def _load_assembly(self, bold_shift):
        assembly = load_voxels(bold_shift_seconds=bold_shift)
        return assembly

    def post_process_ceilings(self, scores):
        if not hasattr(scores, 'neuroid_id'):
            scores['neuroid_id'] = 'neuroid', [".".join([str(value) for value in values]) for values in zip(*[
                scores[coord].values for coord in ['subject_UID', 'fROI_area']])]
        return scores

    @property
    def ceiling(self):
        return self._ceiler(identifier=self.identifier, assembly=self._target_assembly, metric=self._metric)

    def __call__(self, candidate):
        _logger.info('Computing activations')
        model_activations = listen_to(candidate, self._target_assembly.attrs['stimulus_set'])
        assert set(model_activations['stimulus_id'].values) == set(self._target_assembly['stimulus_id'].values)
        _logger.info('Scoring model')
        score = self.apply_metric(model_activations, self._target_assembly)
        score = self.ceiling_normalize(score)
        return score

    def apply_metric(self, model_activations, target_assembly):
        return self._metric(model_activations, target_assembly)

    def ceiling_normalize(self, score):
        raw_neuroids = apply_aggregate(lambda values: values.mean('split'), score.raw)
        score = ceil_neuroids(raw_neuroids, self.ceiling, subject_column='subject_UID')
        return score

class Blank2014fROIEncoding(Blank2014VoxelEncoding):
    """
    data source:
        Blank et al., Journal of Neurophysiology 2014
        https://journals.physiology.org/doi/full/10.1152/jn.00884.2013
    """

    def __init__(self, *args, **kwargs):
        super(Blank2014fROIEncoding, self).__init__(*args, **kwargs)

        regression = linear_regression(xarray_kwargs=dict(
            stimulus_coord='stimulus_id', neuroid_coord='fROI_area'))
        correlation = pearsonr_correlation(xarray_kwargs=dict(
            correlation_coord='stimulus_id', neuroid_coord='fROI_area'))
        self._metric = CrossRegressedCorrelation(
            regression=regression, correlation=correlation,
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord='stimulus_id', stratification_coord='story'))

    @load_s3(key='Blank2014fROI')
    def _load_assembly(self, bold_shift):
        assembly = super(Blank2014fROIEncoding, self)._load_assembly(bold_shift)
        assembly = self.average_subregions(bold_shift=bold_shift, assembly=assembly)
        return assembly

    @store(identifier_ignore=['assembly'])
    def average_subregions(self, bold_shift, assembly):
        attrs = assembly.attrs
        del assembly['threshold']
        # group by stimuli, fROI, subject after one another.
        # this gets rid of adjacent coords unfortunately, but we accept that for now.
        averaged_assembly = assembly.groupby('stimulus_id').apply(
            lambda stimulus_group: stimulus_group.groupby('fROI_area').apply(
                lambda fROI_group: fROI_group.groupby('subject_UID').mean()
            ))
        averaged_assembly = averaged_assembly.stack(presentation=['stimulus_id'], neuroid=['fROI_area', 'subject_UID'])
        # copy presentation coords back since those are needed for e.g. metric stratification
        order = [averaged_assembly['stimulus_id'].values.tolist().index(stimulus_id)
                 for stimulus_id in assembly['stimulus_id'].values]
        for copy_coord, dims, copy_value in walk_coords(assembly):
            if not array_is_element(dims, 'presentation') or hasattr(averaged_assembly, copy_coord):
                continue
            averaged_assembly[copy_coord] = dims, copy_value[order]
        averaged_assembly.attrs = attrs
        averaged_assembly['neuroid_id'] = 'neuroid', [".".join([str(value) for value in values]) for values in zip(*[
            averaged_assembly[coord].values for coord in ['subject_UID', 'fROI_area']])]
        return averaged_assembly

    @property
    @load_s3(key='Blank2014fROI-encoding-ceiling')
    def ceiling(self):
        return super(Blank2014fROIEncoding, self).ceiling

class Blank2014SentencefROIEncoding(Blank2014fROIEncoding):
    def __init__(self, *args, sentence_num, **kwargs):
        super(Blank2014SentencefROIEncoding, self).__init__(*args, **kwargs)
        self.sentence_num = sentence_num

    def _load_assembly(self, bold_shift):
        assembly = super(Blank2014fROIEncoding, self)._load_assembly(bold_shift)
        # choose only up to nth sentence
        # stimulus_id is ['story', 'sentence_num', 'sentence_part']
        assembly = assembly[{'presentation': [
            int(stimulus_id.split('.')[1]) == self.sentence_num
            for stimulus_id in assembly['stimulus_id'].values]}]
        return assembly

    def __call__(self, candidate):
        _logger.info('Computing activations')
        model_activations = listen_to(candidate, self._target_assembly.attrs['stimulus_set'])
        assert all(stimulus_id in set(model_activations['stimulus_id'].values)
                   for stimulus_id in set(self._target_assembly['stimulus_id'].values))
        _logger.info('Scoring model')
        score = self.apply_metric(model_activations, self._target_assembly)
        score = self.ceiling_normalize(score)
        return score

    def apply_metric(self, model_activations, target_assembly):
        stimulus_ids = set(self._target_assembly['stimulus_id'].values)
        model_activations = model_activations[{'presentation': [
            stimulus_id in stimulus_ids for stimulus_id in model_activations['stimulus_id'].values]}]
        return super(Blank2014SentencefROIEncoding, self).apply_metric(model_activations, target_assembly)

    def ceiling_normalize(self, score):
        raw_neuroids = apply_aggregate(lambda values: values.mean('split'), score.raw)
        if not hasattr(raw_neuroids, 'neuroid_id'):
            raw_neuroids['neuroid_id'] = 'neuroid', [".".join([str(value) for value in values]) for values in zip(*[
                raw_neuroids[coord].values for coord in ['subject_UID', 'fROI_area']])]
        score = ceil_neuroids(raw_neuroids, self.ceiling, subject_column='subject_UID')
        return score

class Blank2014fROIRDM(Blank2014fROIEncoding):
    """
    data source:
        Blank et al., Journal of Neurophysiology 2014
        https://journals.physiology.org/doi/full/10.1152/jn.00884.2013
    """

    def __init__(self, *args, **kwargs):
        super(Blank2014fROIRDM, self).__init__(*args, **kwargs)
        self._metric = RDMCrossValidated(
            comparison_coord='stimulus_id',
            crossvalidation_kwargs=dict(split_coord='stimulus_id', stratification_coord=None, splits=5,
                                        kfold=True, test_size=None))
        self._ceiler.extrapolation_dimension = 'subject_UID'
        self._cross = CartesianProduct(dividers=['subject_UID'])

    def apply_metric(self, source_assembly, target_assembly):
        # transformation sub-selection would be left with only one coordinate for the neuroid dimension
        # to work around this, we add another coord that will prevent the MultiIndex from collapsing
        if not hasattr(target_assembly, 'neuroid_id'):
            target_assembly['neuroid_id'] = 'neuroid', target_assembly['subject_UID'].values
        target_assembly = target_assembly.__class__(target_assembly)  # reconstruct to ensure proper indexing
        cross_scores = self._cross(target_assembly, apply=
        lambda cross_assembly: super(Blank2014fROIRDM, self).apply_metric(source_assembly, cross_assembly))
        score = cross_scores.median(['subject_UID'])
        score.attrs['raw'] = cross_scores
        return score

    @property
    @load_s3(key='Blank2014fROI-rdm-ceiling')
    def ceiling(self):
        return super(Blank2014fROIRDM, self).ceiling

    def ceiling_normalize(self, score):
        score = aggregate_ceiling(score.raw, ceiling=self.ceiling, subject_column='subject_UID')
        return score

    def post_process_ceilings(self, scores):
        return scores

class Blank2014fROICKA(Blank2014fROIRDM):
    """
    data source:
        Blank et al., Journal of Neurophysiology 2014
        https://journals.physiology.org/doi/full/10.1152/jn.00884.2013
    """

    def __init__(self, *args, **kwargs):
        super(Blank2014fROICKA, self).__init__(*args, **kwargs)
        self._metric = CKACrossValidated(
            comparison_coord='stimulus_id',
            crossvalidation_kwargs=dict(split_coord='stimulus_id', stratification_coord=None, splits=5,
                                        kfold=True, test_size=None))

    @property
    def ceiling(self):
        return super(Blank2014VoxelEncoding, self).ceiling

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
        self._single_metric = metric
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
        return self._single_metric(source_assembly, cross_assembly)

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
    # merging does not maintain stimulus order. the following orders again
    idx = [model_activations['stimulus_id'].values.tolist().index(stimulus_id) for stimulus_id in
           itertools.chain.from_iterable(s['stimulus_id'].values for s in activations)]
    assert len(set(idx)) == len(idx), "Found duplicate indices to order activations"
    model_activations = model_activations[{'presentation': idx}]
    return model_activations


def listen_to_v2(candidate, stimulus_set, reset_column='story', average_sentence=True):
    """
    Pass a `stimulus_set` through a model `candidate`.
    Operates on a sentence-based `stimulus_set`.
    """
    activations = []
    for story in ordered_set(stimulus_set[reset_column].values):
        story_stimuli = stimulus_set[stimulus_set[reset_column] == story]
        story_stimuli.name = f"listen_to_{stimulus_set.name}-{story}"
        # IMPORTANT: fix so that the full sentence get parsed instead of individual word
        #story_stimuli.iloc[0,0]=[story_stimuli.iloc[0,0]]
        story_activations = candidate(stimuli=story_stimuli, average_sentence=average_sentence)
        if average_sentence==False:
            # picke up the last word of each sentence
            story_activations = story_activations[{'presentation':[-1]}]
        activations.append(story_activations)
    model_activations = merge_data_arrays(activations)
    # merging does not maintain stimulus order. the following orders again
    idx = [model_activations['stimulus_id'].values.tolist().index(stimulus_id) for stimulus_id in
           itertools.chain.from_iterable(s['stimulus_id'].values for s in activations)]
    assert len(set(idx)) == len(idx), "Found duplicate indices to order activations"
    model_activations = model_activations[{'presentation': idx}]
    return model_activations

def read_words(candidate, stimulus_set, reset_column='sentence_id', copy_columns=(), average_sentence=False):
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

class PereiraLangEncoding(PereiraEncoding):
    """
    data source:
        Pereira et al., nature communications 2018
        https://www.nature.com/articles/s41467-018-03068-4?fbclid=IwAR0W7EZrnIFFO1kvANgeOEICaoDG5fhmdHipazy6n-APUJ6lMY98PkvuTyU
    """
    def _load_assembly(self,version='language'):
        assembly=xr.load_dataarray(f'{PEREIRA2018_SAMPLE}/Pereira2018.nc')
        # select only langauge atlas
        language_atlas=assembly.atlas.values=='language'
        assembly=assembly.sel(neuroid=language_atlas)
        # copy over the attributes from assembly
        # explicitly load the stimulus set
        stimulus_set_file=assembly.attrs['stimulus_set'].replace('s3:',f'{PEREIRA2018_SAMPLE}/')
        stimulus_set=pd.read_csv(stimulus_set_file)

        assembly.attrs['stimulus_set']=StimulusSet(stimulus_set)
        assembly.attrs['stimulus_set'].name='Pereira2018'
        assembly.attrs['version']='language'
        assembly=NeuroidAssembly(assembly)
        return assembly


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
        # make sure the sentence_id dimension is the same between the two
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

class _PereiraSubjectWise(_PereiraBenchmark):
    def __init__(self, **kwargs):
        super(_PereiraSubjectWise, self).__init__(**kwargs)
        self._cross = CartesianProduct(dividers=['experiment', 'atlas', 'subject'])
        self._ceiler = self.PereiraSubjectWiseExtrapolationCeiling(
            extrapolation_dimension='subject', subject_column='subject', num_bootstraps=self._ceiler.num_bootstraps)

    def _apply_cross(self, source_assembly, cross_assembly):
        # some subjects have only done one experiment which leads to nans
        cross_assembly = cross_assembly.dropna('neuroid')
        if len(cross_assembly['neuroid']) == 0:
            return Score([np.nan, np.nan], coords={'aggregation': ['center', 'error']}, dims=['aggregation'])
        return super(_PereiraSubjectWise, self)._apply_cross(
            source_assembly=source_assembly, cross_assembly=cross_assembly)

    def _average_cross_scores(self, cross_scores):
        return super(_PereiraSubjectWise, self)._average_cross_scores(cross_scores).median('subject')

    class PereiraSubjectWiseExtrapolationCeiling(_PereiraBenchmark.PereiraExtrapolationCeiling):
        def post_process(self, scores):
            return scores.mean('sub_experiment').sel(aggregation='center')

        def extrapolate(self, ceilings):
            # skip parent implementation, go straight to parent's parent
            return super(_PereiraBenchmark.PereiraExtrapolationCeiling, self).extrapolate(ceilings)

class PereiraDecoding(_PereiraSubjectWise):
    """
    data source:
        Pereira et al., nature communications 2018
        https://www.nature.com/articles/s41467-018-03068-4?fbclid=IwAR0W7EZrnIFFO1kvANgeOEICaoDG5fhmdHipazy6n-APUJ6lMY98PkvuTyU
    """

    def __init__(self, **kwargs):
        metric = CrossRegressedCorrelation(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(split_coord='stimulus_id', stratification_coord=None))
        metric = Invert(metric)
        super(PereiraDecoding, self).__init__(metric=metric, **kwargs)

class PereiraRDM(_PereiraSubjectWise):
    """
    data source:
        Pereira et al., nature communications 2018
        https://www.nature.com/articles/s41467-018-03068-4?fbclid=IwAR0W7EZrnIFFO1kvANgeOEICaoDG5fhmdHipazy6n-APUJ6lMY98PkvuTyU
    """

    def __init__(self, **kwargs):
        metric = RDMCrossValidated(
            comparison_coord='stimulus_id',
            crossvalidation_kwargs=dict(split_coord='stimulus_id', stratification_coord=None, splits=5,
                                        kfold=True, test_size=None))
        super(PereiraRDM, self).__init__(metric=metric, **kwargs)

    @property
    @load_s3(key='Pereira2018-rdm-ceiling')
    def ceiling(self):
        return super(PereiraRDM, self).ceiling

class PereiraCKA(_PereiraSubjectWise):
    """
    data source:
        Pereira et al., nature communications 2018
        https://www.nature.com/articles/s41467-018-03068-4?fbclid=IwAR0W7EZrnIFFO1kvANgeOEICaoDG5fhmdHipazy6n-APUJ6lMY98PkvuTyU
    """

    def __init__(self, **kwargs):
        metric = CKACrossValidated(
            comparison_coord='stimulus_id',
            crossvalidation_kwargs=dict(split_coord='stimulus_id', stratification_coord=None, splits=5,
                                        kfold=True, test_size=None))
        super(PereiraCKA, self).__init__(metric=metric, **kwargs)

    @property
    def ceiling(self):
        return super(_PereiraSubjectWise, self).ceiling

class _ANNSet1fMRIBenchmark(Benchmark):
    """
    data source:
        Pereira et al., nature communications 2018
        https://www.nature.com/articles/s41467-018-03068-4
    """

    def __init__(self, identifier, metric,version='base'):
        self._identifier = identifier
        assembly = self._load_assembly(version=version)
        self._target_assembly = assembly
        self._metric = metric
        self._ceiler = FewSubjectExtrapolation(subject_column='subject',extrapolation_dimension='neuroid',post_process=None,num_subsamples=150,num_bootstraps=200)
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
            #sentence_stimuli = StimulusSet({'sentence': part_stimuli.values[0],
            #                                reset_column: list(set(part_stimuli[reset_column].values))})
            sentence_stimuli = StimulusSet({'sentence': part_stimuli.sentence,
                                            reset_column: list(set(part_stimuli[reset_column].values))})
            #self._target_assembly.attrs['stimuli_group']
            #sentence_stimuli.name = f"{self._target_assembly.identifier}-read-{reset_id}"
            sentence_stimuli.name = f"{self._target_assembly.attrs['stimuli_group']}-read-{reset_id}"
            sentence_activations = candidate(stimuli=sentence_stimuli, average_sentence=average_sentence)
            num_words=len(str(part_stimuli.sentence.values[0]).split(' '))
            assert(sentence_activations.shape[0]==num_words)
            sentence_activations = sentence_activations.isel(presentation=slice(-1,None))
            #sentence_activations=sentence_activations[-1, :]
            # for column in copy_columns:
            #    sentence_activations[column] = ('presentation', part_stimuli[column])
            activations.append(sentence_activations)

        # model_activations = merge_data_arrays(activations)
        model_activations = xr.concat(activations, dim='presentation')
        # merging does not maintain stimulus order. the following orders again
        idx = [model_activations['stimulus_id'].values.tolist().index(stimulus_id) for stimulus_id in
               [int(s['stimulus_id'].values) for s in activations]]
        assert len(set(idx)) == len(idx), "Found duplicate indices to order activations"
        model_activations = model_activations[{'presentation': idx}]

        return model_activations

    def apply_metric(self, source_assembly, target_assembly):
        """ for ceiling compute """
        cross_scores = self._cross(target_assembly, apply=
        lambda cross_assembly: self._apply_cross(source_assembly, cross_assembly))
        score = self._average_cross_scores(cross_scores)
        return score

    def _average_cross_scores(self, cross_scores):
        return cross_scores.mean(['experiment', 'atlas'])


    def _load_assembly(self,version):
        if version=='base':
            assembly = pd.read_pickle(f'{ANNfMRI_PARENT}/ANNSet1_fMRI-train-language_top_90.pkl')

        elif version=='wordForm':
            assembly = pd.read_pickle(f'{ANNfMRI_PARENT}/ANNSet1_fMRI.train.language_top_90_wordForm.pkl')
        elif version=='best':
            assembly = pd.read_pickle(f'{ANNfMRI_PARENT}/ANNSet1_fMRI-best-language_top_90_V2.pkl')

        # save a.sentence and b.sentence  as csv file with 2 columns a.sentence, b.sentence
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
        assembly.attrs['stimuli_group'] = f'ANNSet1_fMRI'
        # drop the period in the end of sentences if they exist in the end


        sentences = assembly['stimulus'].str.replace(r'\.$', '', regex=True)
        stimulus_set = StimulusSet({'sentence': sentences.values,
                                    'stimulus_num': assembly['stimulus_num'].values,
                                    'stimulus_id': assembly['stimulus_id'].values,
                                    'stim_name': assembly['stim_name'].values,
                                    'stumulus': sentences.values,
                                    'sentence_id': assembly['stimulus_id'].values})

        stimulus_set.name = f"{assembly.attrs['stimuli_group']}"
        assembly.attrs['stimulus_set'] = stimulus_set

        return assembly
    def __call__(self, candidate):
        #stimulus_set = self._target_assembly['stimulus']
        stimulus_set = self._target_assembly.attrs['stimulus_set']
        #stimulus_set = stimulus_set.assign_coords({'sentence_id': ('presentation', stimulus_set.stimulus_id.values)})
        #stimulus_set.word_id
        model_activations = self._read_words(candidate, stimulus_set, copy_columns=['word_id'],reset_column='stimulus_id')
        #model_activations = listen_to_v2(candidate, stimulus_set,reset_column='stimulus_id',average_sentence=False)
        #model_activations = read_words(candidate, stimulus_set, copy_columns=['word_id'],reset_column='stimulus_id')
        assert set(model_activations['stimulus_id'].values) == set(self._target_assembly['stimulus_id'].values)

        _logger.info('Scoring across experiments & atlases')
        cross_scores = self._cross(self._target_assembly,
                                   apply=lambda cross_assembly: self._apply_cross(model_activations, cross_assembly))
        raw_scores = cross_scores.raw
        raw_neuroids = apply_aggregate(lambda values: values.mean('split'), raw_scores)
        language_neuroids = raw_neuroids.sel(atlas='language', _apply_raw=False)
        score = self._aggregate_ceiling(language_neuroids, ceiling=self.ceiling, subject_column='subject')
        return score

    def _apply_cross(self, source_assembly, cross_assembly):
        cross_assembly = cross_assembly.dropna('neuroid')  # some subjects have only done one experiment
        source_assembly = source_assembly.dropna('neuroid')  # only relevant when running audio-visual self as "model"
        assert len(cross_assembly['presentation']) in [200]
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

class ANNSet1fMRIEncoding(_ANNSet1fMRIBenchmark):
    """
    data source:
    """

    def __init__(self, **kwargs):
        metric = CrossRegressedCorrelation(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord='stimulus_id', stratification_coord=None))
        super(ANNSet1fMRIEncoding, self).__init__(metric=metric, **kwargs)


    @property
    def ceiling(self):
        #ceiling_val=pd.read_pickle(f'{ANNfMRI_PARENT}/ANNSet1_fMRI-train-language_top_90-linear_ceiling.pkl')
        ceiling_val=super(ANNSet1fMRIEncoding, self).ceiling
        return ceiling_val

class ANNSet1fMRIEncoding_V2(_ANNSet1fMRIBenchmark):
    """
    data source:
    """
    def __init__(self, **kwargs):

        metric = CrossRegressedCorrelation(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord='stimulus_id', stratification_coord=None))

        super(ANNSet1fMRIEncoding_V2, self).__init__(metric=metric,version='wordForm', **kwargs)

    # def _load_assembly(self):
    #     # read UD data and replace stimuli
    #     assembly = pd.read_pickle(f'{ANNfMRI_PARENT}/ANNSet1_fMRI-train-language_top_90.pkl')
    #     UD_data = pd.read_pickle(f'{ANNfMRI_PARENT}/ud_sentencez_data_token_filter_v3_brainscore.pkl')
    #     sentence_texts=[]
    #     for stim_id,stim in UD_data.groupby('stimulus_id'):
    #         sentence_texts.append(np.unique(stim.text.values)[0])
    #     sentence_index=[sentence_texts.index(x) for x in assembly.stimulus.values]
    #     assert(len(sentence_index)==200)
    #     selected_stim=[]
    #     for sent_id in sentence_index:
    #         location=(UD_data.stimulus_id==sent_id).values
    #         selected_stim.append(UD_data.sel(index=location))
    #
    #     assert all([np.unique(x.text)[0]==assembly.stimulus.values[idx] for idx, x in enumerate(selected_stim)])
    #     stimulus_form=[' '.join(x.word_FORM.values) for x in selected_stim]
    #
    #     new_assembly = NeuroidAssembly(assembly.values, coords={
    #         'experiment': ('presentation', assembly.experiment.values),
    #         'stimulus_num': ('presentation', assembly.stimulus_num.values),
    #         'stimulus_id': ('presentation', assembly.stimulus_id.values),
    #         'sentence': ('presentation', stimulus_form),
    #         'stimulus': ('presentation', stimulus_form),
    #         'list_id': ('presentation', assembly.list_id.values),
    #         'stim_type': ('presentation', assembly.stim_type.values),
    #         'stim_name': ('presentation', assembly.stim_name.values),
    #         'Trial_id': ('presentation', assembly.Trial_id.values),
    #         'TR_onset': ('presentation', assembly.TR_onset.values),
    #         'TR_recorded': ('presentation', assembly.TR_recorded.values),
    #         'TR_duration': ('presentation', assembly.TR_duration.values),
    #         'subject': ('neuroid', assembly.subject.values),
    #         'neuroid_id': ('neuroid', assembly.neuroid_id.values),
    #         'voxel_num': ('neuroid', assembly.voxel_num.values),
    #         'repetition_corr_ratio': ('neuroid', assembly.repetition_corr_ratio.values),
    #         'repetition_corr': ('neuroid', assembly.repetition_corr.values),
    #         'roi': ('neuroid', assembly.roi.values),
    #         'atlas': ('neuroid',assembly.atlas.values)
    #     }, dims=['presentation', 'neuroid'])
    #     new_assembly = new_assembly.sortby('stimulus_id')
    #
    #     new_assembly.attrs['identifier']=assembly.identifier+'_wordForm'
    #     name=new_assembly.identifier.replace('.','-')
    #     with open(Path(f'{ANNfMRI_PARENT}/{name}.pkl').__str__(),'wb') as f:
    #         pickle.dump(new_assembly,f)
    #
    #     return new_assembly
    @property
    def ceiling(self):
        ceiling_val = pd.read_pickle(f'{ANNfMRI_PARENT}/ANNSet1_fMRI-train-language_top_90-linear_ceiling.pkl')
        return ceiling_val


class ANNSet1fMRIBestEncoding(_ANNSet1fMRIBenchmark):
    """
    data source:
    """

    def __init__(self, **kwargs):
        metric = CrossRegressedCorrelation(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord='stimulus_id', stratification_coord=None))
        super(ANNSet1fMRIBestEncoding, self).__init__(metric=metric, version='best', ** kwargs)



    @property
    def ceiling(self):
        #ceiling_val=pd.read_pickle(f'{ANNfMRI_PARENT}/ANNSet1_fMRI-train-language_top_90-linear_ceiling.pkl')
        ceiling_val=super(ANNSet1fMRIBestEncoding, self).ceiling
        return ceiling_val

class ANNSet1fMRIBestV3Encoding(_ANNSet1fMRIBenchmark):
    """
    data source:
    """

    def __init__(self, **kwargs):
        metric = CrossRegressedCorrelation(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord='stimulus_id', stratification_coord=None))
        super(ANNSet1fMRIBestV3Encoding, self).__init__(metric=metric, version='best', ** kwargs)

    def _load_assembly(self, version):
        if version == 'base':
            # raise ValueError('No base version for ANNSet1fMRIBestV3Encoding')
            NotImplementedError('No base version for ANNSet1fMRIBestV3Encoding')

            #assembly = pd.read_pickle(f'{ANNfMRI_PARENT}/ANNSet1_fMRI-train-language_top_90.pkl')

        elif version == 'wordForm':
            #assembly = pd.read_pickle(f'{ANNfMRI_PARENT}/ANNSet1_fMRI.train.language_top_90_wordForm.pkl')
            NotImplementedError('No base version for ANNSet1fMRIBestV3Encoding')
        elif version == 'best':
            assembly = pd.read_pickle(f'{ANNfMRI_PARENT}/ANNsent_best_subs_8_language_top_90_V3.pkl')

        # save a.sentence and b.sentence  as csv file with 2 columns a.sentence, b.sentence
        vox_reliability = {'language': (True, .95), 'auditory': (False, .95), 'visual': (False, .95)}
        vox_corr = {'language': (True, .1), 'auditory': (False, .1), 'visual': (False, .1)}
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
        assembly.attrs['stimuli_group'] = f'ANNSet1_fMRI'
        # drop the period in the end of sentences if they exist in the end

        sentences = assembly['stimulus'].str.replace(r'\.$', '', regex=True)
        stimulus_set = StimulusSet({'sentence': sentences.values,
                                    'stimulus_num': assembly['stimulus_num'].values,
                                    'stimulus_id': assembly['stimulus_id'].values,
                                    'stim_name': assembly['stim_name'].values,
                                    'stumulus': sentences.values,
                                    'sentence_id': assembly['stimulus_id'].values})

        stimulus_set.name = f"{assembly.attrs['stimuli_group']}"
        assembly.attrs['stimulus_set'] = stimulus_set

        return assembly

    @property
    def ceiling(self):
        #ceiling_val=pd.read_pickle(f'{ANNfMRI_PARENT}/ANNSet1_fMRI-train-language_top_90-linear_ceiling.pkl')
        ceiling_val=super(ANNSet1fMRIBestV3Encoding, self).ceiling
        return ceiling_val

class _ANNSet1fMRISentenceBenchmark(Benchmark):
    """
    data source:

    """

    def __init__(self, identifier, metric,version='base'):
        self._identifier = identifier
        assembly = self._load_assembly(version=version)
        self._target_assembly = assembly
        self._single_metric = metric
        # self._ceiler = self.PereiraExtrapolationCeiling(subject_column='subject', num_bootstraps=100)
        self._cross = CartesianProduct(dividers=['experiment', 'atlas'])

    @property
    def identifier(self):
        return self._identifier

    def _listen_to(self,candidate, stimulus_set, reset_column='stimulus_id', average_sentence=True):
        """
        Pass a `stimulus_set` through a model `candidate`.
        Operates on a sentence-based `stimulus_set`.
        """
        activations = []
        for story in ordered_set(stimulus_set[reset_column].values):
            story_stimuli = stimulus_set[stimulus_set[reset_column] == story]
            stim_number=int(story_stimuli[reset_column].values)
            story_stimuli.name = f"{self._target_assembly.identifier}-sentence-ave-{average_sentence}-{stim_number}"
            story_activations = candidate(stimuli=story_stimuli, average_sentence=average_sentence)
            activations.append(story_activations)
        model_activations = merge_data_arrays(activations)
        # merging does not maintain stimulus order. the following orders again
        idx = [model_activations['stimulus_id'].values.tolist().index(stimulus_id) for stimulus_id in
               itertools.chain.from_iterable(s['stimulus_id'].values for s in activations)]
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
    def _load_assembly(self,version):
        if version=='base':
            assembly = pd.read_pickle(f'{ANNfMRI_PARENT}/ANNSet1_fMRI-train-language_top_90.pkl')
        elif version=='wordForm':
            assembly = pd.read_pickle(f'{ANNfMRI_PARENT}/ANNSet1_fMRI.train.language_top_90_wordForm.pkl')
        return assembly


    def __call__(self, candidate):
        stimulus_set = self._target_assembly['stimulus']

        stimulus_set = stimulus_set.assign_coords({'sentence_id': ('presentation', stimulus_set.stimulus_id.values)})
        model_activations = self._listen_to(candidate, stimulus_set)
        assert set(model_activations['stimulus_id'].values) == set(self._target_assembly['stimulus_id'].values)

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
        score = self._aggregate_ceiling(language_neuroids, ceiling=self.ceiling, subject_column='subject')
        return score

    def _apply_cross(self, source_assembly, cross_assembly):
        cross_assembly = cross_assembly.dropna('neuroid')  # some subjects have only done one experiment
        source_assembly = source_assembly.dropna('neuroid')  # only relevant when running audio-visual self as "model"
        assert len(cross_assembly['presentation']) in [200]
        assert not np.isnan(cross_assembly).any()
        source_assembly = source_assembly[{'presentation': [stimulus_id in cross_assembly['stimulus_id'].values
                                                            for stimulus_id in source_assembly['stimulus_id'].values]}]
        return self._single_metric(source_assembly, cross_assembly)

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

class ANNSet1fMRISentenceEncoding(_ANNSet1fMRISentenceBenchmark):
    """
    data source:
    """

    def __init__(self, **kwargs):
        metric = CrossRegressedCorrelation(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord='stimulus_id', stratification_coord=None))
        super(ANNSet1fMRISentenceEncoding, self).__init__(metric=metric, **kwargs)


    @property
    def ceiling(self):
        ceiling_val=pd.read_pickle(f'{ANNfMRI_PARENT}/ANNSet1_fMRI-train-language_top_90-linear_ceiling.pkl')
        return ceiling_val

class ANNSet1fMRISentenceEncoding_V2(_ANNSet1fMRISentenceBenchmark):
    """
    data source:
    """
    def __init__(self, **kwargs):
        metric = CrossRegressedCorrelation(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord='stimulus_id', stratification_coord=None))
        super(ANNSet1fMRISentenceEncoding_V2, self).__init__(metric=metric, version='wordForm', **kwargs)
    @property
    def ceiling(self):
        ceiling_val = pd.read_pickle(f'{ANNfMRI_PARENT}/ANNSet1_fMRI-train-language_top_90-linear_ceiling.pkl')
        return ceiling_val

class _DsParametricfMRIBenchmark(Benchmark):
    """
    data source:
        Pereira et al., nature communications 2018
        https://www.nature.com/articles/s41467-018-03068-4
    """

    def __init__(self, identifier, metric,version='DsParametricfMRI_subs_12_language',group='max',threshold=90):
        self._identifier = identifier
        assembly = self._load_assembly(version=version,group=group,threshold=threshold)
        self._target_assembly = assembly
        self._single_metric = metric
        # self._ceiler = self.PereiraExtrapolationCeiling(subject_column='subject', num_bootstraps=100)
        self._cross = CartesianProduct(dividers=['atlas'])

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
            # sentence_stimuli = StimulusSet({'sentence': part_stimuli.values[0],
            #                                reset_column: list(set(part_stimuli[reset_column].values))})
            sentence_stimuli = StimulusSet({'sentence': part_stimuli.sentence,
                                            reset_column: list(set(part_stimuli[reset_column].values))})
            # self._target_assembly.attrs['stimuli_group']
            # sentence_stimuli.name = f"{self._target_assembly.identifier}-read-{reset_id}"
            sentence_stimuli.name = f"{stimulus_set.name}-read-{reset_id}"
            sentence_activations = candidate(stimuli=sentence_stimuli, average_sentence=average_sentence)
            num_words = len(str(part_stimuli.sentence.values[0]).split(' '))
            assert (sentence_activations.shape[0] == num_words)
            #sentence_activations.sel(word=sentence_activations.word.values[-1],drop=False)
            sentence_activations = sentence_activations.isel(presentation=slice(-1,None))
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
    def _load_assembly(self,version='DsParametricfMRI_subs_12_language',group='max',threshold=90):
        assembly = pd.read_pickle(f'{DsParametricfMRI_PARENT}/{version}_top_{threshold}_reliability_random_analyzed_aug2024.pkl')
        # select stimuli that have the stim_group= version
        vox_reliability = {'language': (False, .95), 'auditory': (False, .95), 'visual': (False, .95)}
        vox_corr = {'language': (False, .1), 'auditory': (False, .1), 'visual': (False, .1)}
        if group=='all':
            pass
        else:
            assembly = assembly[assembly['stim_group'] == group]

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
        assembly.attrs['stimuli_group'] = 'DsParametricfMRI_' + group #+ f'_thr_{threshold}'
        if group=='all':
            # assign a new coordinate called stimulus_id with values from stim_id
            assembly=assembly.assign_coords({'stimulus_id': ('presentation', assembly.stim_id.values)})
        else:
            assembly=assembly.assign_coords({'stimulus_id': ('presentation', assembly.stim_group_id.values)})

        sentences = assembly['stimulus'].str.replace(r'\.$', '', regex=True)
        stimulus_set = StimulusSet({'sentence': sentences.values,
                                    'stimulus_num': assembly['stimulus_num'].values,
                                    'stimulus_id': assembly['stimulus_id'].values,
                                    'stim_name': assembly['stim_name'].values,
                                    'stim_group':assembly['stim_group'].values,
                                    'stim_group_id': assembly['stim_group_id'].values,
                                    'stumulus': sentences.values,
                                    'sentence_id': assembly['stimulus_id'].values})
        stimulus_set.name = assembly.attrs['stimuli_group']
        assembly.attrs['stimulus_set'] = stimulus_set
        return assembly

    def __call__(self, candidate):
        stimulus_set = self._target_assembly.attrs['stimulus_set']
        #stimulus_set = stimulus_set.assign_coords({'sentence_id': ('presentation', stimulus_set.stimulus_id.values)})

        model_activations = self._read_words(candidate, stimulus_set, copy_columns=['word_id'],reset_column='stimulus_id')
        assert set(model_activations['stimulus_id'].values) == set(self._target_assembly['stimulus_id'].values)
        # add a new cooordinate called stimulus_id
        #model_activations = model_activations.assign_coords({'stimulus_id': ('presentation', model_activations.stim_group_id.values)})
        # for target assembly go through all coordsinate and assign them again
        #coords = {coord: coord_value for coord, coord_value in self._target_assembly.coords.items() }
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
        #score = self._aggregate_no_ceiling(language_neuroids, ceiling=[], subject_column='subject')
        score = self._aggregate_no_ceiling(language_neuroids, subject_column='subject')
        return score

    def _apply_cross(self, source_assembly, cross_assembly):
        cross_assembly = cross_assembly.dropna('neuroid')  # some subjects have only done one experiment
        source_assembly = source_assembly.dropna('neuroid')  # only relevant when running audio-visual self as "model"
        #assert len(cross_assembly['presentation']) in [80]
        assert not np.isnan(cross_assembly).any()
        source_assembly = source_assembly[{'presentation': [stimulus_id in cross_assembly['stimulus_id'].values
                                                            for stimulus_id in source_assembly['stimulus_id'].values]}]
        return self._single_metric(source_assembly, cross_assembly)

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

# class _DsParametricSujectwise(_DsParametricfMRIBenchmark):
#     def __init__(self, **kwargs):
#         super(_DsParametricSujectwise, self).__init__(**kwargs)
#         self._cross = CartesianProduct(dividers=['subject','atlas'])
#         self._ceiler = None
#     def _apply_cross(self, source_assembly, cross_assembly):
#         # some subjects have only done one experiment which leads to nans
#         cross_assembly = cross_assembly.dropna('neuroid')
#         return super(_DsParametricSujectwise, self)._apply_cross(
#             source_assembly=source_assembly, cross_assembly=cross_assembly)
#
#     def _average_cross_scores(self, cross_scores):
#         return super(_DsParametricSujectwise, self)._average_cross_scores(cross_scores).median('subject')




class DsParametricfMRIEncoding(_DsParametricfMRIBenchmark):
    """
    data source:
    """

    def __init__(self, **kwargs):
        metric = CrossRegressedCorrelation(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord='stimulus_id', stratification_coord=None))
        super(DsParametricfMRIEncoding, self).__init__(metric=metric, **kwargs)


    # @property
    # def ceiling(self):
    #     ceiling_val=pd.read_pickle(f'{ANNfMRI_PARENT}/ANNSet1_fMRI-train-language_top_90-linear_ceiling.pkl')
    #     return ceiling_val

class DsParametricfMRISharedMaxEncoding(DsParametricfMRIEncoding):
    def _load_assembly(self,version='DsParametricfMRI_subs_7_language',group='max',threshold=90):
        return super()._load_assembly(version='DsParametricfMRI_subs_7_language',group='max',threshold=90)

class DsParametricfMRIFullMaxEncoding(DsParametricfMRIEncoding):
    def _load_assembly(self,version='DsParametricfMRI_subs_12_language',group='max',threshold=90):
        return super()._load_assembly(version='DsParametricfMRI_subs_12_language',group='max',threshold=90)


class DsParametricfMRIShared70MaxEncoding(DsParametricfMRIEncoding):
    def _load_assembly(self,version='DsParametricfMRI_subs_7_language',group='max',threshold=70):
        return super()._load_assembly(version='DsParametricfMRI_subs_7_language',group='max',threshold=70)

class DsParametricfMRISharedMinEncoding(DsParametricfMRIEncoding):
    def _load_assembly(self, version='DsParametricfMRI_subs_7_language', group='min', threshold=90):
        return super()._load_assembly(version='DsParametricfMRI_subs_7_language', group='min', threshold=90)

class DsParametricfMRIFullMinEncoding(DsParametricfMRIEncoding):
    def _load_assembly(self, version='DsParametricfMRI_subs_12_language', group='min', threshold=90):
        return super()._load_assembly(version='DsParametricfMRI_subs_12_language', group='min', threshold=90)

class DsParametricfMRIShared70MinEncoding(DsParametricfMRIEncoding):
    def _load_assembly(self,version='DsParametricfMRI_subs_7_language',group='min',threshold=70):
        return super()._load_assembly(version='DsParametricfMRI_subs_7_language',group='min',threshold=70)

class DsParametricfMRIShared70RandEncoding(DsParametricfMRIEncoding):
    def _load_assembly(self,version='DsParametricfMRI_subs_7_language',group='random',threshold=70):
        return super()._load_assembly(version='DsParametricfMRI_subs_7_language',group='random',threshold=70)


class DsParametricfMRISharedRandEncoding(DsParametricfMRIEncoding):
    def _load_assembly(self, version='DsParametricfMRI_subs_7_language', group='random', threshold=90):
        return super()._load_assembly(version='DsParametricfMRI_subs_7_language', group='random', threshold=90)

class DsParametricfMRIFullRandEncoding(DsParametricfMRIEncoding):
    def _load_assembly(self, version='DsParametricfMRI_subs_12_language', group='random', threshold=90):
        return super()._load_assembly(version='DsParametricfMRI_subs_12_language', group='random', threshold=90)
class DsParametricfMRISharedAllEncoding(DsParametricfMRIEncoding):
    def _load_assembly(self, version='DsParametricfMRI_subs_7_language', group='all', threshold=90):
        return super()._load_assembly(version='DsParametricfMRI_subs_7_language', group='all', threshold=90)

class DsParametricfMRIRidgeEncoding(_DsParametricfMRIBenchmark):
    """
    data source:
    """

    def __init__(self, **kwargs):
        metric = CrossRegressedCorrelation(
            regression=rgcv_linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord='stimulus_id', stratification_coord=None))
        super(DsParametricfMRIRidgeEncoding, self).__init__(metric=metric, **kwargs)


    # @property
    # def ceiling(self):
    #     ceiling_val=pd.read_pickle(f'{ANNfMRI_PARENT}/ANNSet1_fMRI-train-language_top_90-linear_ceiling.pkl')
    #     return ceiling_val

class DsParametricfMRISharedMaxRidgeEncoding(DsParametricfMRIRidgeEncoding):
    def _load_assembly(self, version='DsParametricfMRI_subs_7_language', group='max', threshold=90):
        return super()._load_assembly(version='DsParametricfMRI_subs_7_language', group='max', threshold=90)

class DsParametricfMRISharedMinRidgeEncoding(DsParametricfMRIRidgeEncoding):
    def _load_assembly(self, version='DsParametricfMRI_subs_7_language', group='min', threshold=90):
        return super()._load_assembly(version='DsParametricfMRI_subs_7_language', group='min', threshold=90)
class DsParametricfMRISharedRandRidgeEncoding(DsParametricfMRIRidgeEncoding):
    def _load_assembly(self, version='DsParametricfMRI_subs_7_language', group='random', threshold=90):
        return super()._load_assembly(version='DsParametricfMRI_subs_7_language', group='random', threshold=90)


class DsParametricfMRIMaxV2RidgeEncoding(DsParametricfMRIRidgeEncoding):
    def _load_assembly(self,version='max',threshold=80):
        return super()._load_assembly(version='max',threshold=80)

class DsParametricfMRIMaxV3RidgeEncoding(DsParametricfMRIRidgeEncoding):
    def _load_assembly(self,version='max',threshold=70):
        return super()._load_assembly(version='max',threshold=70)

class DsParametricfMRIMinV1RidgeEncoding(DsParametricfMRIRidgeEncoding):
    def _load_assembly(self,version='min',threshold=90):
        return super()._load_assembly(version='min',threshold=90)

class DsParametricfMRIMinV2RidgeEncoding(DsParametricfMRIRidgeEncoding):
    def _load_assembly(self,version='min',threshold=80):
        return super()._load_assembly(version='min',threshold=80)

class DsParametricfMRIMinV3RidgeEncoding(DsParametricfMRIRidgeEncoding):
    def _load_assembly(self,version='min',threshold=70):
        return super()._load_assembly(version='min',threshold=70)

class DsParametricfMRIRandV1RidgeEncoding(DsParametricfMRIRidgeEncoding):
    def _load_assembly(self,version='random',threshold=90):
        return super()._load_assembly(version='random',threshold=90)

class DsParametricfMRIRandV2RidgeEncoding(DsParametricfMRIRidgeEncoding):
    def _load_assembly(self,version='random',threshold=80):
        return super()._load_assembly(version='random',threshold=80)

class DsParametricfMRIRandV3RidgeEncoding(DsParametricfMRIRidgeEncoding):
    def _load_assembly(self,version='random',threshold=70):
        return super()._load_assembly(version='random',threshold=70)

class DsParametricfMRIMinRidgeEncoding(DsParametricfMRIRidgeEncoding):
    def _load_assembly(self,version='min',threshold=90):
        return super()._load_assembly(version='min',threshold=90)
class DsParametricfMRIRandRidgeEncoding(DsParametricfMRIRidgeEncoding):
    def _load_assembly(self,version='random',threshold=90):
        return super()._load_assembly(version='random',threshold=90)

class DsParametricSinglefMRIEncoding(_DsParametricfMRIBenchmark):
    """
    data source:
    """

    def __init__(self, **kwargs):
        metric = CrossRegressedCorrelation(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord='stimulus_id', stratification_coord=None))
        super(DsParametricSinglefMRIEncoding, self).__init__(metric=metric, **kwargs)

    def _load_assembly(self,version='DsParametricfMRI_rsa_subs_12_language',group='max',threshold=90,repetition=0):
        assembly = pd.read_pickle(f'{DsParametricfMRI_PARENT}/{version}_top_{threshold}_reliability_random_analyzed_aug2024.pkl')
        # select first repetion
        assembly = assembly[{'repeat':repetition}]
        # select stimuli that have the stim_group= version
        vox_reliability = {'language': (False, .95), 'auditory': (False, .95), 'visual': (False, .95)}
        vox_corr = {'language': (True, .1), 'auditory': (False, .1), 'visual': (False, .1)}
        if group=='all':
            pass
        else:
            assembly = assembly[assembly['stim_group'] == group]

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
        assembly.attrs['stimuli_group'] = 'DsParametricfMRI_' + group #+ f'_thr_{threshold}'
        if group=='all':
            # assign a new coordinate called stimulus_id with values from stim_id
            assembly=assembly.assign_coords({'stimulus_id': ('presentation', assembly.stim_id.values)})
        else:
            assembly=assembly.assign_coords({'stimulus_id': ('presentation', assembly.stim_group_id.values)})

        sentences = assembly['stimulus'].str.replace(r'\.$', '', regex=True)
        stimulus_set = StimulusSet({'sentence': sentences.values,
                                    'stimulus_num': assembly['stimulus_num'].values,
                                    'stimulus_id': assembly['stimulus_id'].values,
                                    'stim_name': assembly['stim_name'].values,
                                    'stim_group':assembly['stim_group'].values,
                                    'stim_group_id': assembly['stim_group_id'].values,
                                    'stumulus': sentences.values,
                                    'sentence_id': assembly['stimulus_id'].values})
        stimulus_set.name = assembly.attrs['stimuli_group']
        assembly.attrs['stimulus_set'] = stimulus_set
        return assembly

class DsParametricfMRIFirstMaxEncoding(DsParametricSinglefMRIEncoding):
    def _load_assembly(self, version='DsParametricfMRI_rsa_subs_12_language', group='max', threshold=90,repetition=0):
        return super()._load_assembly(version='DsParametricfMRI_rsa_subs_12_language', group='max', threshold=90,repetition=0)

class DsParametricfMRIFirstMinEncoding(DsParametricSinglefMRIEncoding):
    def _load_assembly(self, version='DsParametricfMRI_rsa_subs_12_language', group='min', threshold=90,repetition=0):
        return super()._load_assembly(version='DsParametricfMRI_rsa_subs_12_language', group='min', threshold=90,repetition=0)
class DsParametricfMRIFirstRandEncoding(DsParametricSinglefMRIEncoding):
    def _load_assembly(self, version='DsParametricfMRI_rsa_subs_12_language', group='random', threshold=90,repetition=0):
        return super()._load_assembly(version='DsParametricfMRI_rsa_subs_12_language', group='random', threshold=90,repetition=0)

class DsParametricfMRISecondMaxEncoding(DsParametricSinglefMRIEncoding):
    def _load_assembly(self, version='DsParametricfMRI_rsa_subs_12_language', group='max', threshold=90,repetition=1):
        return super()._load_assembly(version='DsParametricfMRI_rsa_subs_12_language', group='max', threshold=90,repetition=1)

class DsParametricfMRISecondMinEncoding(DsParametricSinglefMRIEncoding):
    def _load_assembly(self, version='DsParametricfMRI_rsa_subs_12_language', group='min', threshold=90,repetition=1):
        return super()._load_assembly(version='DsParametricfMRI_rsa_subs_12_language', group='min', threshold=90,repetition=1)
class DsParametricfMRISecondRandEncoding(DsParametricSinglefMRIEncoding):
    def _load_assembly(self, version='DsParametricfMRI_rsa_subs_12_language', group='random', threshold=90,repetition=1):
        return super()._load_assembly(version='DsParametricfMRI_rsa_subs_12_language', group='random', threshold=90,repetition=1)

# do instead of .1, top 10% of relaiblity in voxel for each subject
class DsParametricfMRISingleReliableEncoding(DsParametricSinglefMRIEncoding):
    def _load_assembly(self, version='DsParametricfMRI_rsa_subs_12_language', group='max', threshold=90, repetition=0):
        assembly = pd.read_pickle(f'{DsParametricfMRI_PARENT}/{version}_top_{threshold}_reliability_random_analyzed_aug2024.pkl')
        # select first repetion
        assembly = assembly[{'repeat': repetition}]
        # select stimuli that have the stim_group= version
        if group == 'all':
            pass
        else:
            assembly = assembly[assembly['stim_group'] == group]
        # group assembly by subject
        grp_q=[]
        for grp_id, grp in assembly.groupby('subject'):
            ind_q=(grp['repetition_corr']>grp['repetition_corr'].quantile(.9)).values
            grp_q.append(grp[:,ind_q])
        assembly = xr.concat(grp_q,dim='neuroid')
        assembly.attrs['stimuli_group'] = 'DsParametricfMRI_' + group  # + f'_thr_{threshold}'
        if group == 'all':
            # assign a new coordinate called stimulus_id with values from stim_id
            assembly = assembly.assign_coords({'stimulus_id': ('presentation', assembly.stim_id.values)})
        else:
            assembly = assembly.assign_coords({'stimulus_id': ('presentation', assembly.stim_group_id.values)})

        sentences = assembly['stimulus'].str.replace(r'\.$', '', regex=True)
        stimulus_set = StimulusSet({'sentence': sentences.values,
                                    'stimulus_num': assembly['stimulus_num'].values,
                                    'stimulus_id': assembly['stimulus_id'].values,
                                    'stim_name': assembly['stim_name'].values,
                                    'stim_group': assembly['stim_group'].values,
                                    'stim_group_id': assembly['stim_group_id'].values,
                                    'stumulus': sentences.values,
                                    'sentence_id': assembly['stimulus_id'].values})
        stimulus_set.name = assembly.attrs['stimuli_group']
        assembly.attrs['stimulus_set'] = stimulus_set
        return assembly


class DsParametricfMRIFirstReliableMaxEncoding(DsParametricfMRISingleReliableEncoding):
    def _load_assembly(self, version='DsParametricfMRI_rsa_subs_12_language', group='max', threshold=90,repetition=0):
        return super()._load_assembly(version='DsParametricfMRI_rsa_subs_12_language', group='max', threshold=90,repetition=0)

class DsParametricfMRIFirstReliableMinEncoding(DsParametricfMRISingleReliableEncoding):
    def _load_assembly(self, version='DsParametricfMRI_rsa_subs_12_language', group='min', threshold=90,repetition=0):
        return super()._load_assembly(version='DsParametricfMRI_rsa_subs_12_language', group='min', threshold=90,repetition=0)
class DsParametricfMRIFirstReliableRandEncoding(DsParametricfMRISingleReliableEncoding):
    def _load_assembly(self, version='DsParametricfMRI_rsa_subs_12_language', group='random', threshold=90,repetition=0):
        return super()._load_assembly(version='DsParametricfMRI_rsa_subs_12_language', group='random', threshold=90,repetition=0)

class DsParametricfMRISecondReliableMaxEncoding(DsParametricfMRISingleReliableEncoding):
    def _load_assembly(self, version='DsParametricfMRI_rsa_subs_12_language', group='max', threshold=90,repetition=1):
        return super()._load_assembly(version='DsParametricfMRI_rsa_subs_12_language', group='max', threshold=90,repetition=1)

class DsParametricfMRISecondReliableMinEncoding(DsParametricfMRISingleReliableEncoding):
    def _load_assembly(self, version='DsParametricfMRI_rsa_subs_12_language', group='min', threshold=90,repetition=1):
        return super()._load_assembly(version='DsParametricfMRI_rsa_subs_12_language', group='min', threshold=90,repetition=1)
class DsParametricfMRISecondReliableRandEncoding(DsParametricfMRISingleReliableEncoding):
    def _load_assembly(self, version='DsParametricfMRI_rsa_subs_12_language', group='random', threshold=90,repetition=1):
        return super()._load_assembly(version='DsParametricfMRI_rsa_subs_12_language', group='random', threshold=90,repetition=1)




class DsParametricSinglefMRIRidgeEncoding(_DsParametricfMRIBenchmark):
    """
    data source:
    """

    def __init__(self, **kwargs):
        metric = CrossRegressedCorrelation(
            regression=rgcv_linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord='stimulus_id', stratification_coord=None))
        super().__init__(metric=metric, **kwargs)

    def _load_assembly(self,version='DsParametricfMRI_rsa_subs_12_language',group='max',threshold=90,repetition=0):
        data_file_name=f'{DsParametricfMRI_PARENT}/{version}_top_{threshold}_reliability_random_analyzed_aug2024.pkl'
        print(f'reading {data_file_name}\n')
        assembly = pd.read_pickle(f'{DsParametricfMRI_PARENT}/{version}_top_{threshold}_reliability_random_analyzed_aug2024.pkl')
        # select first repetion
        assembly = assembly[{'repeat':repetition}]
        # select stimuli that have the stim_group= version
        vox_reliability = {'language': (False, .95), 'auditory': (False, .95), 'visual': (False, .95)}
        vox_corr = {'language': (True, .1), 'auditory': (False, .1), 'visual': (False, .1)}
        if group=='all':
            pass
        else:
            assembly = assembly[assembly['stim_group'] == group]

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
        assembly.attrs['stimuli_group'] = 'DsParametricfMRI_' + group #+ f'_thr_{threshold}'
        if group=='all':
            # assign a new coordinate called stimulus_id with values from stim_id
            assembly=assembly.assign_coords({'stimulus_id': ('presentation', assembly.stim_id.values)})
        else:
            assembly=assembly.assign_coords({'stimulus_id': ('presentation', assembly.stim_group_id.values)})

        sentences = assembly['stimulus'].str.replace(r'\.$', '', regex=True)
        stimulus_set = StimulusSet({'sentence': sentences.values,
                                    'stimulus_num': assembly['stimulus_num'].values,
                                    'stimulus_id': assembly['stimulus_id'].values,
                                    'stim_name': assembly['stim_name'].values,
                                    'stim_group':assembly['stim_group'].values,
                                    'stim_group_id': assembly['stim_group_id'].values,
                                    'stumulus': sentences.values,
                                    'sentence_id': assembly['stimulus_id'].values})
        stimulus_set.name = assembly.attrs['stimuli_group']
        assembly.attrs['stimulus_set'] = stimulus_set
        return assembly
#
class DsParametricfMRIFirstReliableMaxRidgeEncoding(DsParametricSinglefMRIRidgeEncoding):
    def _load_assembly(self, version='DsParametricfMRI_rsa_subs_12_language', group='max', threshold=90,repetition=0):
        return super()._load_assembly(version='DsParametricfMRI_rsa_subs_12_language', group='max', threshold=90,repetition=0)
#
class DsParametricfMRIFirstReliableMinRidgeEncoding(DsParametricSinglefMRIRidgeEncoding):
    def _load_assembly(self, version='DsParametricfMRI_rsa_subs_12_language', group='min', threshold=90,repetition=0):
        return super()._load_assembly(version='DsParametricfMRI_rsa_subs_12_language', group='min', threshold=90,repetition=0)
class DsParametricfMRIFirstReliableRandRidgeEncoding(DsParametricSinglefMRIRidgeEncoding):
    def _load_assembly(self, version='DsParametricfMRI_rsa_subs_12_language', group='random', threshold=90,repetition=0):
        return super()._load_assembly(version='DsParametricfMRI_rsa_subs_12_language', group='random', threshold=90,repetition=0)

class DsParametricfMRISecondReliableMaxRidgeEncoding(DsParametricSinglefMRIRidgeEncoding):
    def _load_assembly(self, version='DsParametricfMRI_rsa_subs_12_language', group='max', threshold=90,repetition=1):
        return super()._load_assembly(version='DsParametricfMRI_rsa_subs_12_language', group='max', threshold=90,repetition=1)

class DsParametricfMRISecondReliableMinRidgeEncoding(DsParametricSinglefMRIRidgeEncoding):
    def _load_assembly(self, version='DsParametricfMRI_rsa_subs_12_language', group='min', threshold=90,repetition=1):
        return super()._load_assembly(version='DsParametricfMRI_rsa_subs_12_language', group='min', threshold=90,repetition=1)
class DsParametricfMRISecondReliableRandRidgeEncoding(DsParametricSinglefMRIRidgeEncoding):
    def _load_assembly(self, version='DsParametricfMRI_rsa_subs_12_language', group='random', threshold=90,repetition=1):
        return super()._load_assembly(version='DsParametricfMRI_rsa_subs_12_language', group='random', threshold=90,repetition=1)

class DsParametricSinglefMRIAllEncoding(DsParametricSinglefMRIEncoding):

    def _load_assembly(self,version='DsParametricfMRI_rsa_subs_12_language',group='max',threshold=90,repetition=0):
        assembly = pd.read_pickle(f'{DsParametricfMRI_PARENT}/{version}_top_{threshold}_reliability_random_analyzed_aug2024.pkl')
        # select first repetion
        assembly = assembly[{'repeat':repetition}]
        # select stimuli that have the stim_group= version
        vox_reliability = {'language': (False, .95), 'auditory': (False, .95), 'visual': (False, .95)}
        vox_corr = {'language': (False, .1), 'auditory': (False, .1), 'visual': (False, .1)}
        if group=='all':
            pass
        else:
            assembly = assembly[assembly['stim_group'] == group]

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
        assembly.attrs['stimuli_group'] = 'DsParametricfMRI_' + group #+ f'_thr_{threshold}'
        if group=='all':
            # assign a new coordinate called stimulus_id with values from stim_id
            assembly=assembly.assign_coords({'stimulus_id': ('presentation', assembly.stim_id.values)})
        else:
            assembly=assembly.assign_coords({'stimulus_id': ('presentation', assembly.stim_group_id.values)})

        sentences = assembly['stimulus'].str.replace(r'\.$', '', regex=True)
        stimulus_set = StimulusSet({'sentence': sentences.values,
                                    'stimulus_num': assembly['stimulus_num'].values,
                                    'stimulus_id': assembly['stimulus_id'].values,
                                    'stim_name': assembly['stim_name'].values,
                                    'stim_group':assembly['stim_group'].values,
                                    'stim_group_id': assembly['stim_group_id'].values,
                                    'stumulus': sentences.values,
                                    'sentence_id': assembly['stimulus_id'].values})
        stimulus_set.name = assembly.attrs['stimuli_group']
        assembly.attrs['stimulus_set'] = stimulus_set
        return assembly

class DsParametricfMRIFirstAllMaxEncoding(DsParametricSinglefMRIAllEncoding):
    def _load_assembly(self, version='DsParametricfMRI_rsa_subs_12_language', group='max', threshold=90,repetition=0):
        return super()._load_assembly(version='DsParametricfMRI_rsa_subs_12_language', group='max', threshold=90,repetition=0)

class DsParametricfMRIFirstAllMinEncoding(DsParametricSinglefMRIAllEncoding):
    def _load_assembly(self, version='DsParametricfMRI_rsa_subs_12_language', group='min', threshold=90,repetition=0):
        return super()._load_assembly(version='DsParametricfMRI_rsa_subs_12_language', group='min', threshold=90,repetition=0)
class DsParametricfMRIFirstAllRandEncoding(DsParametricSinglefMRIAllEncoding):
    def _load_assembly(self, version='DsParametricfMRI_rsa_subs_12_language', group='random', threshold=90,repetition=0):
        return super()._load_assembly(version='DsParametricfMRI_rsa_subs_12_language', group='random', threshold=90,repetition=0)
class DsParametricfMRISecondAllMaxEncoding(DsParametricSinglefMRIAllEncoding):
    def _load_assembly(self, version='DsParametricfMRI_rsa_subs_12_language', group='max', threshold=90,repetition=1):
        return super()._load_assembly(version='DsParametricfMRI_rsa_subs_12_language', group='max', threshold=90,repetition=1)
class DsParametricfMRISecondAllMinEncoding(DsParametricSinglefMRIAllEncoding):
    def _load_assembly(self, version='DsParametricfMRI_rsa_subs_12_language', group='min', threshold=90,repetition=1):
        return super()._load_assembly(version='DsParametricfMRI_rsa_subs_12_language', group='min', threshold=90,repetition=1)
class DsParametricfMRISecondAllRandEncoding(DsParametricSinglefMRIAllEncoding):
    def _load_assembly(self, version='DsParametricfMRI_rsa_subs_12_language', group='random', threshold=90,repetition=1):
        return super()._load_assembly(version='DsParametricfMRI_rsa_subs_12_language', group='random', threshold=90,repetition=1)
class DsParametricfMRISecondAllMaxEncoding(DsParametricSinglefMRIAllEncoding):
    def _load_assembly(self, version='DsParametricfMRI_rsa_subs_12_language', group='max', threshold=90,repetition=1):
        return super()._load_assembly(version='DsParametricfMRI_rsa_subs_12_language', group='max', threshold=90,repetition=1)

class DsParametricSinglefMRIStrictEncoding(_DsParametricfMRIBenchmark):
    """
    data source:
    """

    def __init__(self, **kwargs):
        metric = CrossRegressedCorrelation(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=10, kfold=True, split_coord='stimulus_id', stratification_coord=None))
        super(DsParametricSinglefMRIStrictEncoding, self).__init__(metric=metric, **kwargs)
    def _load_assembly(self,version='DsParametricfMRI_rsa_subs_12_language',group='max',threshold=90,repetition=0):
        assembly = pd.read_pickle(f'{DsParametricfMRI_PARENT}/{version}_top_{threshold}_reliability_random_analyzed_aug2024.pkl')
        # select first repetion
        assembly = assembly[{'repeat':repetition}]
        # select stimuli that have the stim_group= version
        vox_reliability = {'language': (False, .95), 'auditory': (False, .95), 'visual': (False, .95)}
        vox_corr = {'language': (True, .1), 'auditory': (False, .1), 'visual': (False, .1)}
        if group=='all':
            pass
        else:
            assembly = assembly[assembly['stim_group'] == group]

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
        assembly.attrs['stimuli_group'] = 'DsParametricfMRI_' + group #+ f'_thr_{threshold}'
        if group=='all':
            # assign a new coordinate called stimulus_id with values from stim_id
            assembly=assembly.assign_coords({'stimulus_id': ('presentation', assembly.stim_id.values)})
        else:
            assembly=assembly.assign_coords({'stimulus_id': ('presentation', assembly.stim_group_id.values)})

        sentences = assembly['stimulus'].str.replace(r'\.$', '', regex=True)
        stimulus_set = StimulusSet({'sentence': sentences.values,
                                    'stimulus_num': assembly['stimulus_num'].values,
                                    'stimulus_id': assembly['stimulus_id'].values,
                                    'stim_name': assembly['stim_name'].values,
                                    'stim_group':assembly['stim_group'].values,
                                    'stim_group_id': assembly['stim_group_id'].values,
                                    'stumulus': sentences.values,
                                    'sentence_id': assembly['stimulus_id'].values})
        stimulus_set.name = assembly.attrs['stimuli_group']
        assembly.attrs['stimulus_set'] = stimulus_set
        return assembly


# do 80%
class DsParametricfMRI80FirstMinEncoding(DsParametricSinglefMRIEncoding):
    def _load_assembly(self, version='DsParametricfMRI_rsa_subs_12_language', group='min', threshold=80, repetition=0):
        return super()._load_assembly(version='DsParametricfMRI_rsa_subs_12_language', group='min', threshold=80,
                                      repetition=0)
class DsParametricfMRI80FirstRandEncoding(DsParametricSinglefMRIEncoding):
    def _load_assembly(self, version='DsParametricfMRI_rsa_subs_12_language', group='random', threshold=80,
                       repetition=0):
        return super()._load_assembly(version='DsParametricfMRI_rsa_subs_12_language', group='random', threshold=80,
                                      repetition=0)
class DsParametricfMRI80FirstMaxEncoding(DsParametricSinglefMRIEncoding):
    def _load_assembly(self, version='DsParametricfMRI_rsa_subs_12_language', group='max', threshold=80, repetition=0):
        return super()._load_assembly(version='DsParametricfMRI_rsa_subs_12_language', group='max', threshold=80,
                                      repetition=0)
class DsParametricfMRI80SecondMinEncoding(DsParametricSinglefMRIEncoding):
    def _load_assembly(self, version='DsParametricfMRI_rsa_subs_12_language', group='min', threshold=80, repetition=1):
        return super()._load_assembly(version='DsParametricfMRI_rsa_subs_12_language', group='min', threshold=80,
                                      repetition=1)
class DsParametricfMRI80SecondRandEncoding(DsParametricSinglefMRIEncoding):
    def _load_assembly(self, version='DsParametricfMRI_rsa_subs_12_language', group='random', threshold=80,
                       repetition=1):
        return super()._load_assembly(version='DsParametricfMRI_rsa_subs_12_language', group='random', threshold=80,
                                      repetition=1)
class DsParametricfMRI80SecondMaxEncoding(DsParametricSinglefMRIEncoding):
    def _load_assembly(self, version='DsParametricfMRI_rsa_subs_12_language', group='max', threshold=80,repetition=1):
        return super()._load_assembly(version='DsParametricfMRI_rsa_subs_12_language', group='max', threshold=80,repetition=1)


class DsParametricSinglefMRIStrictEncoding(_DsParametricfMRIBenchmark):
    """
    data source:
    """

    def __init__(self, **kwargs):
        metric = CrossRegressedCorrelation(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=10, kfold=True, split_coord='stimulus_id', stratification_coord=None))
        super(DsParametricSinglefMRIStrictEncoding, self).__init__(metric=metric, **kwargs)

    def _load_assembly(self, version='DsParametricfMRI_rsa_subs_12_language', group='max', threshold=90, repetition=0):
        assembly = pd.read_pickle(f'{DsParametricfMRI_PARENT}/{version}_top_{threshold}_reliability_random_analyzed_aug2024.pkl')
        # select first repetion
        assembly = assembly[{'repeat': repetition}]
        # select stimuli that have the stim_group= version
        vox_reliability = {'language': (False, .95), 'auditory': (False, .95), 'visual': (False, .95)}
        vox_corr = {'language': (True, .1), 'auditory': (False, .1), 'visual': (False, .1)}
        if group == 'all':
            pass
        else:
            assembly = assembly[assembly['stim_group'] == group]

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
        assembly.attrs['stimuli_group'] = 'DsParametricfMRI_' + group  # + f'_thr_{threshold}'
        if group == 'all':
            # assign a new coordinate called stimulus_id with values from stim_id
            assembly = assembly.assign_coords({'stimulus_id': ('presentation', assembly.stim_id.values)})
        else:
            assembly = assembly.assign_coords({'stimulus_id': ('presentation', assembly.stim_group_id.values)})

        sentences = assembly['stimulus'].str.replace(r'\.$', '', regex=True)
        stimulus_set = StimulusSet({'sentence': sentences.values,
                                    'stimulus_num': assembly['stimulus_num'].values,
                                    'stimulus_id': assembly['stimulus_id'].values,
                                    'stim_name': assembly['stim_name'].values,
                                    'stim_group': assembly['stim_group'].values,
                                    'stim_group_id': assembly['stim_group_id'].values,
                                    'stumulus': sentences.values,
                                    'sentence_id': assembly['stimulus_id'].values})
        stimulus_set.name = assembly.attrs['stimuli_group']
        assembly.attrs['stimulus_set'] = stimulus_set
        return assembly


class DsParametricfMRIFirstMaxStrictEncoding(DsParametricSinglefMRIStrictEncoding):
    def _load_assembly(self, version='DsParametricfMRI_rsa_subs_12_language', group='max', threshold=90,repetition=0):
        return super()._load_assembly(version='DsParametricfMRI_rsa_subs_12_language', group='max', threshold=90,repetition=0)
class DsParametricfMRIFirstMinStrictEncoding(DsParametricSinglefMRIStrictEncoding):
    def _load_assembly(self, version='DsParametricfMRI_rsa_subs_12_language', group='min', threshold=90,repetition=0):
        return super()._load_assembly(version='DsParametricfMRI_rsa_subs_12_language', group='min', threshold=90,repetition=0)
class DsParametricfMRIFirstRandStrictEncoding(DsParametricSinglefMRIStrictEncoding):
    def _load_assembly(self, version='DsParametricfMRI_rsa_subs_12_language', group='random', threshold=90,repetition=0):
        return super()._load_assembly(version='DsParametricfMRI_rsa_subs_12_language', group='random', threshold=90,repetition=0)
class DsParametricfMRISecondMaxStrictEncoding(DsParametricSinglefMRIStrictEncoding):
    def _load_assembly(self, version='DsParametricfMRI_rsa_subs_12_language', group='max', threshold=90,repetition=1):
        return super()._load_assembly(version='DsParametricfMRI_rsa_subs_12_language', group='max', threshold=90,repetition=1)

class DsParametricfMRISecondMinStrictEncoding(DsParametricSinglefMRIStrictEncoding):
    def _load_assembly(self, version='DsParametricfMRI_rsa_subs_12_language', group='min', threshold=90,repetition=1):
        return super()._load_assembly(version='DsParametricfMRI_rsa_subs_12_language', group='min', threshold=90,repetition=1)
class DsParametricfMRISecondRandStrictEncoding(DsParametricSinglefMRIStrictEncoding):
    def _load_assembly(self, version='DsParametricfMRI_rsa_subs_12_language', group='random', threshold=90,repetition=1):
        return super()._load_assembly(version='DsParametricfMRI_rsa_subs_12_language', group='random', threshold=90,repetition=1)


class DsParametricSinglefMRIRidgeEncoding(_DsParametricfMRIBenchmark):

    def __init__(self, **kwargs):
        metric = CrossRegressedCorrelation(
            regression=rgcv_linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord='stimulus_id', stratification_coord=None))
        super(DsParametricSinglefMRIRidgeEncoding, self).__init__(metric=metric, **kwargs)
    def _load_assembly(self,version='DsParametricfMRI_rsa_subs_12_language',group='max',threshold=90,repetition=0):
        assembly = pd.read_pickle(f'{DsParametricfMRI_PARENT}/{version}_top_{threshold}_reliability_random_analyzed_aug2024.pkl')
        # select first repetion
        assembly = assembly[{'repeat':repetition}]
        # select stimuli that have the stim_group= version
        vox_reliability = {'language': (False, .95), 'auditory': (False, .95), 'visual': (False, .95)}
        vox_corr = {'language': (True, .1), 'auditory': (False, .1), 'visual': (False, .1)}
        if group=='all':
            pass
        else:
            assembly = assembly[assembly['stim_group'] == group]

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
        assembly.attrs['stimuli_group'] = 'DsParametricfMRI_' + group #+ f'_thr_{threshold}'
        if group=='all':
            # assign a new coordinate called stimulus_id with values from stim_id
            assembly=assembly.assign_coords({'stimulus_id': ('presentation', assembly.stim_id.values)})
        else:
            assembly=assembly.assign_coords({'stimulus_id': ('presentation', assembly.stim_group_id.values)})

        sentences = assembly['stimulus'].str.replace(r'\.$', '', regex=True)
        stimulus_set = StimulusSet({'sentence': sentences.values,
                                    'stimulus_num': assembly['stimulus_num'].values,
                                    'stimulus_id': assembly['stimulus_id'].values,
                                    'stim_name': assembly['stim_name'].values,
                                    'stim_group':assembly['stim_group'].values,
                                    'stim_group_id': assembly['stim_group_id'].values,
                                    'stumulus': sentences.values,
                                    'sentence_id': assembly['stimulus_id'].values})
        stimulus_set.name = assembly.attrs['stimuli_group']
        assembly.attrs['stimulus_set'] = stimulus_set
        return assembly


class DsParametricfMRIFirstMaxRidgeEncoding(DsParametricSinglefMRIRidgeEncoding):

    def _load_assembly(self, version='DsParametricfMRI_rsa_subs_12_language', group='max', threshold=90,repetition=0):
        return super()._load_assembly(version='DsParametricfMRI_rsa_subs_12_language', group='max', threshold=90,repetition=0)
class DsParametricfMRIFirstMinRidgeEncoding(DsParametricSinglefMRIRidgeEncoding):

    def _load_assembly(self, version='DsParametricfMRI_rsa_subs_12_language', group='min', threshold=90,repetition=0):
        return super()._load_assembly(version='DsParametricfMRI_rsa_subs_12_language', group='min', threshold=90,repetition=0)
class DsParametricfMRIFirstRandRidgeEncoding(DsParametricSinglefMRIRidgeEncoding):

    def _load_assembly(self, version='DsParametricfMRI_rsa_subs_12_language', group='random', threshold=90,repetition=0):
        return super()._load_assembly(version='DsParametricfMRI_rsa_subs_12_language', group='random', threshold=90,repetition=0)
class DsParametricfMRISecondMaxRidgeEncoding(DsParametricSinglefMRIRidgeEncoding):

    def _load_assembly(self, version='DsParametricfMRI_rsa_subs_12_language', group='max', threshold=90,repetition=1):
        return super()._load_assembly(version='DsParametricfMRI_rsa_subs_12_language', group='max', threshold=90,repetition=1)
class DsParametricfMRISecondMinRidgeEncoding(DsParametricSinglefMRIRidgeEncoding):

    def _load_assembly(self, version='DsParametricfMRI_rsa_subs_12_language', group='min', threshold=90,repetition=1):
        return super()._load_assembly(version='DsParametricfMRI_rsa_subs_12_language', group='min', threshold=90,repetition=1)
class DsParametricfMRISecondRandRidgeEncoding(DsParametricSinglefMRIRidgeEncoding):

    def _load_assembly(self, version='DsParametricfMRI_rsa_subs_12_language', group='random', threshold=90,repetition=1):
        return super()._load_assembly(version='DsParametricfMRI_rsa_subs_12_language', group='random', threshold=90,repetition=1)

class _DsParametricfMRIV2Benchmark(Benchmark):
    """
    data source:
        Pereira et al., nature communications 2018
        https://www.nature.com/articles/s41467-018-03068-4
    """

    def __init__(self, identifier, metric, version='max', threshold=90):
        self._identifier = identifier
        assembly = self._load_assembly(version=version, threshold=threshold)
        self._target_assembly = assembly
        self._single_metric = metric
        # self._ceiler = self.PereiraExtrapolationCeiling(subject_column='subject', num_bootstraps=100)
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

    # @load_s3(key='Pereira2018')
    def _load_assembly(self, version='max', threshold=90):
        assembly = pd.read_pickle(f'{DsParametricfMRI_PARENT}/DsParametricfMRI_train_language_top_{threshold}_reliability_random_analyzed_aug2024.pkl')
        # select stimuli that have the stim_group= version
        vox_reliability = {'language': (False, .95), 'auditory': (False, .95), 'visual': (False, .95)}
        vox_corr = {'language': (False, .1), 'auditory': (False, .1), 'visual': (False, .1)}
        # normalize the assembly
        a = stats.zscore(assembly.values, axis=0)
        assembly = assembly.copy(data=a)
        # make sure that the mean of each voxel is 0
        #assert np.allclose(assembly.mean('presentation').values, 0)
        assembly = assembly[assembly['stim_group'] == version]
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
        # create a stimulus set for the assembly
        stimulus_set = StimulusSet({'sentence': assembly['stimulus'].values,
                                    'stimulus_num': assembly['stimulus_num'].values,
                                    'stimulus_id': assembly['stimulus_id'].values,
                                    'stimulus_group': assembly['stim_group'].values,
                                    'stimulus_group_id': assembly['stim_group_id'].values,
                                    'stimulus_type': assembly['stim_type'].values,
                                    'stim_name': assembly['stim_name'].values,
                                    'experiment': assembly['experiment'].values,
                                    'stimulus': assembly['stimulus'].values,
                                    'sentence_id': assembly['stimulus_id'].values})
        # attach stimulus set as an attribute to the assembly
        # add name to stimulus set
        experiment = assembly['experiment'].values[0]
        stimulus_group = assembly['stim_group'].values[0]
        stimulus_set.name = f"{experiment}_{stimulus_group}"
        assembly.attrs['stimulus_set'] = stimulus_set

        return assembly

    def __call__(self, candidate):
        stimulus_set = self._target_assembly.attrs['stimulus_set']
        model_activations = listen_to_v2(candidate, stimulus_set,reset_column='stimulus_id',average_sentence=False)

        #model_activations = read_words(candidate, stimulus_set, copy_columns=['word_id'],reset_column='stimulus_id')
        # model_activations = self._listen_to(candidate, stimulus_set, reset_column='stimulus_passage_index')
        assert set(model_activations['stimulus_id'].values) == set(self._target_assembly['stimulus_id'].values)
        # add a new cooordinate called stimulus_id

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
        assert len(cross_assembly['presentation']) in [80]
        assert not np.isnan(cross_assembly).any()
        source_assembly = source_assembly[{'presentation': [stimulus_id in cross_assembly['stimulus_id'].values
                                                            for stimulus_id in source_assembly['stimulus_id'].values]}]
        return self._single_metric(source_assembly, cross_assembly)

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


class DsParametricfMRINormRidgeEncoding(_DsParametricfMRIV2Benchmark):
    def __init__(self, **kwargs):
        metric = CrossRegressedCorrelation(
            regression=rgcv_linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord='stimulus_id', stratification_coord=None,random_state=2))
        super(DsParametricfMRINormRidgeEncoding, self).__init__(metric=metric, **kwargs)


class DsParametricfMRINormMaxRidgeEncoding(DsParametricfMRINormRidgeEncoding):
    def _load_assembly(self,version='max',threshold=90):
        return super()._load_assembly(version='max',threshold=90)

class DsParametricfMRINormMinRidgeEncoding(DsParametricfMRINormRidgeEncoding):
    def _load_assembly(self,version='min',threshold=90):
        return super()._load_assembly(version='min',threshold=90)

class DsParametricfMRINormRandRidgeEncoding(DsParametricfMRINormRidgeEncoding):
    def _load_assembly(self,version='random',threshold=90):
        return super()._load_assembly(version='random',threshold=90)

class DsParametricfMRIPLSEncoding(_DsParametricfMRIBenchmark):
    """
    data source:
    """

    def __init__(self, **kwargs):
        metric = CrossRegressedCorrelation(
            regression=pls_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord='stimulus_id', stratification_coord=None))
        super(DsParametricfMRIRidgeEncoding, self).__init__(metric=metric, **kwargs)


    # @property
    # def ceiling(self):
    #     ceiling_val=pd.read_pickle(f'{ANNfMRI_PARENT}/ANNSet1_fMRI-train-language_top_90-linear_ceiling.pkl')
    #     return ceiling_val

class DsParametricfMRIMaxPLSEncoding(DsParametricfMRIRidgeEncoding):
    def _load_assembly(self,version='max'):
        return super()._load_assembly(version='max')

class DsParametricfMRIMinPLSEncoding(DsParametricfMRIRidgeEncoding):
    def _load_assembly(self,version='min'):
        return super()._load_assembly(version='min')

class DsParametricfMRIRandPLSEncoding(DsParametricfMRIRidgeEncoding):
    def _load_assembly(self,version='random'):
        return super()._load_assembly(version='random')


class DsParametricRDM(_DsParametricfMRIBenchmark):
    def __init__(self, **kwargs):
        metric = rsa_correlation()
        super(DsParametricRDM, self).__init__(metric=metric, **kwargs)

    def _load_assembly(self, version='max', threshold=90):
        assembly = pd.read_pickle(
            f'{DsParametricfMRI_PARENT}/DsParametricfMRI_rsa_train_language_top_{threshold}_reliability_random_analyzed_aug2024.pkl')
        # select stimuli that have the stim_group= version
        vox_reliability = {'language': (True, .95), 'auditory': (False, .95), 'visual': (False, .95)}
        vox_corr = {'language': (False, .1), 'auditory': (False, .1), 'visual': (False, .1)}
        assembly = assembly[:, assembly['stim_group'].values == version, :]
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
        assembly.attrs['stimuli_group'] = 'DsParametricfMRI_' + version  # + f'_thr_{threshold}'

        rep_list = []
        for rep_id, rep in assembly.groupby('repeat'):
            rep_list.append(
                rep.assign_coords({'repeat_id': ('neuroid', (np.ones(rep.shape[1]).astype(int) * rep.repeat.values))}))
        rep_xr = xr.concat(rep_list, 'presentation')
        # seperate the data based on subject:
        rsa_dataset = []
        for sub_id, sub_dat in rep_xr.groupby('subject'):
            descriptors = {'subject': sub_dat.subject.values[0]
                           }
            ch_descriptors = {'neuroid': sub_dat.neuroid_id.values,
                              'roi': sub_dat.roi.values}

            obs_descriptors = {'stimulus_id': sub_dat.stimulus_id.values,
                               'stimulus_num': sub_dat.stimulus_num.values,
                               'sentence': sub_dat.sentence.values,
                               'stimulus': sub_dat.stimulus.values,
                               'session': sub_dat.repeat_id.values[:, 0],
                               }
            rsa_dataset.append(
                rsd.Dataset(measurements=sub_dat.values, descriptors=descriptors, obs_descriptors=obs_descriptors,
                            channel_descriptors=ch_descriptors))
        stimulus_passage_category_id = [f'{assembly["stim_group"].values[i]}_{assembly["stim_group_id"].values[i]}' for
                                        i in range(len(assembly['stim_group_id'].values))]
        # use assign coords to add the new stimulus_passage_category_id to presentation diemsnions to the assembly
        assembly = assembly.assign_coords(stimulus_group_category_id=('presentation', stimulus_passage_category_id))

        # construct stimulus set from the assembly
        stimulus_set = StimulusSet({'sentence': assembly['stimulus'].values,
                                    'stimulus_num': assembly['stimulus_num'].values,
                                    'stimulus_id': assembly['stimulus_id'].values,
                                    'stim_group': assembly['stim_group'].values,
                                    'stim_group_id': assembly['stim_group_id'].values,
                                    'stim_name': assembly['stim_name'].values,
                                    'experiment': assembly['experiment'].values,
                                    'stumulus': assembly['stimulus'].values,
                                    'sentence_id': assembly['stimulus_id'].values,
                                    'stimulus_group_category_id': assembly['stimulus_group_category_id'].values})
        # attach stimulus set as an attribute to the assembly
        # add name to stimulus set
        stimulus_set.name = f"{assembly.attrs['stimuli_group']}"
        assembly.attrs['stimulus_set'] = stimulus_set

        rsa_dict = {}
        # add rsa dataset and stimulsset to the dictionary
        rsa_dict['rsa_dataset'] = rsa_dataset
        rsa_dict['stimulus_set'] = stimulus_set
        return rsa_dict

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
            sentence_stimuli = StimulusSet({'sentence': part_stimuli['sentence'].values,
                                            reset_column: list(set(part_stimuli[reset_column].values))})
            sentence_stimuli.name = f"{stimulus_set.name}-{reset_id}"
            sentence_activations = candidate(stimuli=sentence_stimuli, average_sentence=average_sentence)[-1, :]
            # for column in copy_columns:
            #    sentence_activations[column] = ('presentation', part_stimuli[column])
            activations.append(sentence_activations)

        # model_activations = merge_data_arrays(activations)
        model_activations = xr.concat(activations, dim='presentation')
        # merging does not maintain stimulus order. the following orders again
        idx = [model_activations['stim_group_id'].values.tolist().index(stimulus_id) for stimulus_id in
               [int(s['stim_group_id'].values) for s in activations]]
        assert len(set(idx)) == len(idx), "Found duplicate indices to order activations"
        model_activations = model_activations[{'presentation': idx}]

        return model_activations

    def _apply_cross(self, source_assembly, cross_assembly):
        cross_assembly = cross_assembly.dropna('neuroid')  # some subjects have only done one experiment
        source_assembly = source_assembly.dropna('neuroid')  # only relevant when running audio-visual self as "model"
        assert len(cross_assembly['presentation']) in [80]
        assert not np.isnan(cross_assembly).any()
        source_assembly = source_assembly[{'presentation': [stimulus_id in cross_assembly['stimulus_id'].values
                                                            for stimulus_id in source_assembly['stimulus_id'].values]}]
        return self._metric(source_assembly, cross_assembly)

    def __call__(self, candidate):
        stimulus_set = self._target_assembly['stimulus_set']
        model_activations = self._read_words(candidate, stimulus_set, copy_columns=['word_id'],
                                             reset_column='stim_group_id')
        # add a new cooordinate called stimulus_id
        model_activations = model_activations.assign_coords(
            {'stimulus_id': ('presentation', model_activations.stim_group_id.values)})
        descriptors = {'model': candidate._model.identifier,
                       }
        obs_descriptors = {'stimulus_id': model_activations.stimulus_id.values,
                           'stimulus': model_activations.presentation.sentence.values}
        ch_descriptors = {'neuroid': model_activations.neuroid_id.values,
                          'neuron_number_in_layer': model_activations.neuroid_num.values}
        predictions_rsd = rsd.Dataset(model_activations.values, descriptors=descriptors,
                                      obs_descriptors=obs_descriptors, channel_descriptors=ch_descriptors)
        # for target assembly go through all coordsinate and assign them again
        # coords = {coord: coord_value for coord, coord_value in self._target_assembly.coords.items() }
        _logger.info('Scoring across experiments & atlases')
        raw_score = self._single_metric(predictions_rsd, self._target_assembly['rsa_dataset'])
        score = Score([raw_score.get_means(), raw_score.get_errorbars()[0]],
                      coords={'aggregation': ['center', 'error'], },
                      dims=['aggregation', 'layer'])
        score.attrs['raw'] = raw_score
        score.attrs['noise_ceiling'] = raw_score.get_noise_ceil()
        return score


class DsParametricRDMMax(DsParametricRDM):
    def _load_assembly(self, version='max', threshold=90):
        return super()._load_assembly(version='max', threshold=90)


class DsParametricRDMMin(DsParametricRDM):
    def _load_assembly(self, version='min', threshold=90):
        return super()._load_assembly(version='min', threshold=90)


class DsParametricRDMRand(DsParametricRDM):
    def _load_assembly(self, version='random', threshold=90):
        return super()._load_assembly(version='random', threshold=90)

class _Pereira2023audBenchmark(Benchmark):
    """
    data source:
        Pereira et al., nature communications 2018
        https://www.nature.com/articles/s41467-018-03068-4
    """

    def __init__(self, identifier, metric,version='sent',reset_column='sentence_id', threshold=90):
        self._identifier = identifier
        assembly = self._load_assembly( threshold=threshold,version=version)
        self._target_assembly = assembly
        self._single_metric = metric
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
        assembly = pd.read_pickle(f'{DsParametricfMRI_PARENT}/Pereira2023aud_{version}_train_language_top_{threshold}_reliability_random_analyzed_aug2024.pkl')
        # select stimuli that have the stim_group= version
        vox_reliability = {'language': (True, .95), 'auditory': (False, .95), 'visual': (False, .95)}
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
                                    'sentence_id': assembly['stimulus_id'].values,
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
        return self._single_metric(source_assembly, cross_assembly)

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
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord='stimulus_id', stratification_coord=None))
        super(Pereira2023audRidgeEncoding, self).__init__(metric=metric, **kwargs)

class Pereira2023audV2RidgeEncoding(_Pereira2023audBenchmark):
    def __init__(self, **kwargs):
        metric = CrossRegressedCorrelation(
            regression=rgcv_linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord='stimulus_id', stratification_coord=None))
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
        super(Pereira2023audSentRidgeEncoding, self).__init__(reset_column='sentence_id',**kwargs)
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
        super(Pereira2023audPassSentenceRidgeEncoding, self).__init__(reset_column='sentence_id',**kwargs)
    def _load_assembly(self, version='pass', threshold=90):
        return super()._load_assembly(version='pass', threshold=90)

    def _get_model_activations(self, candidate, reset_column='sentence_id',
                               copy_columns=['sentence_id']):
        return super()._get_model_activations(candidate, reset_column='sentence_id',
                                              copy_columns=['stimulus_id'])


class _Pereira2023audSentenceBenchmark(_Pereira2023audBenchmark):
    def __init__(self, **kwargs):
        super(_Pereira2023audSentenceBenchmark, self).__init__(**kwargs)
    def _load_assembly(self,version='sent',threshold=90):
        return super()._load_assembly(version='sent',threshold=90)

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
            #sentence_stimuli = StimulusSet({'sentence': part_stimuli.values[0],
            #                                reset_column: list(set(part_stimuli[reset_column].values))})
            sentence_stimuli = StimulusSet({'sentence': part_stimuli.sentence,
                                            reset_column: list(set(part_stimuli[reset_column].values))})
            sentence_stimuli.name = f"{self._target_assembly.stimuli_group}-read-{reset_id}"
            sentence_activations = candidate(stimuli=sentence_stimuli, average_sentence=average_sentence)
            num_words=len(str(part_stimuli.sentence.values[0]).split(' '))
            assert(sentence_activations.shape[0]==num_words)
            sentence_activations=sentence_activations[-1, :]
            # for column in copy_columns:
            #    sentence_activations[column] = ('presentation', part_stimuli[column])
            activations.append(sentence_activations)

        # model_activations = merge_data_arrays(activations)
        model_activations = xr.concat(activations, dim='presentation')
        # merging does not maintain stimulus order. the following orders again
        idx = [model_activations['stimulus_id'].values.tolist().index(stimulus_id) for stimulus_id in
               [int(s['stimulus_id'].values) for s in activations]]
        assert len(set(idx)) == len(idx), "Found duplicate indices to order activations"
        model_activations = model_activations[{'presentation': idx}]

        return model_activations
    def __call__(self, candidate):
        stimulus_set = self._target_assembly.attrs['stimulus_set']
        _logger.warning(f'extracting activation on {self._reset_column}')
            # #model_activations = self._read_words(candidate, stimulus_set, copy_columns=['word_id'],
            # #                                     reset_column='stimulus_id')
        model_activations = self._read_words(candidate, stimulus_set, copy_columns=['word_id'],reset_column='stimulus_id')
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



class Pereira2023audSentRidgeEncoding(_Pereira2023audSentenceBenchmark):
    def __init__(self, **kwargs):
        metric = CrossRegressedCorrelation(
            regression=rgcv_linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord='stimulus_id', stratification_coord=None))
        super(Pereira2023audSentRidgeEncoding, self).__init__(metric=metric, **kwargs)
class Pereira2023audSentSentenceEncoding(_Pereira2023audSentenceBenchmark):
    def __init__(self, **kwargs):
        metric = CrossRegressedCorrelation(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord='stimulus_id', stratification_coord=None))
        super(Pereira2023audSentSentenceEncoding, self).__init__(metric=metric, **kwargs)
class Pereira2023audSentSentenceRidgeEncoding(Pereira2023audSentRidgeEncoding):
    def __init__(self, **kwargs):
        super(Pereira2023audSentSentenceRidgeEncoding, self).__init__(reset_column='stim_name',**kwargs)
    def _load_assembly(self, version='sent', threshold=90):
        return super()._load_assembly(version='sent', threshold=90)

    # def _get_model_activations(self, candidate, reset_column='sentence_id',
    #                            copy_columns=['sentence_id']):
    #     return super()._get_model_activations(candidate, reset_column='sentence_id',
    #                                           copy_columns=['stimulus_id'])

class Pereira2023audSentPassageRidgeEncoding(Pereira2023audSentRidgeEncoding):
    def __init__(self, **kwargs):
        super(Pereira2023audSentPassageRidgeEncoding, self).__init__(reset_column='stimulus_passage_category_id',**kwargs)
    def _load_assembly(self, version='sent', threshold=90):
        return super()._load_assembly(version='sent', threshold=90)

    # def _get_model_activations(self, candidate, reset_column='sentence_id',
    #                            copy_columns=['sentence_id']):
    #     return super()._get_model_activations(candidate, reset_column='sentence_id',
    #                                           copy_columns=['stimulus_id'])

class Pereira2023audPassPassageSampleRidgeEncoding(Pereira2023audSentRidgeEncoding):
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
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord='stimulus_id', stratification_coord=None))
        super(Pereira2023audEncoding, self).__init__(metric=metric, **kwargs)


class Pereira2023audPassPassageEncoding(Pereira2023audEncoding):
    def __init__(self, **kwargs):
        super(Pereira2023audPassPassageEncoding, self).__init__(reset_column='stimulus_passage_category_id',
                                                                     **kwargs)

    def _load_assembly(self, version='pass', threshold=90):
        return super()._load_assembly(version='pass', threshold=90)


class Pereira2023audPassPassageSampleEncoding(Pereira2023audPassPassageEncoding):
    def __call__(self, candidate):
        stimulus_set = self._target_assembly.attrs['stimulus_set']
        _logger.warning(f'extracting activation on {self._reset_column}')
            # #model_activations = self._read_words(candidate, stimulus_set, copy_columns=['word_id'],
            # #                                     reset_column='stimulus_id')
        model_activations = listen_to(candidate, stimulus_set,reset_column=self._reset_column)
        #model_activations = self._get_model_activations(candidate)
        # make sure model_activations and target_assembly have the same order of stimuli
        assert (model_activations['stimulus_id'].values == self._target_assembly['stimulus_id'].values).all()
        random_stimulus_ids = np.random.choice(model_activations['stimulus_id'].values, 80, replace=False)
        # find location of random_stimulus_ids in model_activations.stimulus_id
        random_stimulus_ids_idx = np.where(np.isin(model_activations['stimulus_id'].values, random_stimulus_ids))[0]
        model_activation_sample = model_activations[random_stimulus_ids_idx, :]
        target_assembly_sample = self._target_assembly[random_stimulus_ids_idx, :]
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

class Pereira2023audPassSentenceEncoding(Pereira2023audEncoding):
    def __init__(self, **kwargs):
        super(Pereira2023audPassSentenceEncoding, self).__init__(reset_column='sentence_id', **kwargs)

    def _load_assembly(self, version='pass', threshold=90):
        return super()._load_assembly(version='pass', threshold=90)

    def _get_model_activations(self, candidate, reset_column='sentence_id',
                               copy_columns=['sentence_id']):
        return super()._get_model_activations(candidate, reset_column='sentence_id',
                                              copy_columns=['stimulus_id'])


# class Pereira2023audSentSentenceEncoding(Pereira2023audEncoding):
#     def __init__(self, **kwargs):
#         super(Pereira2023audSentSentenceEncoding, self).__init__(reset_column='stim_name', **kwargs)
#
#     def _load_assembly(self, version='sent', threshold=90):
#         return super()._load_assembly(version='sent', threshold=90)
#
#     # def _get_model_activations(self, candidate, reset_column='sentence_id',
#     #                            copy_columns=['sentence_id']):
#     #     return super()._get_model_activations(candidate, reset_column='sentence_id',
#     #                                           copy_columns=['stimulus_id'])


class Pereira2023audSentPassageEncoding(Pereira2023audEncoding):
    def __init__(self, **kwargs):
        super(Pereira2023audSentPassageEncoding, self).__init__(reset_column='stimulus_passage_category_id',
                                                                     **kwargs)

    def _load_assembly(self, version='sent', threshold=90):
        return super()._load_assembly(version='sent', threshold=90)

    # def _get_model_activations(self, candidate, reset_column='sentence_id',
    #                            copy_columns=['sentence_id']):
    #     return super()._get_model_activations(candidate, reset_column='sentence_id',
    #                                           copy_columns=['stimulus_id'])


class _Fedorenko2016:
    """
    data source:
        Fedorenko et al., PNAS 2016
        https://www.pnas.org/content/113/41/E6256
    """

    def __init__(self, identifier, metric):
        self._identifier = identifier
        assembly = LazyLoad(self.load_assembly)
        self._target_assembly = assembly
        self._metric = metric
        self._average_sentence = False
        self._ceiler = ExtrapolationCeiling(subject_column='subject_UID')
        self._electrode_ceiler = self.ElectrodeExtrapolation(subject_column='subject_UID')

    @property
    def identifier(self):
        return self._identifier

    def load_assembly(self):
        raise NotImplementedError()

    def __call__(self, candidate):
        _logger.info('Computing activations')
        stimulus_set = self._target_assembly.attrs['stimulus_set']
        model_activations = read_words(candidate, stimulus_set,
                                       average_sentence=self._average_sentence, copy_columns=['stimulus_id'])
        assert (model_activations['stimulus_id'].values == self._target_assembly['stimulus_id'].values).all()
        score = self.apply_metric(model_activations, self._target_assembly)
        score = self.ceiling_normalize(score)
        return score

    def apply_metric(self, model_activations, target_assembly):
        return self._metric(model_activations, target_assembly)

    def ceiling_normalize(self, score):
        raw_neuroids = apply_aggregate(lambda values: values.mean('split'), score.raw)
        score = ceil_neuroids(raw_neuroids, self.ceiling, subject_column='subject_UID')
        return score

    @property
    def ceiling(self):
        return self._ceiler(identifier=self.identifier, assembly=self._target_assembly, metric=self._metric)

    @property
    def electrode_ceiling(self):
        return self._electrode_ceiler(identifier=self.identifier, assembly=self._target_assembly, metric=self._metric)

    class ElectrodeExtrapolation(ExtrapolationCeiling):
        """ extrapolate to infinitely many electrodes """

        def __init__(self, *args, **kwargs):
            super(_Fedorenko2016.ElectrodeExtrapolation, self).__init__(*args, **kwargs)
            self._rng = RandomState(0)
            self._num_samples = 15  # number of samples per electrode selection

        def collect(self, identifier, assembly, metric):
            """ Instead of iterating over subject combinations and then afterwards over holdout subjects,
            we here iterate over holdout subjects and then over electrode sub-combinations of the remaining pool. """
            subjects = set(assembly[self.subject_column].values)
            scores = []
            for holdout_subject in tqdm(subjects, desc='subjects'):
                subject_pool = subjects - {holdout_subject}
                subject_pool_assembly = assembly[{'neuroid': [subject in subject_pool
                                                              for subject in assembly[self.subject_column].values]}]
                holdout_subject_assembly = assembly[{'neuroid': [subject == holdout_subject
                                                                 for subject in assembly[self.subject_column].values]}]

                electrodes = subject_pool_assembly['neuroid_id'].values
                electrodes_range = np.arange(5, len(electrodes), 5)
                for num_electrodes in tqdm(electrodes_range, desc='num electrodes'):
                    electrodes_combinations = self._choose_electrodes(electrodes, num_electrodes,
                                                                      num_choices=self._num_samples)
                    for electrodes_split, electrodes_selection in enumerate(electrodes_combinations):
                        electrodes_assembly = subject_pool_assembly[{'neuroid': [
                            neuroid_id in electrodes_selection
                            for neuroid_id in subject_pool_assembly['neuroid_id'].values]}]
                        score = metric(electrodes_assembly, holdout_subject_assembly)
                        # store scores
                        score = score.expand_dims(f"sub_{self.subject_column}")
                        score.__setitem__(f"sub_{self.subject_column}", [holdout_subject])
                        score = score.expand_dims('num_electrodes').expand_dims('electrodes_split')
                        score['num_electrodes'] = [num_electrodes]
                        score['electrodes_split'] = [electrodes_split]
                        scores.append(score)

            scores = Score.merge(*scores)
            ceilings = scores.raw
            ceilings = ceilings.rename({'split': 'subsplit'}).stack(split=['electrodes_split', 'subsplit'])
            ceilings.attrs['raw'] = scores
            return ceilings

        def _choose_electrodes(self, electrodes, num_electrodes, num_choices):
            choices = [self._rng.choice(electrodes, size=num_electrodes, replace=False) for _ in range(num_choices)]
            return choices

class Fedorenko2016Encoding(_Fedorenko2016):
    """
    Fedorenko benchmark with encoding metric

    data source:
        Fedorenko et al., PNAS 2016
        https://www.pnas.org/content/113/41/E6256
    """

    def __init__(self, identifier):
        regression = linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id'))  # word
        correlation = pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id'))
        metric = CrossRegressedCorrelation(regression=regression, correlation=correlation,
                                           crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord='stimulus_id',
                                                                       stratification_coord='sentence_id'))
        super(Fedorenko2016Encoding, self).__init__(identifier=identifier, metric=metric)

class Fedorenko2016V3Encoding(Fedorenko2016Encoding):
    """
    Fedorenko benchmark, language electrodes

    data source:
        Fedorenko et al., PNAS 2016
        https://www.pnas.org/content/113/41/E6256
    """

    @load_s3(key='Fedorenko2016v3')
    def load_assembly(self):
        return LazyLoad(lambda: load_Fedorenko2016(electrodes='language', version=3))

    @property
    @load_s3(key='Fedorenko2016v3-encoding-ceiling')
    def ceiling(self):
        return super(Fedorenko2016V3Encoding, self).ceiling

class Fedorenko2016V3NonLangEncoding(Fedorenko2016Encoding):
    """
    Fedorenko benchmark, non-language electrodes (only sorted based on signal)
    Data 03/24/2020: sentence_electrode_more_elec_max_window_dat (not demeaned across sentences)

    data source:
        Fedorenko et al., PNAS 2016
        https://www.pnas.org/content/113/41/E6256
    """

    @load_s3(key='Fedorenko2016v3nonlang')
    def load_assembly(self):
        return LazyLoad(lambda: load_Fedorenko2016(electrodes='non-language', version=3))

    @property
    @load_s3(key='Fedorenko2016v3nonlang-encoding-ceiling')
    def ceiling(self):
        return super(Fedorenko2016V3NonLangEncoding, self).ceiling

class _Fedorenko2016V3SubjectWise(_Fedorenko2016):
    """
    data source:
        Fedorenko et al., PNAS 2016
        https://www.pnas.org/content/113/41/E6256
    """

    def __init__(self, identifier, metric):
        super(_Fedorenko2016V3SubjectWise, self).__init__(identifier=identifier, metric=metric)
        self._ceiler.extrapolation_dimension = 'subject_UID'
        self._cross = CartesianProduct(dividers=['subject_UID'])

    @load_s3(key='Fedorenko2016v3')
    def load_assembly(self):
        return LazyLoad(lambda: load_Fedorenko2016(electrodes='language', version=3))

    def apply_metric(self, source_assembly, target_assembly):
        cross_scores = self._cross(target_assembly, apply=
        lambda cross_assembly: super(_Fedorenko2016V3SubjectWise, self).apply_metric(source_assembly, cross_assembly))
        score = cross_scores.median(['subject_UID'])
        score.attrs['raw'] = cross_scores
        return score

    def ceiling_normalize(self, score):
        score = aggregate_ceiling(score.raw, ceiling=self.ceiling, subject_column='subject_UID')
        return score

class Fedorenko2016V3RDM(_Fedorenko2016V3SubjectWise):
    """
    data source:
        Fedorenko et al., PNAS 2016
        https://www.pnas.org/content/113/41/E6256
    """

    def __init__(self, identifier):
        metric = RDMCrossValidated(
            comparison_coord='stimulus_id',
            crossvalidation_kwargs=dict(split_coord='stimulus_id', stratification_coord='sentence_id',
                                        # doesn't work because train_size is deemed too small.
                                        # even though `train` is not needed, CrossValidation still splits it that way
                                        splits=5, kfold=True, test_size=None))
        super(Fedorenko2016V3RDM, self).__init__(identifier=identifier, metric=metric)

    # @property
    # @load_s3(key='Fedorenko2016v3-rdm-ceiling')
    # def ceiling(self):
    #     return super(Fedorenko2016V3RDM, self).ceiling

class Fedorenko2016V3CKA(_Fedorenko2016V3SubjectWise):
    """
    data source:
        Fedorenko et al., PNAS 2016
        https://www.pnas.org/content/113/41/E6256
    """

    def __init__(self, identifier):
        metric = CKACrossValidated(
            comparison_coord='stimulus_id',
            crossvalidation_kwargs=dict(split_coord='stimulus_id', stratification_coord='sentence_id',
                                        # doesn't work because train_size is deemed too small.
                                        # even though `train` is not needed, CrossValidation still splits it that way
                                        splits=5, kfold=True, test_size=None))
        super(Fedorenko2016V3CKA, self).__init__(identifier=identifier, metric=metric)

class _ANNSet1ECoGBenchmark:
    def __init__(self, identifier, metric,version='HighGamma_bipolar_gauss_zscore',type='language',threshold=0.05):
        self._identifier = identifier
        self._metric = metric
        assembly = self._load_assembly(version=version,type=type,threshold=threshold)
        self._target_assembly = assembly
        self.average_sentence=False
        self._ceiler = ExtrapolationCeiling(subject_column='subject')

    @property
    def ceiling(self):
        return self._ceiler(identifier=self.identifier, assembly=self._target_assembly, metric=self._metric)

    @property
    def identifier(self):
        return self._identifier

    def apply_metric(self, model_activations, target_assembly):
        return self._metric(model_activations, target_assembly)

    def _load_assembly(self,version='HighGamma_unipolar_gauss_zscore',type='language',threshold=0.05):
        file_id = Path(ANNfMRI_PARENT, f'ANNSet1_ECoG.{version}.pkl')
        assembly = pd.read_pickle(file_id.__str__())
        # make sure the assembly is ordered based on stimulus_id
        assembly = assembly.sortby('stimulus_id')
        # this is to make sure that the assembly is ordered based on stimulus_id
        # and that the word_ids are in increasing order
        assembly = assembly.groupby('stimulus_id').apply(lambda x: x.sortby('word_id'))
        # define a new coordinate called sentence_id with the same values as stimulus_id
        assembly = assembly.assign_coords(sentence_id=assembly.stimulus_id)
        # define a new coordiante stimuli_id that goes from 0 to size presentation dimension
        assembly = assembly.assign_coords({'stimuli_id':('presentation',np.arange(assembly.stimulus_id.size))})

        if type == 'language':
            s_v_n = assembly.S_vs_N_ratio >= (1 - threshold)
        elif type == 'non-language':
            s_v_n = assembly.S_vs_N_ratio < (1 - threshold)

        assembly = assembly[{'neuroid': (s_v_n) & (assembly['electrode_valid'] == 1)}]
        assembly_new = []
        for grp, sub in assembly.groupby('subject'):
            if sub.neuroid.size < 5:
                pass
            else:
                assembly_new.append(sub)
        assembly_new = xr.concat(assembly_new, dim='neuroid')
        assert not np.isnan(assembly_new).any()
        assembly = NeuroidAssembly(assembly_new)
        # add identifier to assembly
        thr_str = str(threshold).replace('.', '')
        assembly.attrs['identifier'] = f"ANNSet1_ECoG.{version}_{type}_thr_{thr_str}"
        return assembly

        return assembly

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
            sentence_stimuli = StimulusSet({'sentence': part_stimuli.values[0],
                                            reset_column: list(set(part_stimuli[reset_column].values))})
            sentence_stimuli.name = f"{self._target_assembly.identifier}-{reset_id}"
            sentence_activations = candidate(stimuli=sentence_stimuli, average_sentence=average_sentence)
            for column in copy_columns:
                sentence_activations[column] = ('presentation', part_stimuli[column])

            activations.append(sentence_activations)

        # model_activations = merge_data_arrays(activations)
        model_activations = xr.concat(activations, dim='presentation')
        # merging does not maintain stimulus order. the following orders again
        # assert that the order of model_activations is the same as stimulus_set
        assert np.all(model_activations[reset_column].values == stimulus_set[reset_column].values)
        #idx = [model_activations[reset_column].values.tolist().index(stimulus_id) for stimulus_id in
        #       [int(s[reset_column].values) for s in activations]]
        #assert len(set(idx)) == len(idx), "Found duplicate indices to order activations"
        #model_activations = model_activations[{'presentation': idx}]

        return model_activations

    def __call__(self,candidate, *args, **kwargs):
        stimulus_set = self._target_assembly['stimulus']
        model_activations = self._read_words(candidate, stimulus_set, copy_columns=['stimuli_id','word_id'])
        # make sure word_ids are in increasing order and the same between target_assembly and model_activations
        model_activations = model_activations.groupby('stimulus_id').apply(lambda x: x.sortby('word_id'))
        target_assembly = self._target_assembly.groupby('stimulus_id').apply(lambda x: x.sortby('word_id'))
        # make sure the model_activations and target_assembly have the same number of words
        assert np.all(model_activations['word_id'].values == target_assembly['word_id'].values)
        assert np.all(model_activations['stimuli_id'].values == target_assembly['stimuli_id'].values)

        _logger.info('Scoring across electrodes')
        score = self.apply_metric(model_activations, self._target_assembly)
        return score

    def _apply_cross(self, source_assembly, cross_assembly):
        cross_assembly = cross_assembly.dropna('neuroid')  # some subjects have only done one experiment
        source_assembly = source_assembly.dropna('neuroid')  # only relevant when running audio-visual self as "model"
        #assert len(cross_assembly['presentation']) in [200]
        assert not np.isnan(cross_assembly).any()
        source_assembly = source_assembly[{'presentation': [stimulus_id in cross_assembly['stimulus_id'].values
                                                            for stimulus_id in source_assembly['stimulus_id'].values]}]
        return self._single_metric(source_assembly, cross_assembly)

class ANNSet1ECoGEncoding(_ANNSet1ECoGBenchmark):
    def __init__(self,identifier):
        regression = linear_regression(xarray_kwargs=dict(stimulus_coord='stimuli_id'))  # word
        correlation = pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimuli_id')) # word
        metric = CrossRegressedCorrelation(regression=regression, correlation=correlation,
                                           crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord='stimuli_id',
                                                                       stratification_coord='stimulus_id'))

        super(ANNSet1ECoGEncoding, self).__init__(identifier=identifier, metric=metric,type='language',version='HighGamma_bipolar_gauss_zscore',threshold=0.05)

    @property
    def ceiling(self):
        return super(ANNSet1ECoGEncoding, self).ceiling

class ANNSetECoGSentenceEncoding(_ANNSet1ECoGBenchmark):
    def __init__(self,identifier):
        regression = linear_regression(xarray_kwargs=dict(stimulus_coord='stimuli_id'))  # word
        correlation = pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimuli_id'))  # word
        metric = CrossRegressedCorrelation(regression=regression, correlation=correlation,
                                           crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord='stimuli_id',
                                                                       stratification_coord=None))

        super(ANNSetECoGSentenceEncoding, self).__init__(identifier=identifier, metric=metric,type='language',version='HighGamma_bipolar_gauss_zscore',threshold=0.05)

    def _read_words(self, candidate, stimulus_set, reset_column='stimulus_id', copy_columns=(), average_sentence=True):
        """
        Pass a `stimulus_set` through a model `candidate`.
        In contrast to the `listen_to` function, this function operates on a word-based `stimulus_set`.
        """
        # Input: stimulus_set = pandas df, col 1 with sentence ID and 2nd col as word.
        activations = []
        for i, reset_id in enumerate(ordered_set(stimulus_set[reset_column].values)):
            part_stimuli = stimulus_set[stimulus_set[reset_column] == reset_id]
            # stimulus_ids = part_stimuli['stimulus_id']
            sentence_stimuli = StimulusSet({'sentence': part_stimuli.values[0],
                                            reset_column: list(set(part_stimuli[reset_column].values))})
            sentence_stimuli.name = f"{self._target_assembly.identifier}-ave-{average_sentence}-{reset_id}"
            sentence_activations = candidate(stimuli=sentence_stimuli, average_sentence=average_sentence)
            for column in copy_columns:
                sentence_activations[column] = ('presentation', part_stimuli[column])

            activations.append(sentence_activations)

        # model_activations = merge_data_arrays(activations)
        model_activations = xr.concat(activations, dim='presentation')
        # merging does not maintain stimulus order. the following orders again
        # assert that the order of model_activations is the same as stimulus_set
        assert np.all(model_activations[reset_column].values == stimulus_set[reset_column].values)
        # idx = [model_activations[reset_column].values.tolist().index(stimulus_id) for stimulus_id in
        #       [int(s[reset_column].values) for s in activations]]
        # assert len(set(idx)) == len(idx), "Found duplicate indices to order activations"
        # model_activations = model_activations[{'presentation': idx}]

        return model_activations

    def __call__(self,candidate, *args, **kwargs):
        # make sure word_ids are in increasing order and the same between target_assembly and model_activations
        stimulus_set = self._target_assembly['stimulus']
        # stimulus_set = stimulus_set.assign_coords({'sentence_id': ('presentation', stimulus_set.stimulus_id.values)})
        model_activations = self._read_words(candidate, stimulus_set, copy_columns=['stimuli_id', 'word_id'],
                                             average_sentence=False)
        model_activations = model_activations.groupby('stimulus_id').apply(lambda x: x.sortby('word_id'))
        # group model activation by stimulus_id and average over word_id
        model_activations = model_activations.groupby('stimulus_id').mean('presentation')
        target_assembly = self._target_assembly.groupby('stimulus_id').apply(lambda x: x.sortby('word_id'))
        # do the same with the target_assembly
        target_assembly = target_assembly.groupby('stimulus_id').mean('presentation')
        target_assembly = target_assembly.rename({'stimulus_id': 'presentation'})
        target_assembly = target_assembly.assign_coords(
            {'stimuli_id': ('presentation', target_assembly.presentation.values)})
        model_activations = model_activations.rename({'stimulus_id': 'presentation'})
        model_activations = model_activations.assign_coords(
            {'stimuli_id': ('presentation', model_activations.presentation.values)})
        assert np.all(model_activations['stimuli_id'].values == target_assembly['stimuli_id'].values)
        _logger.info('Scoring across electrodes')
        score = self.apply_metric(model_activations, target_assembly)
        raw_neuroids = apply_aggregate(lambda values: values.mean('split'), score.raw)

        score = aggregate_neuroid_scores(raw_neuroids, 'subject')
        score.attrs['raw'] = raw_neuroids
        score.attrs['ceiling'] = 1
        score.attrs['description'] = "per-neuroid no-ceiling-normalized score"
        return score


class _ANNSet1ECoGV2Benchmark:
    def __init__(self, identifier, metric, version='HighGamma_bipolar_gauss_zscore', type='language', threshold=0.05):
        self._identifier = identifier
        self._metric = metric
        assembly = self._load_assembly(version=version, type=type, threshold=threshold)
        self._target_assembly = assembly
        self.average_sentence = False
        self._ceiler = ExtrapolationCeiling(subject_column='subject')

    @property
    def ceiling(self):
        return self._ceiler(identifier=self.identifier, assembly=self._target_assembly, metric=self._metric)

    @property
    def identifier(self):
        return self._identifier

    def apply_metric(self, model_activations, target_assembly):
        return self._metric(model_activations, target_assembly)

    def _load_assembly(self, version='HighGamma_unipolar_gauss_zscore', type='language', threshold=0.05):
        file_id = Path(ANNfMRI_PARENT, f'ANNSet1_ECoG.{version}.pkl')
        assembly = pd.read_pickle(file_id.__str__())
        # find if ther eis repetition in  stim_name and if there are find them
        # and replace them with the same stim_name
        # drop 'sentence_repeat' and 'nonword condition
        # change neuroid_id to neuroid
        # add a neuroid dimension that goes from 0 to size of neuroid dimension
        assembly.assign_coords({'neuroid':('neuroid',np.arange(assembly.neuroid.size))})
        assembly = assembly[~assembly.Trial_condition.isin(['sentence_repeat', 'nonword'])]
        # do a sort based on sentence_id
        # this is to make sure that the assembly is ordered based on stimulus_id
        # and that the word_ids are in increasing order
        assert np.all(assembly['stim_type'].values == 'S')
        assembly = assembly.rename({'stim_id': 'sentence_id'})
        # drop Trial_onset and Trial_abs_onset
        # delete these because they messe up regression
        #assembly = assembly.drop(['Trial_id'], axis=1)
        #assembly = assembly.drop(['Trial_abs_id'], axis=1)
        #assembly = assembly.drop(['Trial_onset'], axis=1)
        #assembly = assembly.drop(['Trial_abs_onset'], axis=1)
        assembly = assembly.sortby(['sentence_id'])
        assembly = assembly.groupby('sentence_id').apply(lambda x: x.sortby('word_id'))
        # make sure there are only S in stim_type



        # define a new coordinate called sentence_id with the same values as stimulus_id
        # define a new coordiante stimuli_id that goes from 0 to size presentation dimension
        assembly = assembly.assign_coords({'stimulus_id': ('presentation', np.arange(assembly.sentence_id.size))})
        # make sure the assembly is ordered based on stimulus_id
        #assembly = assembly.sortby('stimulus_id')

        # this is to make sure that the assembly is ordered based on stimulus_id
        if type == 'language':
            s_v_n = assembly.s_v_n_ratio >= (1 - threshold)
        elif type == 'non-language':
            s_v_n = assembly.s_v_n_ratio < (1 - threshold)

        assembly = assembly.sel(neuroid=((s_v_n) & (assembly['electrode_valid'] == 1)).values)
        # assert there are no nans in assembly
        assert not np.isnan(assembly).any()

        assembly=assembly.dropna('neuroid')

        assembly_new = []
        for grp, sub in assembly.groupby('subject'):
            if sub.neuroid.size < 5:
                pass
            else:
                assembly_new.append(sub)
        assembly_new = xr.concat(assembly_new, dim='neuroid')

        assert not np.isnan(assembly_new).any()
        assembly = NeuroidAssembly(assembly_new)
        # add identifier to assembly
        thr_str = str(threshold).replace('.', '')
        assembly.attrs['identifier'] = f"ANNSet1_ECoG.{version}_{type}_thr_{thr_str}"

        sentenceID=assembly.sentence_id.values
        word_number=assembly.stimulus_id.values
        word_id= assembly.word_id.values
        sentence_words=assembly.string.values

        zipped_lst = list(zip(sentenceID, word_number, sentence_words,word_id))
        df_stimulus_set = StimulusSet(zipped_lst, columns=['sentence_id', 'stimulus_id', 'word','word_id'])
        df_stimulus_set.name = f"ANNSet1_ECoG"


        # stimulus_set = StimulusSet({'sentence_id': assembly['sentence_id'].values,
        #                             'stimulus_id': assembly['stimulus_id'].values,
        #                             'word': assembly['string'].values,
        #                             'word_id': assembly['word_id'].values})
        #                             #'sentence': assembly['stimulus'].values})
        # attach stimulus set as an attribute to the assembly
        # add name to stimulus set

        assembly.attrs['stimulus_set'] = df_stimulus_set
        return assembly

    def __call__(self, candidate, *args, **kwargs):
        stimulus_set = self._target_assembly.attrs['stimulus_set']
        model_activations = read_words(candidate, stimulus_set,
                                       average_sentence=False, copy_columns=['stimulus_id', 'word_id'])
        assert (model_activations['stimulus_id'].values == self._target_assembly['stimulus_id'].values).all()
        # zscore model_activations along presentation
        _logger.info('Scoring across electrodes')
        score = self.apply_metric(model_activations, self._target_assembly)
        return score

    def _apply_cross(self, source_assembly, cross_assembly):
        cross_assembly = cross_assembly.dropna('neuroid')  # some subjects have only done one experiment
        source_assembly = source_assembly.dropna('neuroid')  # only relevant when running audio-visual self as "model"
        # assert len(cross_assembly['presentation']) in [200]
        assert not np.isnan(cross_assembly).any()
        source_assembly = source_assembly[{'presentation': [stimulus_id in cross_assembly['stimulus_id'].values
                                                            for stimulus_id in source_assembly['stimulus_id'].values]}]
        return self._single_metric(source_assembly, cross_assembly)

class ANNSet1ECoGV2Encoding(_ANNSet1ECoGV2Benchmark):
    def __init__(self,identifier):
        regression = linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id'))  # word
        correlation = pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')) # word
        metric = CrossRegressedCorrelation(regression=regression, correlation=correlation,
                                           crossvalidation_kwargs=dict(splits=10, kfold=True, split_coord='stimulus_id',
                                                                       stratification_coord='sentence_id'))

        super(ANNSet1ECoGV2Encoding, self).__init__(identifier=identifier, metric=metric,type='language',version='HighGamma_bipolar_gauss_zscore',threshold=0.05)

    @property
    def ceiling(self):
        return super(ANNSet1ECoGEncoding, self).ceiling
class ANNSet1ECoGRidgeEncoding(_ANNSet1ECoGV2Benchmark):
    def __init__(self,identifier):
        regression = rgcv_linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id'))  # word
        correlation = pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')) # word
        metric = CrossRegressedCorrelation(regression=regression, correlation=correlation,
                                           crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord='stimulus_id',
                                                                       stratification_coord='sentence_id'))

        super(ANNSet1ECoGRidgeEncoding, self).__init__(identifier=identifier, metric=metric,type='language',version='HighGamma_bipolar_gauss_zscore',threshold=0.05)

    @property
    def ceiling(self):
        return super(ANNSet1ECoGRidgeEncoding, self).ceiling

class ANNSet1ECoGLinearEncoding(_ANNSet1ECoGV2Benchmark):
    def __init__(self,identifier):
        regression = linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id'))  # word
        correlation = pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id'))  # word
        metric = CrossRegressedCorrelation(regression=regression, correlation=correlation,
                                           crossvalidation_kwargs=dict(splits=10, kfold=True,
                                                                       split_coord='stimulus_id',
                                                                       stratification_coord='sentence_id'))
        super(ANNSet1ECoGLinearEncoding, self).__init__(identifier=identifier, metric=metric,type='language',version='HighGamma_bipolar_gauss_zscore',threshold=0.05)

    #@property
    def ceiling(self):
        return super(ANNSet1ECoGLinearEncoding, self).ceiling

class ANNSet1ECoGBipGaussEncoding(ANNSet1ECoGLinearEncoding):
    def _load_assembly(self,version='HighGamma_bipolar_gauss_subs_21_only_ANN',type='language',threshold=0.05):
        return super()._load_assembly(version='HighGamma_bipolar_gauss_subs_21_only_ANN',type='language',threshold=0.05)

class ANNSet1ECoGBipGaussStrictEncoding(ANNSet1ECoGLinearEncoding):
    def _load_assembly(self,version='HighGamma_bipolar_gauss_subs_21_only_ANN',type='language',threshold=0.01):
        return super()._load_assembly(version='HighGamma_bipolar_gauss_subs_21_only_ANN',type='language',threshold=0.01)
class ANNSet1ECoGBipBandEncoding(ANNSet1ECoGLinearEncoding):
    def _load_assembly(self,version='HighGamma_bipolar_bandpass_subs_21_only_ANN',type='language',threshold=0.05):
        return super()._load_assembly(version='HighGamma_bipolar_bandpass_subs_21_only_ANN',type='language',threshold=0.05)

class ANNSet1ECoGBipBandStrictEncoding(ANNSet1ECoGLinearEncoding):
    def _load_assembly(self,version='HighGamma_bipolar_bandpass_subs_21_only_ANN',type='language',threshold=0.01):
        return super()._load_assembly(version='HighGamma_bipolar_bandpass_subs_21_only_ANN',type='language',threshold=0.01)
class ANNSet1ECoGUniGaussEncoding(ANNSet1ECoGLinearEncoding):
    def _load_assembly(self,version='HighGamma_unipolar_gauss_subs_21_only_ANN',type='language',threshold=0.05):
        return super()._load_assembly(version='HighGamma_unipolar_gauss_subs_21_only_ANN',type='language',threshold=0.05)

class ANNSet1ECoGUniGaussStrictEncoding(ANNSet1ECoGLinearEncoding):
    def _load_assembly(self,version='HighGamma_unipolar_gauss_subs_21_only_ANN',type='language',threshold=0.01):
        return super()._load_assembly(version='HighGamma_unipolar_gauss_subs_21_only_ANN',type='language',threshold=0.01)
class ANNSet1ECoGUniBandEncoding(ANNSet1ECoGLinearEncoding):
    def _load_assembly(self,version='HighGamma_unipolar_bandpass_subs_21_only_ANN',type='language',threshold=0.05):
        return super()._load_assembly(version='HighGamma_unipolar_bandpass_subs_21_only_ANN',type='language',threshold=0.05)

class ANNSet1ECoGUniBandStrictEncoding(ANNSet1ECoGLinearEncoding):
    def _load_assembly(self,version='HighGamma_unipolar_bandpass_subs_21_only_ANN',type='language',threshold=0.01):
        return super()._load_assembly(version='HighGamma_unipolar_bandpass_subs_21_only_ANN',type='language',threshold=0.01)

# do a version for subjects shared with langloc
class ANNSet1ECoGBipGaussSharedLanglocEncoding(ANNSet1ECoGLinearEncoding):
    def _load_assembly(self,version='HighGamma_bipolar_gauss_subs_16_shared_LangLoc_ANN',type='language',threshold=0.05):
        return super()._load_assembly(version='HighGamma_bipolar_gauss_subs_16_shared_LangLoc_ANN',type='language',threshold=0.05)

class ANNSet1ECoGBipGaussSharedLanglocStrictEncoding(ANNSet1ECoGLinearEncoding):
    def _load_assembly(self,version='HighGamma_bipolar_gauss_subs_16_shared_LangLoc_ANN',type='language',threshold=0.01):
        return super()._load_assembly(version='HighGamma_bipolar_gauss_subs_16_shared_LangLoc_ANN',type='language',threshold=0.01)
class ANNSet1ECoGBipBandSharedLanglocEncoding(ANNSet1ECoGLinearEncoding):
    def _load_assembly(self,version='HighGamma_bipolar_bandpass_subs_16_shared_LangLoc_ANN',type='language',threshold=0.05):
        return super()._load_assembly(version='HighGamma_bipolar_bandpass_subs_16_shared_LangLoc_ANN',type='language',threshold=0.05)

class ANNSet1ECoGBipBandSharedLanglocStrictEncoding(ANNSet1ECoGLinearEncoding):
    def _load_assembly(self,version='HighGamma_bipolar_bandpass_subs_16_shared_LangLoc_ANN',type='language',threshold=0.01):
        return super()._load_assembly(version='HighGamma_bipolar_bandpass_subs_16_shared_LangLoc_ANN',type='language',threshold=0.01)
class ANNSet1ECoGUniGaussSharedLanglocEncoding(ANNSet1ECoGLinearEncoding):
    def _load_assembly(self,version='HighGamma_unipolar_gauss_subs_16_shared_LangLoc_ANN',type='language',threshold=0.05):
        return super()._load_assembly(version='HighGamma_unipolar_gauss_subs_16_shared_LangLoc_ANN',type='language',threshold=0.05)

class ANNSet1ECoGUniGaussSharedLanglocStrictEncoding(ANNSet1ECoGLinearEncoding):
    def _load_assembly(self,version='HighGamma_unipolar_gauss_subs_16_shared_LangLoc_ANN',type='language',threshold=0.01):
        return super()._load_assembly(version='HighGamma_unipolar_gauss_subs_16_shared_LangLoc_ANN',type='language',threshold=0.01)
class ANNSet1ECoGUniBandSharedLanglocEncoding(ANNSet1ECoGLinearEncoding):
    def _load_assembly(self,version='HighGamma_unipolar_bandpass_subs_16_shared_LangLoc_ANN',type='language',threshold=0.05):
        return super()._load_assembly(version='HighGamma_unipolar_bandpass_subs_16_shared_LangLoc_ANN',type='language',threshold=0.05)

class ANNSet1ECoGUniBandSharedLanglocStrictEncoding(ANNSet1ECoGLinearEncoding):
    def _load_assembly(self,version='HighGamma_unipolar_bandpass_subs_16_shared_LangLoc_ANN',type='language',threshold=0.01):
        return super()._load_assembly(version='HighGamma_unipolar_bandpass_subs_16_shared_LangLoc_ANN',type='language',threshold=0.01)
class _LanglocECOG:
    """
    data source:
    """

    def __init__(self, identifier, metric,version='HighGamma_bipolar_gauss_zscore_subs_17',type='language',threshold=0.05):
        self._identifier = identifier
        assembly = self._load_assembly(version=version,type=type,threshold=threshold)
        self._target_assembly = assembly
        self._metric = metric
        self._average_sentence = False
        self._ceiler = ExtrapolationCeiling(subject_column='subject')
        self._few_sub_ceiler= FewSubjectExtrapolation(subject_column='subject',extrapolation_dimension='neuroid',num_bootstraps=100,post_process=None)


    @property
    def identifier(self):
        return self._identifier

    @property
    def electrode_ceiling(self):
        return self._electrode_ceiler(identifier=self.identifier, assembly=self._target_assembly, metric=self._metric)

    @property
    def ceiling(self):
        return self._ceiler(identifier=self.identifier, assembly=self._target_assembly, metric=self._metric)
        #return self._few_sub_ceiler(identifier=self.identifier, assembly=self._target_assembly, metric=self._metric)

    @property
    def ceiling_estimate(self):
        return self._few_sub_ceiler(identifier=self.identifier, assembly=self._target_assembly, metric=self._metric)
        # return self._few_sub_ceiler(identifier=self.identifier, assembly=self._target_assembly, metric=self._metric)

    def ceiling_normalize_(self,raw_score: Score, ceiling: Score) -> Score:
        # normalize by ceiling, but not above 1
        score = raw_score / ceiling
        score.attrs['raw'] = raw_score
        score.attrs['ceiling'] = ceiling
        if score > 1:
            overshoot_value = score.item()
            # ideally we would just update the value, but I could not figure out how to update a scalar DataArray
            attrs = score.attrs
            score = type(score)(1, coords=score.coords, dims=score.dims)
            score.attrs = attrs
            score.attrs['overshoot'] = overshoot_value
        return score

    def ceiling_normalize(self, score):
        raw_neuroids = apply_aggregate(lambda values: values.mean('split'), score.raw)
        score = ceil_neuroids(raw_neuroids, self.ceiling, subject_column='subject')
        return score

    def _load_assembly(self,version,type,threshold):
        file_id=Path(ANNfMRI_PARENT,f'LangLoc_ECoG.{version}.pkl')
        assembly_raw = pd.read_pickle(file_id.__str__())
        # get only S from stim_type in assembly_raw
        assembly = assembly_raw[{'presentation': assembly_raw['stim_type'] == 'S'}]
        # make sure stim_id is sorted
        # rename stim_id as stimulus_id in assembly
        assembly = assembly.rename({'stim_id': 'sentence_id'})
        # this is to make sure that the assembly is ordered based on stimulus_id
        # and that the word_ids are in increasing order
        assembly = assembly.groupby('sentence_id').apply(lambda x: x.sortby('word_id'))
        # define a new coordinate called sentence_id with the same values as stimulus_id
        # define a new coordiante stimuli_id that goes from 0 to size presentation dimension
        assembly = assembly.assign_coords({'stimulus_id': ('presentation', np.arange(assembly.sentence_id.size))})
        # select electrdoes that are are s_v_n_ratio and are in electrode_valid
        if type=='language':
            s_v_n=assembly.s_v_n_ratio>=(1-threshold)
        elif type=='non-language':
            s_v_n=assembly.s_v_n_ratio<(1-threshold)
        assembly = assembly[{'neuroid': (s_v_n) & (assembly['electrode_valid'] == 1)}]
        # remove subject that have less than 5 electrodes
        assembly_new=[]
        for grp, sub in assembly.groupby('subject'):
            if sub.neuroid.size < 5:
                pass
            else:
                assembly_new.append(sub)

        assembly_new=xr.concat(assembly_new, dim='neuroid')

        # check if there is nan in assembly_new
        assert not np.isnan(assembly_new).any()
        # do a zscore on the assembly along neuroid
        #assembly_new = (assembly_new - assembly_new.mean(dim='presentation')) / assembly_new.std(dim='presentation')
        # count number of electrodes with s_v_n_ratio > 0.99
        # print(f"Number of electrodes with s_v_n_ratio > 0.99: {np.sum(assembly['s_v_n_ratio'] > 0.99)}")
        # make a neural assembly from xarray assembly

        assembly = NeuroidAssembly(assembly_new)
        sentenceID=assembly.sentence_id.values
        word_number=assembly.stimulus_id.values
        word_id= assembly.word_id.values
        sentence_words=assembly.string.values

        zipped_lst = list(zip(sentenceID, word_number, sentence_words,word_id))
        df_stimulus_set = StimulusSet(zipped_lst, columns=['sentence_id', 'stimulus_id', 'word','word_id'])
        df_stimulus_set.name = 'LangLoc_ECoG'
        # construct stimulus set for the assmebly


        assembly.attrs['stimulus_set'] = df_stimulus_set

        thr_str=str(threshold).replace('.','')
        assembly.attrs['identifier'] = f"LangLoc_ECoG.{version}_{type}_thr_{thr_str}"

        return assembly

    def apply_metric(self, model_activations, target_assembly):
        return self._metric(model_activations, target_assembly)

    def __call__(self, candidate, *args, **kwargs):
        _logger.info('Computing activations')
        stimulus_set = self._target_assembly.attrs['stimulus_set']
        model_activations = read_words(candidate, stimulus_set,
                                       average_sentence=self._average_sentence, copy_columns=['stimulus_id','word_id'])

        #model_activations = self._read_words(candidate, stimulus_set, copy_columns=['stimuli_id', 'word_id'])
        # make sure word_ids are in increasing order and the same between target_assembly and model_activations
        # assert model_activation
        assert (model_activations['stimulus_id'].values == self._target_assembly['stimulus_id'].values).all()
        # zscore model_activations along presentation
        #model_activations = (model_activations - model_activations.mean(dim='presentation')) / model_activations.std('presentation')
        # make sure the model_activations and target_assembly have the same number of words
        _logger.info('Scoring across electrodes')
        score = self.apply_metric(model_activations, self._target_assembly)
        #score = self.ceiling_normalize(score)
        return score

    def _apply_cross(self, source_assembly, cross_assembly):
        cross_assembly = cross_assembly.dropna('neuroid')  # some subjects have only done one experiment
        source_assembly = source_assembly.dropna('neuroid')  # only relevant when running audio-visual self as "model"
        #assert len(cross_assembly['presentation']) in [200]
        assert not np.isnan(cross_assembly).any()
        source_assembly = source_assembly[{'presentation': [stimulus_id in cross_assembly['stimulus_id'].values
                                                            for stimulus_id in source_assembly['stimulus_id'].values]}]
        return self._single_metric(source_assembly, cross_assembly)


class LangLocECoGEncoding(_LanglocECOG):
    def __init__(self, identifier):
        regression = linear_regression(xarray_kwargs=dict(stimulus_coord='stimuli_id'))  # word
        correlation = pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimuli_id'))  # word
        metric = CrossRegressedCorrelation(regression=regression, correlation=correlation,
                                           crossvalidation_kwargs=dict(splits=10, kfold=True, split_coord='stimuli_id',
                                                                       stratification_coord='stimulus_id'))

        super(LangLocECoGEncoding, self).__init__(identifier=identifier, metric=metric,type='language',version='HighGamma_bipolar_gauss_zscore_subs_17',threshold=0.05)

    def ceiling(self):
        return super(_LanglocECOG, self).ceiling

    def ceiling_estimate(self):
        return super(_LanglocECOG, self).ceiling_estimate


class LangLocECoGV2Encoding(_LanglocECOG):
        def __init__(self,identifier,**kwargs):
            regression = linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id'))  # word
            correlation = pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id'))  # word
            split_coord='stimulus_id'
            stratification_coord='sentence_id'
            crossvalidation_kwargs=dict(splits=10, kfold=True, split_coord=split_coord,stratification_coord=stratification_coord)
            metric = CrossRegressedCorrelation(regression=regression, correlation=correlation,
                                               crossvalidation_kwargs=crossvalidation_kwargs)
            super(LangLocECoGV2Encoding, self).__init__(identifier=identifier, metric=metric, type='language',
                                                      version='HighGamma_bipolar_gauss_zscore_subs_17', threshold=0.05)

        def ceiling(self):
            return super(_LanglocECOG, self).ceiling

        def ceiling_estimate(self):
            return super(_LanglocECOG, self).ceiling_estimate


class LangLocECoGRidgeEncoding(_LanglocECOG):
    def __init__(self, identifier, **kwargs):
        regression = rgcv_linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id'))  # word
        correlation = pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id'))  # word
        metric = CrossRegressedCorrelation(regression=regression, correlation=correlation,
                                           crossvalidation_kwargs=dict(splits=10, kfold=True,
                                                                       split_coord='stimulus_id',
                                                                       stratification_coord='sentence_id'))
        super(LangLocECoGRidgeEncoding, self).__init__(identifier=identifier, metric=metric, type='language',
                                                    version='HighGamma_bipolar_gamma_zscore_subs_17', threshold=0.05)

    def ceiling(self):
        return super(LangLocECoGRidgeEncoding, self).ceiling

    def ceiling_estimate(self):
        return super(LangLocECoGRidgeEncoding, self).ceiling_estimate

class LangLocECoGLinearEncoding(_LanglocECOG):
    def __init__(self, identifier, **kwargs):
        regression = linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id'))  # word
        correlation = pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id'))  # word
        metric = CrossRegressedCorrelation(regression=regression, correlation=correlation,
                                           crossvalidation_kwargs=dict(splits=10, kfold=True,
                                                                       split_coord='stimulus_id',
                                                                       stratification_coord='sentence_id'))
        super(LangLocECoGLinearEncoding, self).__init__(identifier=identifier, metric=metric, type='language',
                                                    version='HighGamma_bipolar_gamma_zscore_subs_17', threshold=0.05)

    def ceiling(self):
        return super(LangLocECoGLinearEncoding, self).ceiling

    def ceiling_estimate(self):
        return super(LangLocECoGLinearEncoding, self).ceiling_estimate

class LangLocECoGBipGaussEncoding(LangLocECoGLinearEncoding):
    def _load_assembly(self,version='HighGamma_bipolar_gauss_subs_28_only_LangLoc',type='language',threshold=0.05):
        return super()._load_assembly(version='HighGamma_bipolar_gauss_subs_28_only_LangLoc',type='language',threshold=0.05)

class LangLocECoGBipGaussStrictEncoding(LangLocECoGLinearEncoding):
    def _load_assembly(self,version='HighGamma_bipolar_gauss_subs_28_only_LangLoc',type='language',threshold=0.01):
        return super()._load_assembly(version='HighGamma_bipolar_gauss_subs_28_only_LangLoc',type='language',threshold=0.01)
class LangLocECoGUniGaussEncoding(LangLocECoGLinearEncoding):
    def _load_assembly(self,version='HighGamma_unipolar_gauss_subs_28_only_LangLoc',type='language',threshold=0.05):
        return super()._load_assembly(version='HighGamma_unipolar_gauss_subs_28_only_LangLoc',type='language',threshold=0.05)

class LangLocECoGUniGaussStrictEncoding(LangLocECoGLinearEncoding):
    def _load_assembly(self,version='HighGamma_unipolar_gauss_subs_28_only_LangLoc',type='language',threshold=0.01):
        return super()._load_assembly(version='HighGamma_unipolar_gauss_subs_28_only_LangLoc',type='language',threshold=0.01)

class LangLocECoGBipGaussZSEncoding(LangLocECoGLinearEncoding):
    def _load_assembly(self,version='HighGamma_bipolar_gauss_zscore_subs_28_only_LangLoc',type='language',threshold=0.05):
        return super()._load_assembly(version='HighGamma_bipolar_gauss_zscore_subs_28_only_LangLoc',type='language',threshold=0.05)

class LangLocECoGBipGaussZSStrictEncoding(LangLocECoGLinearEncoding):
    def _load_assembly(self,version='HighGamma_bipolar_gauss_zscore_subs_28_only_LangLoc',type='language',threshold=0.01):
        return super()._load_assembly(version='HighGamma_bipolar_gauss_zscore_subs_28_only_LangLoc',type='language',threshold=0.01)
class LangLocECoGUniGaussZSEncoding(LangLocECoGLinearEncoding):
    def _load_assembly(self,version='HighGamma_unipolar_gauss_zscore_subs_28_only_LangLoc',type='language',threshold=0.05):
        return super()._load_assembly(version='HighGamma_unipolar_gauss_zscore_subs_28_only_LangLoc',type='language',threshold=0.05)

class LangLocECoGUniGaussZSStrictEncoding(LangLocECoGLinearEncoding):
    def _load_assembly(self,version='HighGamma_unipolar_gauss_zscore_subs_28_only_LangLoc',type='language',threshold=0.01):
        return super()._load_assembly(version='HighGamma_unipolar_gauss_zscore_subs_28_only_LangLoc',type='language',threshold=0.01)
class LangLocECoGBipBandEncoding(LangLocECoGLinearEncoding):
    def _load_assembly(self,version='HighGamma_bipolar_bandpass_subs_28_only_LangLoc',type='language',threshold=0.05):
        return super()._load_assembly(version='HighGamma_bipolar_bandpass_subs_28_only_LangLoc',type='language',threshold=0.05)

class LangLocECoGBipBandStrictEncoding(LangLocECoGLinearEncoding):
    def _load_assembly(self,version='HighGamma_bipolar_bandpass_subs_28_only_LangLoc',type='language',threshold=0.01):
        return super()._load_assembly(version='HighGamma_bipolar_bandpass_subs_28_only_LangLoc',type='language',threshold=0.01)
class LangLocECoGUniBandEncoding(LangLocECoGLinearEncoding):
    def _load_assembly(self,version='HighGamma_unipolar_bandpass_subs_28_only_LangLoc',type='language',threshold=0.05):
        return super()._load_assembly(version='HighGamma_unipolar_bandpass_subs_28_only_LangLoc',type='language',threshold=0.05)

class LangLocECoGUniBandStrictEncoding(LangLocECoGLinearEncoding):
    def _load_assembly(self,version='HighGamma_unipolar_bandpass_subs_28_only_LangLoc',type='language',threshold=0.01):
        return super()._load_assembly(version='HighGamma_unipolar_bandpass_subs_28_only_LangLoc',type='language',threshold=0.01)

class LangLocECoGBipGaussSharedANNEncoding(LangLocECoGLinearEncoding):
    def _load_assembly(self,version='HighGamma_bipolar_gauss_subs_16_shared_LangLoc_ANN',type='language',threshold=0.05):
        return super()._load_assembly(version='HighGamma_bipolar_gauss_subs_16_shared_LangLoc_ANN',type='language',threshold=0.05)

class LangLocECoGBipGaussSharedANNStrictEncoding(LangLocECoGLinearEncoding):
    def _load_assembly(self,version='HighGamma_bipolar_gauss_subs_16_shared_LangLoc_ANN',type='language',threshold=0.01):
        return super()._load_assembly(version='HighGamma_bipolar_gauss_subs_16_shared_LangLoc_ANN',type='language',threshold=0.01)
class LangLocECoGUniGaussSharedANNEncoding(LangLocECoGLinearEncoding):
    def _load_assembly(self,version='HighGamma_unipolar_gauss_subs_16_shared_LangLoc_ANN',type='language',threshold=0.05):
        return super()._load_assembly(version='HighGamma_unipolar_gauss_subs_16_shared_LangLoc_ANN',type='language',threshold=0.05)

class LangLocECoGUniGaussSharedANNStrictEncoding(LangLocECoGLinearEncoding):
    def _load_assembly(self,version='HighGamma_unipolar_gauss_subs_16_shared_LangLoc_ANN',type='language',threshold=0.01):
        return super()._load_assembly(version='HighGamma_unipolar_gauss_subs_16_shared_LangLoc_ANN',type='language',threshold=0.01)
class LangLocECoGBipBandSharedANNEncoding(LangLocECoGLinearEncoding):
    def _load_assembly(self,version='HighGamma_bipolar_bandpass_subs_16_shared_LangLoc_ANN',type='language',threshold=0.05):
        return super()._load_assembly(version='HighGamma_bipolar_bandpass_subs_16_shared_LangLoc_ANN',type='language',threshold=0.05)

class LangLocECoGBipBandSharedANNStrictEncoding(LangLocECoGLinearEncoding):
    def _load_assembly(self,version='HighGamma_bipolar_bandpass_subs_16_shared_LangLoc_ANN',type='language',threshold=0.01):
        return super()._load_assembly(version='HighGamma_bipolar_bandpass_subs_16_shared_LangLoc_ANN',type='language',threshold=0.01)
class LangLocECoGUniBandSharedANNEncoding(LangLocECoGLinearEncoding):
    def _load_assembly(self,version='HighGamma_unipolar_bandpass_subs_16_shared_LangLoc_ANN',type='language',threshold=0.05):
        return super()._load_assembly(version='HighGamma_unipolar_bandpass_subs_16_shared_LangLoc_ANN',type='language',threshold=0.05)

class LangLocECoGUniBandSharedANNStrictEncoding(LangLocECoGLinearEncoding):
    def _load_assembly(self,version='HighGamma_unipolar_bandpass_subs_16_shared_LangLoc_ANN',type='language',threshold=0.01):
        return super()._load_assembly(version='HighGamma_unipolar_bandpass_subs_16_shared_LangLoc_ANN',type='language',threshold=0.01)
class LangLocECoGSampleEncoding(_LanglocECOG):
    def __init__(self, identifier, **kwargs):
        regression = linear_regression(xarray_kwargs=dict(stimulus_coord='stimuli_id'))  # word
        correlation = pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimuli_id'))  # word
        metric = CrossRegressedCorrelation(regression=regression, correlation=correlation,
                                           crossvalidation_kwargs=dict(splits=5, kfold=True,
                                                                       split_coord='stimuli_id',
                                                                       stratification_coord='stimulus_id'))
        super(LangLocECoGSampleEncoding, self).__init__(identifier=identifier, metric=metric, type='language',
                                                    version='HighGamma_bipolar_gauss_zscore_subs_3', threshold=0.05)

class LangLocECoGSentenceEncoding(_LanglocECOG):
    def __init__(self, identifier, **kwargs):
        regression = linear_regression(xarray_kwargs=dict(stimulus_coord='stimuli_id'))  # sentence
        correlation = pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimuli_id'))  # sentence
        metric = CrossRegressedCorrelation(regression=regression, correlation=correlation,
                                           crossvalidation_kwargs=dict(splits=5, kfold=True,
                                                                       split_coord='stimuli_id',
                                                                       stratification_coord=None))
        super(LangLocECoGSentenceEncoding, self).__init__(identifier=identifier, metric=metric, type='language',
                                                    version='HighGamma_bipolar_gauss_zscore_subs_17', threshold=0.05)

    def _read_words(self, candidate, stimulus_set, reset_column='stimulus_id', copy_columns=(), average_sentence=True):
        """
        Pass a `stimulus_set` through a model `candidate`.
        In contrast to the `listen_to` function, this function operates on a word-based `stimulus_set`.
        """
        # Input: stimulus_set = pandas df, col 1 with sentence ID and 2nd col as word.
        activations = []
        for i, reset_id in enumerate(ordered_set(stimulus_set[reset_column].values)):
            part_stimuli = stimulus_set[stimulus_set[reset_column] == reset_id]
            # stimulus_ids = part_stimuli['stimulus_id']
            sentence_stimuli = StimulusSet({'sentence': part_stimuli.values[0],
                                            reset_column: list(set(part_stimuli[reset_column].values))})
            sentence_stimuli.name = f"{self._target_assembly.identifier}-ave-{average_sentence}-{reset_id}"
            sentence_activations = candidate(stimuli=sentence_stimuli, average_sentence=average_sentence)
            for column in copy_columns:
                sentence_activations[column] = ('presentation', part_stimuli[column])

            activations.append(sentence_activations)

        # model_activations = merge_data_arrays(activations)
        model_activations = xr.concat(activations, dim='presentation')
        # merging does not maintain stimulus order. the following orders again
        # assert that the order of model_activations is the same as stimulus_set
        assert np.all(model_activations[reset_column].values == stimulus_set[reset_column].values)
        # idx = [model_activations[reset_column].values.tolist().index(stimulus_id) for stimulus_id in
        #       [int(s[reset_column].values) for s in activations]]
        # assert len(set(idx)) == len(idx), "Found duplicate indices to order activations"
        # model_activations = model_activations[{'presentation': idx}]

        return model_activations

    def __call__(self, candidate, *args, **kwargs):
        stimulus_set = self._target_assembly['stimulus']
        model_activations = self._read_words(candidate, stimulus_set, copy_columns=['stimuli_id', 'word_id'],
                                             average_sentence=False)
        model_activations = model_activations.groupby('stimulus_id').apply(lambda x: x.sortby('word_id'))
        model_activations = model_activations.groupby('stimulus_id').mean('presentation')
        target_assembly = self._target_assembly.groupby('stimulus_id').apply(lambda x: x.sortby('word_id'))
        target_assembly = target_assembly.groupby('stimulus_id').mean('presentation')
        target_assembly = target_assembly.rename({'stimulus_id': 'presentation'})
        target_assembly = target_assembly.assign_coords(
            {'stimuli_id': ('presentation', target_assembly.presentation.values)})
        model_activations = model_activations.rename({'stimulus_id': 'presentation'})
        model_activations = model_activations.assign_coords(
            {'stimuli_id': ('presentation', model_activations.presentation.values)})
        _logger.info('Scoring across electrodes')
        assert np.all(model_activations['stimuli_id'].values == target_assembly['stimuli_id'].values)
        _logger.info('Scoring across electrodes')
        score = self.apply_metric(model_activations, target_assembly)
        raw_neuroids = apply_aggregate(lambda values: values.mean('split'), score.raw)
        score = aggregate_neuroid_scores(raw_neuroids, 'subject')
        score.attrs['raw'] = raw_neuroids
        score.attrs['ceiling'] = 1
        score.attrs['description'] = "per-neuroid no-ceiling-normalized score"
        return score
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
    ceiling_center=ceiling[(ceiling.aggregation=='center').values]
    score = consistency(aggregate_raw, ceiling_center)
    score.attrs['raw'] = aggregate_raw
    score.attrs['ceiling'] = ceiling
    score.attrs['description'] = "ceiling-normalized score"
    return score


def consistency(score, ceiling):
    return score / ceiling


benchmark_pool = [
    # primary benchmarks
    ('Pereira2018-encoding', PereiraEncoding),
    ('Pereira2018-lang-encoding', PereiraLangEncoding),
    ('Pereira2018-RDM', PereiraRDM),
    ('Pereira2018-max-encoding', PereiraSamplerMaxEncoding),
    ('Pereira2018-max-V2-encoding', PereiraSamplerMaxV2Encoding),
    ('Pereira2018-min-V2-encoding', PereiraSamplerMinV2Encoding),
    ('Pereira2018-rand-V2-encoding', PereiraSamplerRandV2Encoding),
    ('Pereira2018-norm-encoding',PereiraNormalizedEncoding),
    ('Pereira2018-norm-v2-encoding',PereiraNormalizedEncoding_V2),
    ('Pereira2018-norm-sentence-encoding',PereiraNormalizedSentenceEncoding),
    ('Pereira2023aud-sent-RidgeEncoding', Pereira2023audSentRidgeEncoding),
    ('Pereira2023aud-sent-sentence-Encoding', Pereira2023audSentSentenceEncoding),
    ('Pereira2023aud-pass-passage-RidgeEncoding', Pereira2023audPassPassageRidgeEncoding),
    ('Pereira2023aud-pass-passage-Encoding', Pereira2023audPassPassageEncoding),
    ('Pereira2023aud-pass-passage-sample-Encoding', Pereira2023audPassPassageSampleEncoding),
    ('Pereira2023aud-pass-passage-sample-RidgeEncoding', Pereira2023audPassPassageSampleRidgeEncoding),
    ('Pereira2023aud-pass-sentence-RidgeEncoding', Pereira2023audPassSentenceRidgeEncoding),
    ('Pereira2023aud-pass-sentence-Encoding', Pereira2023audPassSentenceEncoding),
    #('Pereira2023aud-sent-sentence-RidgeEncoding', Pereira2023audSentSentenceRidgeEncoding),

    ('Pereira2023aud-sent-passage-RidgeEncoding', Pereira2023audSentPassageRidgeEncoding),
    ('Pereira2023aud-sent-passage-Encoding', Pereira2023audSentPassageEncoding),

    ('DsParametricfMRI-shared-90-max-encoding', DsParametricfMRISharedMaxEncoding),
    ('DsParametricfMRI-full-90-max-encoding', DsParametricfMRIFullMaxEncoding),
    ('DsParametricfMRI-shared-70-max-encoding', DsParametricfMRIShared70MaxEncoding),
    ('DsParametricfMRI-shared-90-min-encoding', DsParametricfMRISharedMinEncoding),
    ('DsParametricfMRI-full-90-min-encoding', DsParametricfMRIFullMinEncoding),
    ('DsParametricfMRI-shared-70-min-encoding', DsParametricfMRIShared70MinEncoding),
    ('DsParametricfMRI-shared-90-rand-encoding', DsParametricfMRISharedRandEncoding),
    ('DsParametricfMRI-full-90-rand-encoding', DsParametricfMRIFullRandEncoding),
    ('DsParametricfMRI-shared-70-rand-encoding', DsParametricfMRIShared70RandEncoding),
    ('DsParametricfMRI-shared-all-encoding', DsParametricfMRISharedAllEncoding),

    ('DsParametricfMRI-shared-max-RidgeEncoding', DsParametricfMRISharedMaxRidgeEncoding),
    ('DsParametricfMRI-shared-min-RidgeEncoding', DsParametricfMRISharedMinRidgeEncoding),
    ('DsParametricfMRI-shared-rand-RidgeEncoding', DsParametricfMRISharedRandRidgeEncoding),

    ('DsParametricfMRI-first-max-Encoding_aug2024', DsParametricfMRIFirstMaxEncoding),
    ('DsParametricfMRI-first-min-Encoding_aug2024', DsParametricfMRIFirstMinEncoding),
    ('DsParametricfMRI-first-rand-Encoding_aug2024', DsParametricfMRIFirstRandEncoding),

    ('DsParametricfMRI-second-max-Encoding_aug2024', DsParametricfMRISecondMaxEncoding),
    ('DsParametricfMRI-second-min-Encoding_aug2024', DsParametricfMRISecondMinEncoding),
    ('DsParametricfMRI-second-rand-Encoding_aug2024', DsParametricfMRISecondRandEncoding),

    ('DsParametricfMRI-first-reliable-max-Encoding_aug2024', DsParametricfMRIFirstReliableMaxEncoding),
    ('DsParametricfMRI-first-reliable-min-Encoding_aug2024', DsParametricfMRIFirstReliableMinEncoding),
    ('DsParametricfMRI-first-reliable-rand-Encoding_aug2024', DsParametricfMRIFirstReliableRandEncoding),

    ('DsParametricfMRI-second-reliable-max-Encoding_aug2024', DsParametricfMRISecondReliableMaxEncoding),
    ('DsParametricfMRI-second-reliable-min-Encoding_aug2024', DsParametricfMRISecondReliableMinEncoding),
    ('DsParametricfMRI-second-reliable-rand-Encoding_aug2024', DsParametricfMRISecondReliableRandEncoding),

    ('DsParametricfMRI-first-reliable-max-RidgeEncoding_aug2024', DsParametricfMRIFirstReliableMaxRidgeEncoding),
    ('DsParametricfMRI-first-reliable-min-RidgeEncoding_aug2024', DsParametricfMRIFirstReliableMinRidgeEncoding),
    ('DsParametricfMRI-first-reliable-rand-RidgeEncoding_aug2024', DsParametricfMRIFirstReliableRandRidgeEncoding),
    #
    ('DsParametricfMRI-second-reliable-max-RidgeEncoding_aug2024', DsParametricfMRISecondReliableMaxRidgeEncoding),
    ('DsParametricfMRI-second-reliable-min-RidgeEncoding_aug2024', DsParametricfMRISecondReliableMinRidgeEncoding),
    ('DsParametricfMRI-second-reliable-rand-RidgeEncoding_aug2024', DsParametricfMRISecondReliableRandRidgeEncoding),

    ('DsParametricfMRI-first-all-max-Encoding_aug2024', DsParametricfMRIFirstAllMaxEncoding),
    ('DsParametricfMRI-first-all-min-Encoding_aug2024', DsParametricfMRIFirstAllMinEncoding),
    ('DsParametricfMRI-first-all-rand-Encoding_aug2024', DsParametricfMRIFirstAllRandEncoding),

    ('DsParametricfMRI-second-all-max-Encoding_aug2024', DsParametricfMRISecondAllMaxEncoding),
    ('DsParametricfMRI-second-all-min-Encoding_aug2024', DsParametricfMRISecondAllMinEncoding),
    ('DsParametricfMRI-second-all-rand-Encoding_aug2024', DsParametricfMRISecondAllRandEncoding),

    ('DsParametricfMRI-80-first-max-Encoding_aug2024', DsParametricfMRI80FirstMaxEncoding),
    ('DsParametricfMRI-80-first-min-Encoding_aug2024', DsParametricfMRI80FirstMinEncoding),
    ('DsParametricfMRI-80-first-rand-Encoding_aug2024', DsParametricfMRI80FirstRandEncoding),

    ('DsParametricfMRI-80-second-max-Encoding_aug2024', DsParametricfMRI80SecondMaxEncoding),
    ('DsParametricfMRI-80-second-min-Encoding', DsParametricfMRI80SecondMinEncoding),
    ('DsParametricfMRI-80-second-rand-Encoding', DsParametricfMRI80SecondRandEncoding),


    ('DsParametricfMRI-first-max-StrictEncoding', DsParametricfMRIFirstMaxStrictEncoding),
    ('DsParametricfMRI-first-min-StrictEncoding', DsParametricfMRIFirstMinStrictEncoding),
    ('DsParametricfMRI-first-rand-StrictEncoding', DsParametricfMRIFirstRandStrictEncoding),

    ('DsParametricfMRI-second-max-StrictEncoding', DsParametricfMRISecondMaxStrictEncoding),
    ('DsParametricfMRI-second-min-StrictEncoding', DsParametricfMRISecondMinStrictEncoding),
    ('DsParametricfMRI-second-rand-StrictEncoding', DsParametricfMRISecondRandStrictEncoding),

    ('DsParametricfMRI-first-max-RidgeEncoding_aug2024', DsParametricfMRIFirstMaxRidgeEncoding),
    ('DsParametricfMRI-first-min-RidgeEncoding_aug2024', DsParametricfMRIFirstMinRidgeEncoding),
    ('DsParametricfMRI-first-rand-RidgeEncoding_aug2024', DsParametricfMRIFirstRandRidgeEncoding),

    ('DsParametricfMRI-second-max-RidgeEncoding_aug2024', DsParametricfMRISecondMaxRidgeEncoding),
    ('DsParametricfMRI-second-min-RidgeEncoding_aug2024', DsParametricfMRISecondMinRidgeEncoding),
    ('DsParametricfMRI-second-rand-RidgeEncoding_aug2024', DsParametricfMRISecondRandRidgeEncoding),

    # ('DsParametricfMRI_v1-max-RidgeEncoding', DsParametricfMRIMaxV1RidgeEncoding),
    # ('DsParametricfMRI_v1-min-RidgeEncoding', DsParametricfMRIMinV1RidgeEncoding),
    # ('DsParametricfMRI_v1-rand-RidgeEncoding', DsParametricfMRIRandV1RidgeEncoding),

    ('DsParametricfMRI_v2-max-RidgeEncoding', DsParametricfMRIMaxV2RidgeEncoding),
    ('DsParametricfMRI_v2-min-RidgeEncoding', DsParametricfMRIMinV2RidgeEncoding),
    ('DsParametricfMRI_v2-rand-RidgeEncoding', DsParametricfMRIRandV2RidgeEncoding),

    ('DsParametricfMRI_v3-max-RidgeEncoding', DsParametricfMRIMaxV3RidgeEncoding),
    ('DsParametricfMRI_v3-min-RidgeEncoding', DsParametricfMRIMinV3RidgeEncoding),
    ('DsParametricfMRI_v3-rand-RidgeEncoding', DsParametricfMRIRandV3RidgeEncoding),

    ('DsParametricfMRI-max-PLSEncoding', DsParametricfMRIMaxPLSEncoding),
    ('DsParametricfMRI-min-PLSEncoding', DsParametricfMRIMinPLSEncoding),
    ('DsParametricfMRI-rand-PLSEncoding', DsParametricfMRIRandPLSEncoding),

    ('DsParametricfMRI-Norm-max-RidgeEncoding',DsParametricfMRINormMaxRidgeEncoding),
    ('DsParametricfMRI-Norm-min-RidgeEncoding', DsParametricfMRINormMinRidgeEncoding),
    ('DsParametricfMRI-Norm-rand-RidgeEncoding', DsParametricfMRINormRandRidgeEncoding),

    ('Pereira2018-min-encoding', PereiraSamplerMinEncoding),
    ('Pereira2018-rand-encoding', PereiraSamplerRandEncoding),
    ('ANNSet1fMRI-encoding', ANNSet1fMRIEncoding),
    ('ANNSet1fMRI-wordForm-encoding',ANNSet1fMRIEncoding_V2),
    ('ANNSet1fMRI-best-encoding', ANNSet1fMRIBestEncoding),
    ('ANNSet1fMRI-v3-best-encoding', ANNSet1fMRIBestV3Encoding),
    ('ANNSet1fMRISentence-encoding', ANNSet1fMRISentenceEncoding),
    ('ANNSet1fMRISentence-wordForm-encoding', ANNSet1fMRISentenceEncoding_V2),
    ('ANNSet1ECoG-encoding', ANNSet1ECoGEncoding),
    ('ANNSet1ECoG-v2-encoding', ANNSet1ECoGV2Encoding),
    ('ANNSet1ECoG-Sentence-encoding', ANNSetECoGSentenceEncoding),

    ('ANNSet1ECoG-bip-gaus-Encoding', ANNSet1ECoGBipGaussEncoding),
    ('ANNSet1ECoG-bip-gaus-strict-Encoding', ANNSet1ECoGBipGaussStrictEncoding),

    ('ANNSet1ECoG-bip-band-Encoding', ANNSet1ECoGBipBandEncoding),
    ('ANNSet1ECoG-bip-band-strict-Encoding', ANNSet1ECoGBipBandStrictEncoding),

    ('ANNSet1ECoG-uni-gaus-Encoding', ANNSet1ECoGUniGaussEncoding),
    ('ANNSet1ECoG-uni-gaus-strict-Encoding', ANNSet1ECoGUniGaussStrictEncoding),

    ('ANNSet1ECoG-uni-band-Encoding', ANNSet1ECoGUniBandEncoding),
    ('ANNSet1ECoG-uni-band-strict-Encoding', ANNSet1ECoGUniBandStrictEncoding),

    ('ANNSet1ECoG-bip-gaus-shared-LangLoc-Encoding', ANNSet1ECoGBipGaussSharedLanglocEncoding),
    ('ANNSet1ECoG-bip-gaus-shared-LangLoc-strict-Encoding', ANNSet1ECoGBipGaussSharedLanglocStrictEncoding),

    ('ANNSet1ECoG-bip-band-shared-LangLoc-Encoding', ANNSet1ECoGBipBandSharedLanglocEncoding),
    ('ANNSet1ECoG-bip-band-shared-LangLoc-strict-Encoding', ANNSet1ECoGBipBandSharedLanglocStrictEncoding),

    ('ANNSet1ECoG-uni-gaus-shared-LangLoc-Encoding', ANNSet1ECoGUniGaussSharedLanglocEncoding),
    ('ANNSet1ECoG-uni-gaus-shared-LangLoc-strict-Encoding', ANNSet1ECoGUniGaussSharedLanglocStrictEncoding),

    ('ANNSet1ECoG-uni-band-shared-LangLoc-Encoding', ANNSet1ECoGUniBandSharedLanglocEncoding),
    ('ANNSet1ECoG-uni-band-shared-LangLoc-strict-Encoding', ANNSet1ECoGUniBandSharedLanglocStrictEncoding),

    ('LangLocECoG-encoding', LangLocECoGEncoding),

    ('LangLocECoG-bip-gaus-Encoding', LangLocECoGBipGaussEncoding),
    ('LangLocECoG-bip-gaus-strict-Encoding', LangLocECoGBipGaussStrictEncoding),

    ('LangLocECoG-uni-gaus-Encoding', LangLocECoGUniGaussEncoding),
    ('LangLocECoG-uni-gaus-strict-Encoding', LangLocECoGUniGaussStrictEncoding),

    ('LangLocECoG-bip-gaus-zs-Encoding', LangLocECoGBipGaussZSEncoding),
    ('LangLocECoG-bip-gaus-zs-strict-Encoding', LangLocECoGBipGaussZSStrictEncoding),

    ('LangLocECoG-uni-gaus-zs-Encoding', LangLocECoGUniGaussZSEncoding),
    ('LangLocECoG-uni-gaus-zs-strict-Encoding', LangLocECoGUniGaussZSStrictEncoding),

    ('LangLocECoG-bip-band-Encoding', LangLocECoGBipBandEncoding),
    ('LangLocECoG-bip-band-strict-Encoding', LangLocECoGBipBandStrictEncoding),

    ('LangLocECoG-uni-band-Encoding', LangLocECoGUniBandEncoding),
    ('LangLocECoG-uni-band-strict-Encoding', LangLocECoGUniBandStrictEncoding),

    ('LangLocECoG-bip-gaus-shared-ANN-Encoding', LangLocECoGBipGaussSharedANNEncoding),
    ('LangLocECoG-bip-gaus-shared-ANN-strict-Encoding', LangLocECoGUniGaussSharedANNStrictEncoding),

    ('LangLocECoG-uni-gaus-shared-ANN-Encoding', LangLocECoGUniGaussSharedANNEncoding),
    ('LangLocECoG-uni-gaus-shared-ANN-strict-Encoding', LangLocECoGUniGaussSharedANNStrictEncoding),

    ('LangLocECoG-bip-band-shared-ANN-Encoding', LangLocECoGBipBandSharedANNEncoding),
    ('LangLocECoG-bip-band-shared-ANN-strict-Encoding', LangLocECoGBipBandSharedANNStrictEncoding),

    ('LangLocECoG-uni-band-shared-ANN-Encoding', LangLocECoGUniBandSharedANNEncoding),
    ('LangLocECoG-uni-band-shared-ANN-strict-Encoding', LangLocECoGUniBandSharedANNStrictEncoding),

    ('LangLocECoG-sentence-encoding', LangLocECoGSentenceEncoding),
    #('LangLocECoGv2-encoding', LangLocECoGV2Encoding),
    ('LangLocECoGSample-encoding', LangLocECoGSampleEncoding),
    ('Fedorenko2016v3-encoding', Fedorenko2016V3Encoding),
    ('Blank2014fROI-encoding', Blank2014fROIEncoding),
    #('MghMockLang-encoding', MghMockLangEncoding),
    # secondary benchmarks
    ('Pereira2018-rdm', PereiraRDM),
    ('Pereira2018-cka', PereiraCKA),
    ('Fedorenko2016v3-rdm', Fedorenko2016V3RDM),
    ('Fedorenko2016v3-cka', Fedorenko2016V3CKA),
    ('Fedorenko2016v3nonlang-encoding', Fedorenko2016V3NonLangEncoding),
    ('Blank2014fROI-rdm', Blank2014fROIRDM),
    ('Blank2014fROI-cka', Blank2014fROICKA),
]
for sentence_num in range(1, 10, 2):
    benchmark_pool.append((f'Blank2014sentence{sentence_num}fROI-encoding',
                           lambda *args, sentence_num=sentence_num, **kwargs:
                           Blank2014SentencefROIEncoding(*args, sentence_num=sentence_num, **kwargs)))

# signal_type=[('bip-gauss','HighGamma_bipolar_gauss_subs_17'),
#             ('bip-band','HighGamma_bipolar_bandpass_subs_17'),
#             ('uni-gaus','HighGamma_unipolar_gauss_subs_17'),
#             ('uni-band','HighGamma_unipolar_bandpass_subs_17')]

# for name,file in signal_type:
#     benchmark_pool.append((f'LangLocECoG-{name}-Encoding',
#                            lambda *args, version=file, **kwargs:
#                            LangLocECoGLinearEncoding(*args, version=file,threshold=0.05, **kwargs)))



benchmark_pool = {identifier: LazyLoad(lambda identifier=identifier, ctr=ctr: ctr(identifier=identifier))
                  for identifier, ctr in benchmark_pool}
