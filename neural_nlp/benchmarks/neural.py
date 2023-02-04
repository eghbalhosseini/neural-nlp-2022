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
from scipy.stats import median_absolute_deviation
from tqdm import tqdm, trange

from brainscore.benchmarks import Benchmark
from brainscore.metrics import Score
from brainscore.metrics.rdm import RDM, RDMSimilarity, RDMCrossValidated
from brainscore.metrics.cka import CKACrossValidated
from brainscore.metrics.regression import linear_regression, pearsonr_correlation, CrossRegressedCorrelation
from brainscore.metrics.transformations import CartesianProduct, CrossValidation, apply_aggregate
from brainscore.utils import LazyLoad
from neural_nlp.benchmarks.ceiling import ExtrapolationCeiling, HoldoutSubjectCeiling, v,ci_error,manual_merge,_coords_match
from neural_nlp.benchmarks.s3 import load_s3
from neural_nlp.neural_data.ecog import load_Fedorenko2016
from neural_nlp.neural_data.mgh_ecog import load_MghMockLang
from neural_nlp.neural_data.fmri import load_voxels, load_rdm_sentences, \
    load_Pereira2018_Blank
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
if getpass.getuser() == 'eghbalhosseini':
    ANNfMRI_PARENT = '/Users/eghbalhosseini/MyData/brain-score-language/dataset/'
    ANNECOG_PARENT = '/Users/eghbalhosseini/MyData/brain-score-language/dataset/'

elif getpass.getuser() == 'ehoseini':
    ANNfMRI_PARENT = '/om2/user/ehoseini/MyData/brain-score-language/dataset/'
    ANNECOG_PARENT = '/om2/user/ehoseini/MyData/brain-score-language/dataset/'



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
        assert len(cross_assembly['presentation']) in [243, 384]
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
        self._single_metric = metric
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
            sentence_stimuli = StimulusSet({'sentence': part_stimuli.values[0],
                                            reset_column: list(set(part_stimuli[reset_column].values))})
            sentence_stimuli.name = f"{self._target_assembly.identifier}-{reset_id}"
            sentence_activations = candidate(stimuli=sentence_stimuli, average_sentence=average_sentence)[-1, :]
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
        stimulus_set.word_id
        model_activations = self._read_words(candidate, stimulus_set, copy_columns=['word_id'],reset_column='stimuls_id')
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
        ceiling_val=pd.read_pickle(f'{ANNfMRI_PARENT}/ANNSet1_fMRI-train-language_top_90-linear_ceiling.pkl')
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

        def _load_assembly(self):
            # read UD data and replace stimuli
            assembly = pd.read_pickle(f'{ANNfMRI_PARENT}/ANNSet1_fMRI-train-language_top_90.pkl')
            UD_data = pd.read_pickle(f'{ANNfMRI_PARENT}/ud_sentencez_data_token_filter_v3_brainscore.pkl')
            sentence_texts=[]
            for stim_id,stim in UD_data.groupby('stimulus_id'):
                sentence_texts.append(np.unique(stim.text.values)[0])
            sentence_index=[sentence_texts.index(x) for x in assembly.stimulus.values]
            assert(len(sentence_index)==200)
            selected_stim=[]
            for sent_id in sentence_index:
                location=(UD_data.stimulus_id==sent_id).values
                selected_stim.append(UD_data.sel(index=location))

            assert all([np.unique(x.text)[0]==assembly.stimulus.values[idx] for idx, x in enumerate(selected_stim)])
            stimulus_form=[' '.join(x.word_FORM.values) for x in selected_stim]

            new_assembly = NeuroidAssembly(assembly.values, coords={
                'experiment': ('presentation', assembly.experiment.values),
                'stimulus_num': ('presentation', assembly.stimulus_num.values),
                'stimulus_id': ('presentation', assembly.stimulus_id.values),
                'sentence': ('presentation', stimulus_form),
                'stimulus': ('presentation', stimulus_form),
                'list_id': ('presentation', assembly.list_id.values),
                'stim_type': ('presentation', assembly.stim_type.values),
                'stim_name': ('presentation', assembly.stim_name.values),
                'Trial_id': ('presentation', assembly.Trial_id.values),
                'TR_onset': ('presentation', assembly.TR_onset.values),
                'TR_recorded': ('presentation', assembly.TR_recorded.values),
                'TR_duration': ('presentation', assembly.TR_duration.values),
                'subject': ('neuroid', assembly.subject.values),
                'neuroid_id': ('neuroid', assembly.neuroid_id.values),
                'voxel_num': ('neuroid', assembly.voxel_num.values),
                'repetition_corr_ratio': ('neuroid', assembly.repetition_corr_ratio.values),
                'repetition_corr': ('neuroid', assembly.repetition_corr.values),
                'roi': ('neuroid', assembly.roi.values),
                'atlas': ('neuroid',assembly.atlas.values)
            }, dims=['presentation', 'neuroid'])
            new_assembly = new_assembly.sortby('stimulus_id')

            new_assembly.attrs['identifier']=assembly.identifier+'_wordForm'
            name=new_assembly.identifier.replace('.','-')
            with open(Path(f'{ANNfMRI_PARENT}/{name}.pkl').__str__(),'wb') as f:
                pickle.dump(new_assembly,f)

            return new_assembly
    @property
    def ceiling(self):
        ceiling_val = pd.read_pickle(f'{ANNfMRI_PARENT}/ANNSet1_fMRI-train-language_top_90-linear_ceiling.pkl')
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
    def __init__(self, identifier, metric,version='base'):
        self._identifier = identifier
        self._metric = metric
        assembly = self._load_assembly(version=version)
        self._target_assembly = assembly
        self.average_sentence=False

    def apply_metric(self, model_activations, target_assembly):
        return self._metric(model_activations, target_assembly)

    @property
    def identifier(self):
        return self._identifier

    def _load_assembly(self,version='base'):
        if version=='base':
            assembly = pd.read_pickle(f'{ANNECOG_PARENT}/ANNSet1_ECoG.train.HighGamma_unipolar_gauss_zscore.pkl')
        else:
            assembly=None
        # make sure the assembly is ordered based on stimulus_id
        assembly = assembly.sortby('stimulus_id')
        # this is to make sure that the assembly is ordered based on stimulus_id
        # and that the word_ids are in increasing order
        assembly = assembly.groupby('stimulus_id').apply(lambda x: x.sortby('word_id'))
        # define a new coordinate called sentence_id with the same values as stimulus_id
        assembly = assembly.assign_coords(sentence_id=assembly.stimulus_id)
        # define a new coordiante stimuli_id that goes from 0 to size presentation dimension
        assembly = assembly.assign_coords({'stimuli_id':('presentation',np.arange(assembly.stimulus_id.size))})
        # select electrodes that have an roi=='language'
        assembly = assembly.sel(neuroid=assembly.roi=='language')

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

        super(ANNSet1ECoGEncoding, self).__init__(identifier=identifier, metric=metric)

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
        self._few_sub_ceiler=self.FewSubjectExtrapolation(subject_column='subject',extrapolation_dimension='neuroid',num_bootstraps=100,post_process=None)


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
        assembly = assembly.rename({'stim_id': 'stimulus_id'})
        # this is to make sure that the assembly is ordered based on stimulus_id
        # and that the word_ids are in increasing order
        assembly = assembly.groupby('stimulus_id').apply(lambda x: x.sortby('word_id'))
        # define a new coordinate called sentence_id with the same values as stimulus_id
        assembly = assembly.assign_coords(sentence_id=assembly.stimulus_id)
        # define a new coordiante stimuli_id that goes from 0 to size presentation dimension
        assembly = assembly.assign_coords({'stimuli_id': ('presentation', np.arange(assembly.stimulus_id.size))})
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
    
        # count number of electrodes with s_v_n_ratio > 0.99
        # print(f"Number of electrodes with s_v_n_ratio > 0.99: {np.sum(assembly['s_v_n_ratio'] > 0.99)}")
        # make a neural assembly from xarray assembly
        
        assembly = NeuroidAssembly(assembly_new)
        
        # add identifier to assembly
        
        thr_str=str(threshold).replace('.','')
        assembly.attrs['identifier'] = f"LangLoc_ECoG.{version}_{type}_thr_{thr_str}"
        
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
        # idx = [model_activations[reset_column].values.tolist().index(stimulus_id) for stimulus_id in
        #       [int(s[reset_column].values) for s in activations]]
        # assert len(set(idx)) == len(idx), "Found duplicate indices to order activations"
        # model_activations = model_activations[{'presentation': idx}]

        return model_activations

    def apply_metric(self, model_activations, target_assembly):
        return self._metric(model_activations, target_assembly)

    def __call__(self, candidate, *args, **kwargs):
        stimulus_set = self._target_assembly['stimulus']
        model_activations = self._read_words(candidate, stimulus_set, copy_columns=['stimuli_id', 'word_id'])
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

    class FewSubjectExtrapolation:
        def __init__(self, subject_column,extrapolation_dimension,num_bootstraps,post_process, *args, **kwargs):
            super(_LanglocECOG.FewSubjectExtrapolation, self).__init__(*args, **kwargs)
            self._rng = RandomState(0)
            self._num_subsamples = 200   # number of subsamples per subject selection
            #self.holdout_ceiling = HoldoutSubjectCeiling(subject_column=subject_column)
            self._logger = logging.getLogger(fullname(self))
            self.subject_column = subject_column
            self.extrapolation_dimension = extrapolation_dimension
            self.num_bootstraps = num_bootstraps
            self._post_process = post_process

        @store(identifier_ignore=['assembly', 'metric'])
        def __call__(self, identifier, assembly, metric):
            scores = self.collect(identifier, assembly=assembly, metric=metric)
            return self.extrapolate(scores)

        @store(identifier_ignore=['assembly', 'metric'])
        def collect(self, identifier, assembly, metric):
            subjects = set(assembly[self.subject_column].values)
            subject_subsamples = self.build_subject_subsamples(subjects)
            scores = []
            scores_raw=[]
            for num_subjects in tqdm(subject_subsamples, desc='num subjects'):
                selection_combinations = self.iterate_subsets(assembly, num_subjects=num_subjects)
                comb_scores=[]
                for selections, sub_assembly in tqdm(selection_combinations, desc='selections'):
                    sub_scores=[]
                    iterate_subjects = np.unique(sub_assembly[self.subject_column].values)
                    for subject in tqdm(iterate_subjects, desc='subjects',disable=True):
                        # select subject data from sub_assembly
                        True
                        subject_assembly = sub_assembly.sel(neuroid=(sub_assembly[self.subject_column] == subject).values)
                        # select data from other subjects
                        other_subject_data = sub_assembly.sel(neuroid=(sub_assembly[self.subject_column] != subject).values)
                        # run subject as a neural candidate
                        score=metric(other_subject_data, subject_assembly)
                        assert not np.all(np.isnan(score.raw.values))
                        apply_raw = 'raw' in score.attrs and not hasattr(score.raw, self.subject_column)  # only propagate if column not part of score
                        score = score.expand_dims(self.subject_column, _apply_raw=apply_raw)
                        score.__setitem__(self.subject_column, [subject], _apply_raw=apply_raw)
                        sub_scores.append(score)
                    comb_scores.append(sub_scores)
                # go into each element of comb_scores and concatenate the raw attributes along neuriod dimension
                comb_scores_raw=[]
                for i, sub_scores in enumerate(comb_scores):
                    sub_scores_raw = [s.raw for s in sub_scores]
                    # take the mean over splits for each element in sub_scores_raw
                    sub_scores_raw = xr.concat(sub_scores_raw, dim='neuroid')
                    sub_scores_raw = sub_scores_raw.mean('split')
                    comb_scores_raw.append(sub_scores_raw)
                # concatenate the comb_scores_raw along neuroid dimension
                comb_scores_raw = xr.concat(comb_scores_raw, dim='subsample')
                # assign a value to subsample coordiante as an array with num_subjects
                comb_scores_raw = comb_scores_raw.assign_coords(subsample=np.ones(comb_scores_raw.shape[0])*num_subjects)
                # make sure that comb_scores_raw has the same number of neuroids as the original assembly
                # check if comb_scores_raw has the same number of neuroids as the original assembly and if not, add the missing neuroids as nan
                if comb_scores_raw.shape[1] != assembly.shape[1]:
                    # get the neuroid ids that are missing
                    missing_neuroids = np.setdiff1d(assembly['neuroid_id'].values, comb_scores_raw['neuroid_id'].values)
                    missing_neuroids =np.sum([assembly.neuroid_id.values==x for x in missing_neuroids],axis=0).astype(bool)
                    temp_assembly = assembly.sel(neuroid=missing_neuroids)
                    temp_assembly = temp_assembly.drop('presentation')
                    temp_assembly=temp_assembly.sum('presentation')
                    temp_assembly.values *= np.nan
                    comb_scores_raw=xr.concat([comb_scores_raw, temp_assembly], dim='neuroid')

                    # sort the neuroid_ids
                    comb_scores_raw = comb_scores_raw.sortby('neuroid_id')
                else:
                    comb_scores_raw = comb_scores_raw.sortby('neuroid_id')
                scores_raw.append(comb_scores_raw)

            scores=xr.concat(scores_raw, dim='subsample')
            # for x in a.transpose().values:
            #     plt.scatter(a.subsample.values,x,s=5,c='k',alpha=0.1)
            # plt.show()

            #
            #         score = Score.merge(*sub_scores)
            #         error = score.sel(aggregation='center').std(self.subject_column)
            #         score = apply_aggregate(lambda score: score.mean(self.subject_column), score)
            #         score.loc[{'aggregation': 'error'}] = error
            #         #score = self.holdout_ceiling(assembly=sub_assembly, metric=metric)
            #         score = score.expand_dims('num_subjects')
            #         score['num_subjects'] = [num_subjects]
            #         for key, selection in selections.items():
            #             expand_dim = f'sub_{key}'
            #             score = score.expand_dims(expand_dim)
            #             score[expand_dim] = [str(selection)]
            #         scores.append(score.raw)
            #
            #         # except KeyError as e:  # nothing to merge
            #         #     if str(e) == "'z'":
            #         #         self._logger.debug(f"Ignoring merge error {e}")
            #         #         continue
            #         #     else:
            #         #         raise e
            #
            # scores = Score.merge(*scores)
            # scores = self.post_process(scores)
            # return scores

        def build_subject_subsamples(self, subjects):
            return tuple(range(2, len(subjects) + 1, 1))  # reduce computational cost by only using every other point

        def iterate_subsets(self, assembly, num_subjects):
            # there are 180 subjects which makes for millions of combinations.
            # to avoid this computational explosion, we choose only a subset of the possible subject sub-samples.
            subjects = set(assembly[self.subject_column].values)
            subject_combinations = self._random_combinations(subjects, num_subjects,
                                                             choice=self._num_subsamples, rng=self._rng)
            for sub_subjects in tqdm(subject_combinations, desc="subject combinations"):
                sub_assembly = assembly[{'neuroid': [subject in sub_subjects
                                                     for subject in assembly[self.subject_column].values]}]
                yield {self.subject_column: sub_subjects}, sub_assembly

        def extrapolate(self, ceilings):
            neuroid_ceilings = []
            raw_keys = ['bootstrapped_params', 'error_low', 'error_high', 'endpoint_x']
            raw_attrs = defaultdict(list)
            asymptote_threshold = .0005
            interpolation_xs = np.arange(1000)
            for i in trange(len(ceilings[self.extrapolation_dimension]),desc=f'{self.extrapolation_dimension} extrapolations'):
                neuroid_ceiling = ceilings.isel(**{self.extrapolation_dimension: [i]})
                subject_subsamples = list(sorted(set(neuroid_ceiling['subsample'].values)))
                rng = RandomState(0)
                bootstrap_params = []
                for bootstrap in range(self.num_bootstraps):
                    bootstrapped_scores = []
                    for num_subjects in subject_subsamples:
                        num_scores = neuroid_ceiling.sel(subsample=num_subjects)
                        choices = num_scores.values.flatten()
                        bootstrapped_score = rng.choice(choices, size=len(choices), replace=True)
                        bootstrapped_scores.append(np.nanmean(bootstrapped_score))
                        #bootstrapped_score = rng.choice(choices, size=10, replace=True)
                        #bootstrapped_scores.append(np.nanmean(bootstrapped_score))

                    valid = ~np.isnan(bootstrapped_scores)
                    if sum(valid) < 1:
                        raise RuntimeError("No valid scores in sample")
                    # drop nan
                    y = np.array(bootstrapped_scores)[valid]
                    x= np.array(subject_subsamples)[valid]
                    try:
                        params = self.fit(x, y)
                    except:
                        params = [np.nan, np.nan]
                    params = DataAssembly([params], coords={'bootstrap': [bootstrap], 'param': ['v0', 'tau0']},
                                          dims=['bootstrap', 'param'])
                    bootstrap_params.append(params)
                bootstrap_params = merge_data_arrays(bootstrap_params)
                if not np.isnan(bootstrap_params.values).all():
                    ys = np.array([v(interpolation_xs, *params) for params in bootstrap_params.values
                                   if not np.isnan(params).any()])
                    median_ys = np.median(ys, axis=0)
                    diffs = np.diff(median_ys)
                    end_x = np.where(diffs < asymptote_threshold)[
                        0].min()  # first x where increase smaller than threshold
                    # put together
                    center = np.median(np.array(bootstrap_params)[:, 0])
                    error_low, error_high = ci_error(ys[:, end_x], center=center)
                    ceiling_estimate = Score(center)
                    ceiling_estimate.attrs['raw'] = neuroid_ceiling
                    ceiling_estimate.attrs['error_low'] = DataAssembly(error_low)
                    ceiling_estimate.attrs['error_high'] = DataAssembly(error_high)
                    ceiling_estimate.attrs['bootstrapped_params'] = bootstrap_params
                    ceiling_estimate.attrs['endpoint_x'] = DataAssembly(end_x)
                ceiling_estimate = self.add_neuroid_meta(ceiling_estimate, neuroid_ceiling)
                neuroid_ceilings.append(ceiling_estimate)
            neuroid_ceilings = manual_merge(*neuroid_ceilings, on=self.extrapolation_dimension)

            neuroid_ceilings.attrs['raw'] = ceilings
            return neuroid_ceilings
        # 
        def _random_combinations(self, subjects, num_subjects, choice, rng):
            # following https://stackoverflow.com/a/55929159/2225200
            # building all `itertools.combinations` followed by `rng.choice` subsampling
            # would lead to >1 trillion initial samples.
            subjects = np.array(list(subjects))
            combinations = set()
            while len(combinations) < choice:
                elements = rng.choice(subjects, size=num_subjects, replace=False)
                combinations.add(tuple(elements))
            return combinations

        def add_neuroid_meta(self, target, source):
            target = target.expand_dims(self.extrapolation_dimension)
            for coord, dims, values in walk_coords(source):
                if array_is_element(dims, self.extrapolation_dimension):
                    target[coord] = dims, values
            return target

        def aggregate_neuroid_ceilings(self, neuroid_ceilings, raw_keys):
            ceiling = neuroid_ceilings.median(self.extrapolation_dimension)
            ceiling.attrs['raw'] = neuroid_ceilings
            for key in raw_keys:
                values = neuroid_ceilings.attrs[key]
                aggregate = values.median(self.extrapolation_dimension)
                if not aggregate.shape:  # scalar value, e.g. for error_low
                    aggregate = aggregate.item()
                ceiling.attrs[key] = aggregate
            return ceiling

        def extrapolate_neuroid(self, ceilings):
            # figure out how many extrapolation x points we have. E.g. for Pereira, not all combinations are possible
            subject_subsamples = list(sorted(set(ceilings['num_subjects'].values)))
            rng = RandomState(0)
            bootstrap_params = []
            for bootstrap in range(self.num_bootstraps):
                bootstrapped_scores = []
                for num_subjects in subject_subsamples:
                    num_scores = ceilings.sel(num_subjects=num_subjects)
                    # the sub_subjects dimension creates nans, get rid of those
                    num_scores = num_scores.dropna(f'sub_{self.subject_column}')
                    assert set(num_scores.dims) == {f'sub_{self.subject_column}', 'split'} or \
                           set(num_scores.dims) == {f'sub_{self.subject_column}'}
                    # choose from subject subsets and the splits therein, with replacement for variance
                    choices = num_scores.values.flatten()
                    bootstrapped_score = rng.choice(choices, size=len(choices), replace=True)
                    bootstrapped_scores.append(np.mean(bootstrapped_score))
                try:
                    params = self.fit(subject_subsamples, bootstrapped_scores)
                except:
                    params = [np.nan, np.nan]
                params = DataAssembly([params], coords={'bootstrap': [bootstrap], 'param': ['v0', 'tau0']},
                                      dims=['bootstrap', 'param'])
                bootstrap_params.append(params)
            bootstrap_params = merge_data_arrays(bootstrap_params)
            # find endpoint and error
            asymptote_threshold = .0005
            interpolation_xs = np.arange(1000)
            if not np.isnan(bootstrap_params.values).all():
                ys = np.array([v(interpolation_xs, *params) for params in bootstrap_params.values
                               if not np.isnan(params).any()])
                median_ys = np.median(ys, axis=0)
                diffs = np.diff(median_ys)
                end_x = np.where(diffs < asymptote_threshold)[0].min()  # first x where increase smaller than threshold
                # put together
                center = np.median(np.array(bootstrap_params)[:, 0])
                error_low, error_high = ci_error(ys[:, end_x], center=center)
                score = Score(center)
                score.attrs['raw'] = ceilings
                score.attrs['error_low'] = DataAssembly(error_low)
                score.attrs['error_high'] = DataAssembly(error_high)
                score.attrs['bootstrapped_params'] = bootstrap_params
                score.attrs['endpoint_x'] = DataAssembly(end_x)
            else:
                score = Score(np.asarray(np.nan))
                score.attrs['raw'] = ceilings
                score.attrs['error_low'] = DataAssembly(np.asarray(np.nan))
                score.attrs['error_high'] = DataAssembly(np.asarray(np.nan))
                score.attrs['bootstrapped_params'] = bootstrap_params
                score.attrs['endpoint_x'] = DataAssembly(np.asarray(np.nan))
            return score

        def fit(self, subject_subsamples, bootstrapped_scores):
            valid = ~np.isnan(bootstrapped_scores)
            if sum(valid) < 1:
                raise RuntimeError("No valid scores in sample")
            params, pcov = curve_fit(v, subject_subsamples, bootstrapped_scores,
                                     # v (i.e. max ceiling) is between 0 and 1, tau0 unconstrained
                                     bounds=([0, -np.inf], [1, np.inf]))
            return params


class LangLocECoGEncoding(_LanglocECOG):
    def __init__(self, identifier):
        regression = linear_regression(xarray_kwargs=dict(stimulus_coord='stimuli_id'))  # word
        correlation = pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimuli_id'))  # word
        metric = CrossRegressedCorrelation(regression=regression, correlation=correlation,
                                           crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord='stimuli_id',
                                                                       stratification_coord='stimulus_id'))

        super(LangLocECoGEncoding, self).__init__(identifier=identifier, metric=metric,type='language',version='HighGamma_bipolar_gauss_zscore_subs_17',threshold=0.05)

    def ceiling(self):
        return super(_LanglocECOG, self).ceiling


class LangLocECoGV2Encoding(_LanglocECOG):
        def __init__(self,identifier,**kwargs):
            regression = linear_regression(xarray_kwargs=dict(stimulus_coord='stimuli_id'))  # word
            correlation = pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimuli_id'))  # word
            metric = CrossRegressedCorrelation(regression=regression, correlation=correlation,
                                               crossvalidation_kwargs=dict(splits=5, kfold=True,
                                                                           split_coord='stimuli_id',
                                                                           stratification_coord='stimulus_id'))
            super(LangLocECoGV2Encoding, self).__init__(identifier=identifier, metric=metric, type='language',
                                                      version='HighGamma_bipolar_gauss_zscore_subs_17', threshold=0.01)


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


benchmark_pool = [
    # primary benchmarks
    ('Pereira2018-encoding', PereiraEncoding),
    ('ANNSet1fMRI-encoding', ANNSet1fMRIEncoding),
    ('ANNSet1fMRI-wordForm-encoding',ANNSet1fMRIEncoding_V2),
    ('ANNSet1fMRISentence-encoding', ANNSet1fMRISentenceEncoding),
    ('ANNSet1fMRISentence-wordForm-encoding', ANNSet1fMRISentenceEncoding_V2),
    ('ANNSet1ECoG-encoding', ANNSet1ECoGEncoding),
    ('LangLocECoG-encoding', LangLocECoGEncoding),
    ('LangLocECoGv2-encoding', LangLocECoGV2Encoding),
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
benchmark_pool = {identifier: LazyLoad(lambda identifier=identifier, ctr=ctr: ctr(identifier=identifier))
                  for identifier, ctr in benchmark_pool}
