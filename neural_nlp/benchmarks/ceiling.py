import itertools
import logging
import numpy as np
from brainio.assemblies import DataAssembly, merge_data_arrays, walk_coords, array_is_element
from brainio.fetch import fullname
from numpy import AxisError
from numpy.random.mtrand import RandomState
from scipy.optimize import curve_fit
from tqdm import tqdm, trange
import xarray as xr
from brainscore.metrics import Score
from brainscore.metrics.transformations import Transformation, Split, enumerate_done, subset
from brainscore.metrics.transformations import apply_aggregate
from result_caching import store
from scipy.special import comb, perm
from collections import defaultdict
import warnings
def v(x, v0, tau0):
    return v0 * (1 - np.exp(-x / tau0))

def v_overflow(x,v0,tau0):
    with np.errstate(over='ignore'):
        return v0 * (1 - np.exp(-x / tau0))

# ceiling crossvalidation, same as crossvalidation with the difference that tqdm is supperessed
class CeilingCrossValidation(Transformation):
    """
    Performs multiple splits over a source and target assembly.
    No guarantees are given for data-alignment, use the metadata.
    """

    def __init__(self, *args, split_coord=Split.Defaults.split_coord,
                 stratification_coord=Split.Defaults.stratification_coord,
                 preprocess_indices=None, show_tqdm=False, **kwargs):
        self._split_coord = split_coord
        self._stratification_coord = stratification_coord
        self._split = Split(*args, split_coord=split_coord, stratification_coord=stratification_coord, **kwargs)
        self._logger = logging.getLogger(fullname(self))
        self._show_tqdm = show_tqdm
        self._preprocess_indices = preprocess_indices

    def pipe(self, source_assembly, target_assembly):
        # check only for equal values, alignment is given by metadata
        assert sorted(source_assembly[self._split_coord].values) == sorted(target_assembly[self._split_coord].values)
        if self._split.do_stratify:
            assert hasattr(source_assembly, self._stratification_coord)
            assert sorted(source_assembly[self._stratification_coord].values) == \
                   sorted(target_assembly[self._stratification_coord].values)
        cross_validation_values, splits = self._split.build_splits(target_assembly)

        split_scores = []
        for split_iterator, (train_indices, test_indices), done \
                in tqdm(enumerate_done(splits), total=len(splits), desc='cross-validation',
                        disable=not self._show_tqdm):

            if hasattr(self, '_preprocess_indices'):
                if self._preprocess_indices is not None:
                    train_indices, test_indices = self._preprocess_indices(train_indices, test_indices, source_assembly)

            train_values, test_values = cross_validation_values[train_indices], cross_validation_values[test_indices]
            train_source = subset(source_assembly, train_values, dims_must_match=False)
            train_target = subset(target_assembly, train_values, dims_must_match=False)
            assert len(train_source[self._split_coord]) == len(train_target[self._split_coord])
            test_source = subset(source_assembly, test_values, dims_must_match=False)
            test_target = subset(target_assembly, test_values, dims_must_match=False)
            assert len(test_source[self._split_coord]) == len(test_target[self._split_coord])
            split_score = yield from self._get_result(train_source, train_target, test_source, test_target,
                                                      done=done)
            split_score = split_score.expand_dims('split')
            split_score['split'] = [split_iterator]
            split_scores.append(split_score)

        split_scores = Score.merge(*split_scores)
        yield split_scores

    def aggregate(self, score):
        return self._split.aggregate(score)


class HoldoutSubjectCeiling:
    def __init__(self, subject_column):
        self.subject_column = subject_column
        self._logger = logging.getLogger(fullname(self))

    def __call__(self, assembly, metric):
        subjects = set(assembly[self.subject_column].values)
        scores = []
        iterate_subjects = self.get_subject_iterations(subjects)
        for subject in tqdm(iterate_subjects, desc='heldout subject'):
            try:
                subject_assembly = assembly[{'neuroid': [subject_value == subject
                                                         for subject_value in assembly[self.subject_column].values]}]
                # run subject pool as neural candidate
                subject_pool = subjects - {subject}
                pool_assembly = assembly[
                    {'neuroid': [subject in subject_pool for subject in assembly[self.subject_column].values]}]
                score = self.score(pool_assembly, subject_assembly, metric=metric)
                # store scores
                apply_raw = 'raw' in score.attrs and \
                            not hasattr(score.raw, self.subject_column)  # only propagate if column not part of score
                score = score.expand_dims(self.subject_column, _apply_raw=apply_raw)
                score.__setitem__(self.subject_column, [subject], _apply_raw=apply_raw)
                scores.append(score)
            except NoOverlapException as e:
                self._logger.debug(f"Ignoring no overlap {e}")
                continue  # ignore
            except ValueError as e:
                if "Found array with" in str(e):
                    self._logger.debug(f"Ignoring empty array {e}")
                    continue
                else:
                    raise e

        scores = Score.merge(*scores)
        center_row = (scores.aggregation=='center').values
        # select only the center row
        error = scores.isel(aggregation=center_row).std(self.subject_column)
        #error = scores.sel(aggregation='center').std(self.subject_column)
        scores = apply_aggregate(lambda scores: scores.mean(self.subject_column), scores)
        # find error location in aggegation dim

        center=scores.isel(aggregation=center_row)
        scores[:]=np.concatenate([center.values,error.values])
        # replace the error_row with the error

        #scores.loc[{'aggregation': 'error'}] = error
        return scores

    def get_subject_iterations(self, subjects):
        return subjects  # iterate over all subjects

    def score(self, pool_assembly, subject_assembly, metric):
        return metric(pool_assembly, subject_assembly)


class ExtrapolationCeiling:
    def __init__(self, subject_column='subject', extrapolation_dimension='neuroid',
                 num_bootstraps=100, post_process=None):
        self._logger = logging.getLogger(fullname(self))
        self.subject_column = subject_column
        self.holdout_ceiling = HoldoutSubjectCeiling(subject_column=subject_column)
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
        for num_subjects in tqdm(subject_subsamples, desc='num subjects'):
            selection_combinations = self.iterate_subsets(assembly, num_subjects=num_subjects)
            for selections, sub_assembly in tqdm(selection_combinations, desc='selections'):
                try:
                    score = self.holdout_ceiling(assembly=sub_assembly, metric=metric)
                    score = score.expand_dims('num_subjects')
                    score['num_subjects'] = [num_subjects]
                    for key, selection in selections.items():
                        expand_dim = f'sub_{key}'
                        score = score.expand_dims(expand_dim)
                        score[expand_dim] = [str(selection)]
                    scores.append(score.raw)
                except KeyError as e:  # nothing to merge
                    if str(e) == "'z'":
                        self._logger.debug(f"Ignoring merge error {e}")
                        continue
                    else:
                        raise e
        scores = Score.merge(*scores)
        scores = self.post_process(scores)
        return scores

    def build_subject_subsamples(self, subjects):
        return tuple(range(2, len(subjects) + 1))

    def iterate_subsets(self, assembly, num_subjects):
        subjects = set(assembly[self.subject_column].values)
        subject_combinations = list(itertools.combinations(subjects, num_subjects))
        for sub_subjects in subject_combinations:
            sub_assembly = assembly[{'neuroid': [subject in sub_subjects
                                                 for subject in assembly[self.subject_column].values]}]
            yield {self.subject_column: sub_subjects}, sub_assembly

    def average_collected(self, scores):
        return scores.median('neuroid')

    def extrapolate(self, ceilings):
        neuroid_ceilings, bootstrap_params, endpoint_xs = [], [], []
        for i in trange(len(ceilings[self.extrapolation_dimension]),
                        desc=f'{self.extrapolation_dimension} extrapolations'):
            try:
                # extrapolate per-neuroid ceiling
                neuroid_ceiling = ceilings.isel(**{self.extrapolation_dimension: [i]})
                extrapolated_ceiling = self.extrapolate_neuroid(neuroid_ceiling.squeeze())
                extrapolated_ceiling = self.add_neuroid_meta(extrapolated_ceiling, neuroid_ceiling)
                neuroid_ceilings.append(extrapolated_ceiling)
                # also keep track of bootstrapped parameters
                neuroid_bootstrap_params = extrapolated_ceiling.bootstrapped_params
                neuroid_bootstrap_params = self.add_neuroid_meta(neuroid_bootstrap_params, neuroid_ceiling)
                bootstrap_params.append(neuroid_bootstrap_params)
                # and endpoints
                endpoint_x = self.add_neuroid_meta(extrapolated_ceiling.endpoint_x, neuroid_ceiling)
                endpoint_xs.append(endpoint_x)
            except AxisError:  # no extrapolation successful (happens for 1 neuroid in Pereira)
                continue
        # merge and add meta
        self._logger.debug("Merging neuroid ceilings")
        neuroid_ceilings = manual_merge(*neuroid_ceilings, on=self.extrapolation_dimension)
        neuroid_ceilings.attrs['raw'] = ceilings
        self._logger.debug("Merging bootstrap params")
        bootstrap_params = manual_merge(*bootstrap_params, on=self.extrapolation_dimension)
        neuroid_ceilings.attrs['bootstrapped_params'] = bootstrap_params
        self._logger.debug("Merging endpoints")
        endpoint_xs = manual_merge(*endpoint_xs, on=self.extrapolation_dimension)
        neuroid_ceilings.attrs['endpoint_x'] = endpoint_xs
        # aggregate
        ceiling = self.aggregate_neuroid_ceilings(neuroid_ceilings)
        return ceiling

    def add_neuroid_meta(self, target, source):
        target = target.expand_dims(self.extrapolation_dimension)
        for coord, dims, values in walk_coords(source):
            if array_is_element(dims, self.extrapolation_dimension):
                target[coord] = dims, values
        return target

    def aggregate_neuroid_ceilings(self, neuroid_ceilings):
        ceiling = neuroid_ceilings.median(self.extrapolation_dimension)
        ceiling.attrs['bootstrapped_params'] = neuroid_ceilings.bootstrapped_params.median(self.extrapolation_dimension)
        ceiling.attrs['endpoint_x'] = neuroid_ceilings.endpoint_x.median(self.extrapolation_dimension)
        ceiling.attrs['raw'] = neuroid_ceilings
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
            except RuntimeError:  # optimal parameters not found
                params = [np.nan, np.nan]
            params = DataAssembly([params], coords={'bootstrap': [bootstrap], 'param': ['v0', 'tau0']},
                                  dims=['bootstrap', 'param'])
            bootstrap_params.append(params)
        bootstrap_params = merge_data_arrays(bootstrap_params)
        # find endpoint and error
        asymptote_threshold = .0005
        interpolation_xs = np.arange(1000)
        ys = np.array([v(interpolation_xs, *params) for params in bootstrap_params.values
                       if not np.isnan(params).any()])
        median_ys = np.median(ys, axis=0)
        diffs = np.diff(median_ys)
        end_x = np.where(diffs < asymptote_threshold)[0].min()  # first x where increase smaller than threshold
        # put together
        center = np.median(np.array(bootstrap_params)[:, 0])
        error = ci_error(ys[:, end_x], center=center)
        score = Score([center] + list(error),
                      coords={'aggregation': ['center', 'error_low', 'error_high']}, dims=['aggregation'])
        score.attrs['raw'] = ceilings
        score.attrs['bootstrapped_params'] = bootstrap_params
        score.attrs['endpoint_x'] = DataAssembly(end_x)
        return score

    def fit(self, subject_subsamples, bootstrapped_scores):
        params, pcov = curve_fit(v, subject_subsamples, bootstrapped_scores,
                                 # v (i.e. max ceiling) is between 0 and 1, tau0 unconstrained
                                 bounds=([0, -np.inf], [1, np.inf]))
        return params

    def post_process(self, scores):
        if self._post_process is not None:
            scores = self._post_process(scores)
        return scores


def ci_error(samples, center, confidence=.95):
    low, high = 100 * ((1 - confidence) / 2), 100 * (1 - ((1 - confidence) / 2))
    confidence_below, confidence_above = np.nanpercentile(samples, low), np.nanpercentile(samples, high)
    confidence_below, confidence_above = center - confidence_below, confidence_above - center
    return confidence_below, confidence_above


def manual_merge(*elements, on='neuroid'):
    dims = elements[0].dims
    assert all(element.dims == dims for element in elements[1:])
    merge_index = dims.index(on)
    # the coordinates in the merge index should have the same keys
    assert _coords_match(elements, dim=on,
                         match_values=False), f"coords in {[element[on] for element in elements]} do not match"
    # all other dimensions, their coordinates and values should already align
    for dim in set(dims) - {on}:
        assert _coords_match(elements, dim=dim,
                             match_values=True), f"coords in {[element[dim] for element in elements]} do not match"
    # merge values without meta
    merged_values = np.concatenate([element.values for element in elements], axis=merge_index)
    # piece together with meta
    result = type(elements[0])(merged_values, coords={
        **{coord: (dims, values)
           for coord, dims, values in walk_coords(elements[0])
           if not array_is_element(dims, on)},
        **{coord: (dims, np.concatenate([element[coord].values for element in elements]))
           for coord, dims, _ in walk_coords(elements[0])
           if array_is_element(dims, on)}}, dims=elements[0].dims)
    return result


def _coords_match(elements, dim, match_values=False):
    first_coords = [(key, tuple(value)) if match_values else key for _, key, value in walk_coords(elements[0][dim])]
    other_coords = [[(key, tuple(value)) if match_values else key for _, key, value in walk_coords(element[dim])]
                    for element in elements[1:]]
    return all(tuple(first_coords) == tuple(coords) for coords in other_coords)


class NoOverlapException(Exception):
    pass



class FewSubjectExtrapolation:
    def __init__(self, subject_column,extrapolation_dimension,post_process,num_subsamples=200,num_bootstraps=100, *args, **kwargs):
        super(FewSubjectExtrapolation, self).__init__(*args, **kwargs)
        self._rng = RandomState(0)
        self._num_subsamples = num_subsamples   # number of subsamples per subject selection ( this is used to reduce the computational cost)
        #self.holdout_ceiling = HoldoutSubjectCeiling(subject_column=subject_column)
        self._logger = logging.getLogger(fullname(self))
        self.subject_column = subject_column
        self.extrapolation_dimension = extrapolation_dimension
        self.num_bootstraps = num_bootstraps
        self._post_process = post_process
        self.epsilon=1e-10

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
            # select a set of subjects combination upto _num_subsamples , for example 200 subject combination for 10 subjects,
            selection_combinations = self.iterate_subsets(assembly, num_subjects=num_subjects)
            comb_scores=[]
            for selections, sub_assembly in tqdm(selection_combinations, desc='selections'):
                sub_scores=[]
                iterate_subjects = np.unique(sub_assembly[self.subject_column].values)
                for subject in tqdm(iterate_subjects, desc='subjects',disable=True):
                    # select subject data from sub_assembly
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
        return scores

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
                    num_subject_row = (neuroid_ceiling.subsample == num_subjects).values
                    # select only the center row
                    num_scores = neuroid_ceiling.isel(subsample=num_subject_row)
                    #num_scores = neuroid_ceiling.sel(subsample=num_subjects)
                    choices = num_scores.values.flatten()
                    bootstrapped_score = rng.choice(choices, size=len(choices), replace=True)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
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
                    params,R_squared = self.fit(x, y)
                    # if R_squared is negative or less than self.epsilon, then the fit is not good so replace it with nan
                    # if R_squared<self.epsilon:
                    #     params=[np.nan,np.nan]
                except:
                    params = [np.nan, np.nan]
                    R_squared=np.nan
                params = DataAssembly([params], coords={'bootstrap': [bootstrap], 'param': ['v0', 'tau0']},
                                      dims=['bootstrap', 'param'])
                params.attrs['R_squared']=R_squared
                bootstrap_params.append(params)
            # get R_squared values
            R_squared_values = np.array([x.attrs['R_squared'] for x in bootstrap_params])
            bootstrap_params = merge_data_arrays(bootstrap_params)
            # add it back to bootstrap_params
            bootstrap_params.attrs['R_squared']=R_squared_values
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
        # ehoseini: there was a bug in the original code that resulted in subjects being in order, now added a shuffle in the end
        subjects = np.array(list(subjects))
        combinations = set()
        # find the maximum number of combinations possible
        max_choice = comb(len(subjects), num_subjects)
        while len(combinations) < min(choice,max_choice):
            elements = rng.choice(subjects, size=num_subjects, replace=False)
            combinations.add(tuple(elements))
        # shuffle combinations
        combinations = list(combinations)
        rng.shuffle(combinations)
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

    def fit(self, x, y):
        x_data= np.array(x)
        y_data=np.array(y)
        valid = ~np.isnan(y)
        x_data=x_data[valid]
        y_data=y_data[valid]
        tau_bound=(0.001,int(max(x_data)*10)) # here for tau=0.5 the model already close to the ceiling for x=2, so we set the lower bound to 0.5


        if sum(valid) < 1:
            raise RuntimeError("No valid scores in sample")
        # note that bounds are in the format : (lower_bounds,upper_bounds)
        params, pcov = curve_fit(v_overflow, x_data, y_data,check_finite=True,
                                 # v (i.e. max ceiling) is between 0 and 1, tau0
                                 bounds=([0, tau_bound[0]], [1,tau_bound[1]]),nan_policy='omit')

        y_pred = v(x_data, *params)
        residuals = y_data - y_pred
        SS_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        SS_res = np.sum(residuals ** 2)
        # Calculate R-squared
        R_squared = 1 - (SS_res / SS_tot)
        return params, R_squared