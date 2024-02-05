
from neural_nlp.benchmarks.new_split import SplitNew as Split #used for new split_by_passage_id functionality
from brainscore.metrics.transformations import Transformation #, Split
from brainscore.metrics import Score
from tqdm import tqdm
import logging
from brainscore.utils import fullname

import numpy as np
import torch
import random
import re

import os
import pickle

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

# source: https://raw.githubusercontent.com/brain-score/brainio_collection/master/brainio_collection/transform.py
def subset(source_assembly, target_assembly, subset_dims=None, dims_must_match=True, repeat=False):
    """
    Returns the subset of the source_assembly whose coordinates align with those specified by target_assembly.
    Ordering is not guaranteed.
    :param subset_dims: either dimensions, then all its levels will be used or levels right away
    :param dims_must_match:
    :return:
    """
    subset_dims = subset_dims or target_assembly.dims
    for dim in subset_dims:
        assert hasattr(target_assembly, dim)
        assert hasattr(source_assembly, dim)
        # we assume here that it does not matter if all levels are present in the source assembly
        # as long as there is at least one level that we can select over
        levels = target_assembly[dim].variable.level_names or [dim]
        assert any(hasattr(source_assembly, level) for level in levels)
        for level in levels:
            if not hasattr(source_assembly, level):
                continue
            target_values = target_assembly[level].values
            source_values = source_assembly[level].values
            if repeat:
                indexer = index_efficient(source_values, target_values)
            else:
                indexer = np.array([val in target_values for val in source_values])
                indexer = np.where(indexer)[0]
            if dim not in target_assembly.dims:
                # not actually a dimension, but rather a coord -> filter along underlying dim
                dim = target_assembly[dim].dims
                assert len(dim) == 1
                dim = dim[0]
            dim_indexes = {_dim: slice(None) if _dim != dim else indexer for _dim in source_assembly.dims}
            if len(np.unique(source_assembly.dims)) == len(source_assembly.dims):  # no repeated dimensions
                source_assembly = source_assembly.isel(**dim_indexes)
                continue
            # work-around when dimensions are repeated. `isel` will keep only the first instance of a repeated dimension
            positional_dim_indexes = [dim_indexes[dim] for dim in source_assembly.dims]
            coords = {}
            for coord, dims, value in walk_coords(source_assembly):
                if len(dims) == 1:
                    coords[coord] = (dims, value[dim_indexes[dims[0]]])
                elif len(dims) == 0:
                    coords[coord] = (dims, value)
                elif len(set(dims)) == 1:
                    coords[coord] = (dims, value[np.ix_(*[dim_indexes[dim] for dim in dims])])
                else:
                    raise NotImplementedError("cannot handle multiple dimensions")
            source_assembly = type(source_assembly)(source_assembly.values[np.ix_(*positional_dim_indexes)],
                                                    coords=coords, dims=source_assembly.dims)
        if dims_must_match:
            # dims match up after selection. cannot compare exact equality due to potentially missing levels
            assert len(target_assembly[dim]) == len(source_assembly[dim])
    return source_assembly


class CrossRegressedCorrelation:
    def __init__(self, regression, correlation, crossvalidation_kwargs=None):
        regression = regression
        crossvalidation_defaults = dict(train_size=.9, test_size=None)
        crossvalidation_kwargs = {**crossvalidation_defaults, **(crossvalidation_kwargs or {})}

        self.cross_validation = CrossValidation(**crossvalidation_kwargs)
        self.regression = regression
        self.correlation = correlation

    def __call__(self, source, target):
        return self.cross_validation(source, target, apply=self.apply, aggregate=self.aggregate)

    def apply(self, source_train, target_train, source_test, target_test):
        self.regression.fit(source_train, target_train)
        prediction = self.regression.predict(source_test)
        score = self.correlation(prediction, target_test)
        return score

    def aggregate(self, scores):
        return scores.median(dim='neuroid')


class CrossValidation(Transformation):
    """
    Performs multiple splits over a source and target assembly.
    No guarantees are given for data-alignment, use the metadata.
    """

    def __init__(self, *args, split_coord=Split.Defaults.split_coord,
                 stratification_coord=Split.Defaults.stratification_coord, **kwargs):
        self._logger = logging.getLogger(fullname(self))
        self._split_coord = split_coord #the explicit keyword argument in the function call (unpacked from crossvalidation_kwargs) has precedence over the default value specified in the function definition.
        self._stratification_coord = stratification_coord
        self._split = Split(*args, split_coord=split_coord, stratification_coord=stratification_coord, **kwargs)

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
                in tqdm(enumerate_done(splits), total=len(splits), desc='cross-validation'):
            layer_identifier = np.unique(source_assembly.layer.data)[0]
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



def enumerate_done(values):
    for i, val in enumerate(values):
        done = i == len(values) - 1
        yield i, val, done
