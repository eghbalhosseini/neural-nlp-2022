import logging
import numpy as np
from brainio.assemblies import walk_coords
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit, KFold, StratifiedKFold, GroupKFold

from brainscore.metrics import Score
from brainscore.utils import fullname
import os

from brainscore.metrics.transformations import extract_coord, standard_error_of_the_mean

import torch
import random

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)


class SplitNew:
    class Defaults:
        splits = 10
        train_size = .9
        split_coord = 'image_id'
        stratification_coord = 'object_name'
        unique_split_values = False
        random_state = 1

    def __init__(self, splits=Defaults.splits, train_size=None, test_size=None,
                 split_coord=Defaults.split_coord, stratification_coord=Defaults.stratification_coord, kfold=False,
                 unique_split_values=Defaults.unique_split_values, random_state=Defaults.random_state, assembly=None):
        super().__init__()
        self._logger = logging.getLogger(fullname(self))
        if train_size is None and test_size is None:
            train_size = self.Defaults.train_size
        if kfold:
            if os.getenv("SPLIT_AT_TOPIC", "0") == "1":
                assert (train_size is None or train_size == self.Defaults.train_size) and test_size is None
                print("Using GroupKFold for topics with split coordinate {}!".format(split_coord))
                self._split = GroupKFold(n_splits=splits)
            elif os.getenv("SPLIT_AT_PASSAGE", "0") == "1":
                assert (train_size is None or train_size == self.Defaults.train_size) and test_size is None
                print("Using GroupKFold for passages with split coordinate {}!".format(split_coord))
                self._split = GroupKFold(n_splits=splits)
            else:
                assert (train_size is None or train_size == self.Defaults.train_size) and test_size is None
                if stratification_coord:
                    self._split = StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_state)
                else:
                    self._split = KFold(n_splits=splits, shuffle=True, random_state=random_state)
        else:
            if stratification_coord:
                self._split = StratifiedShuffleSplit(
                    n_splits=splits, train_size=train_size, test_size=test_size, random_state=random_state)
            else:
                self._split = ShuffleSplit(
                    n_splits=splits, train_size=train_size, test_size=test_size, random_state=random_state)
        self._split_coord = split_coord
        self._stratification_coord = stratification_coord
        self._unique_split_values = unique_split_values

    @property
    def do_stratify(self):
        return bool(self._stratification_coord)

    def build_splits(self, assembly):
        cross_validation_values, indices = extract_coord(assembly, self._split_coord, unique=self._unique_split_values)
        data_shape = np.zeros(len(cross_validation_values))
        args = [assembly[self._stratification_coord].values[indices]] if self.do_stratify else []
        
        if os.getenv("SPLIT_AT_PASSAGE", "0") == "1" or os.getenv("SPLIT_AT_TOPIC", "0") == "1":
            groups = list(assembly._split_coord.data)
            splits = self._split.split(data_shape, groups=groups, *args)
            
        else:
            splits = self._split.split(data_shape, *args)

        return cross_validation_values, list(splits)

    @classmethod
    def aggregate(cls, values):
        center = values.mean('split')
        error = standard_error_of_the_mean(values, 'split')
        return Score([center, error],
                     coords={**{'aggregation': ['center', 'error']},
                             **{coord: (dims, values) for coord, dims, values in walk_coords(center)}},
                     dims=('aggregation',) + center.dims)