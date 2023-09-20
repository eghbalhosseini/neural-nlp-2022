import numpy as np
import scipy.stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale

from brainio.assemblies import NeuroidAssembly, array_is_element, DataAssembly
from brainio.assemblies import walk_coords
from brainscore.metrics import Score, Metric
import rsatoolbox

class Defaults:
    expected_dims = ('presentation', 'neuroid')
    stimulus_coord = 'stimulus_id'
    neuroid_dim = 'neuroid'
    neuroid_coord = 'neuroid_id'

class XarrayRSA:
    def __init__(self, expected_dims=Defaults.expected_dims, neuroid_dim=Defaults.neuroid_dim,
                 neuroid_coord=Defaults.neuroid_coord, stimulus_coord=Defaults.stimulus_coord):
        self._rsa = None
        self._expected_dims = expected_dims
        self._neuroid_dim = neuroid_dim
        self._neuroid_coord = neuroid_coord
        self._stimulus_coord = stimulus_coord
        self._target_neuroid_values = None


    def _estimate_dissimilarity(self,metric='eculidean'):
        NotImplementedError

    def _subsample(self):
        NotImplementedError

    def _split(self):
        NotImplementedError

    def _sort(self):
        NotImplementedError

    def _reodrder(self):
        NotImplementedError


    def _define_rsa_model(self,type=None):
        NotImplementedError

    def __call__(self,source,target, *args, **kwargs):
        # 1. turn get rsa dissimilarities
        #TODO : make this more flexible to do more type of rdm
        #target_rdm = rsatoolbox.rdm.calc_rdm(target, method='crossnobis',descriptor='stimulus_id',cv_descriptor='session')
        target_rdm = rsatoolbox.rdm.calc_rdm(target, method='correlation', descriptor='stimulus_id')
        # 2. get source and turn into rdm
        source_rdm = rsatoolbox.rdm.calc_rdm(source, method='correlation')
        model_dec=source.descriptors['model']
        source_model = rsatoolbox.rdm.RDMs(source_rdm.dissimilarities,
                                             rdm_descriptors={'brain_computational_model': f'{model_dec}'},
                                             dissimilarity_measure='correlation')

        fixed_model = rsatoolbox.model.ModelFixed(model_dec, source_model)
        results_ = rsatoolbox.inference.eval_fixed(fixed_model, target_rdm, method='corr')
        return results_



def rsa_correlation(*args, **kwargs):
    return XarrayRSA(*args, **kwargs)