import os
from glob import glob

import logging
import numpy as np
import scipy.io as sio
import xarray as xr
from brainio.assemblies import NeuroidAssembly
from pathlib import Path
from scipy import stats

from neural_nlp.stimuli import StimulusSet
from result_caching import cache, store, store_netcdf
_logger = logging.getLogger(__name__)


neural_data_dir = (Path(os.path.dirname(__file__)) / '..' / '..' / 'ressources' / 'neural_data' / 'mgh_ecog').resolve()

def load_MghMockLang():
    data_dir= neural_data_dir / "MghMockLang"
    _logger.info(f'Neural data directory: {data_dir}')
    data = xr.open_dataset(os.path.join(data_dir, 'mock_stim_resp_elec_data_gpt2_layer_11.nc')).squeeze()
    data = data.to_array().squeeze()
    subj_names = data.subject_id.values
    subjectID = np.sum([idx * (subj_names == x) for idx, x in enumerate(np.unique(subj_names))], axis=0)
    word_number = list(range(data.shape[0]))
    electrode_numbers = list(range(data.shape[1]))
    word_nums = data.word_id.values
    # Add a pd df as the stimulus_set
    zipped_lst = list(zip(data.sentence_id.values, word_number, data.word.values,data.word_id.values))
    df_stimulus_set = StimulusSet(zipped_lst, columns=['sentence_id', 'stimulus_id', 'word','word_id'])
    df_stimulus_set.name = 'MghMockLang'
    assembly = xr.DataArray(data.values,
                            dims=('presentation', 'neuroid'),
                            coords={'stimulus_id': ('presentation', word_number),
                                    'word': ('presentation', data.word.values),
                                    'word_num': ('presentation', word_nums),
                                    'sentence_id': ('presentation', data.sentence_id.values),
                                    'electrode': ('neuroid', electrode_numbers),
                                    'neuroid_id': ('neuroid', electrode_numbers),
                                    'subject_UID': ('neuroid', subjectID),  # Name is subject_UID for consistency
                                    'subject_name': ('neuroid', subj_names)
                                    })

    assembly.attrs['stimulus_set'] = df_stimulus_set  # Add the stimulus_set dataframe
    #data = assembly if data is None else xr.concat(data, assembly)
    return NeuroidAssembly(assembly)