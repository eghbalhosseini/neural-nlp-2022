import argparse
import fire
import logging
import sys
from datetime import datetime

from neural_nlp import score as score_function

_logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--log_level', type=str, default='INFO')
FLAGS, FIRE_FLAGS = parser.parse_known_args()
logging.basicConfig(stream=sys.stdout, level=logging.getLevelName(FLAGS.log_level))
_logger.info(f"Running with args {FLAGS}, {FIRE_FLAGS}")
for ignore_logger in ['transformers.data.processors', 'botocore', 'boto3', 'urllib3', 's3transfer']:
    logging.getLogger(ignore_logger).setLevel(logging.INFO)


def run(benchmark, model, layers=None, subsample=None):
    start = datetime.now()
    
    # add to not overwite score files
    if os.getenv('SPLIT_AT_PASSAGE', '0') == '1':
        split_coord = "Passage"
    elif os.getenv('SPLIT_AT_TOPIC', '0') == '1':
        split_coord = "Topic"
    else:
        split_coord = "Sentence"
        
    score = score_function(model=model, layers=layers, subsample=subsample, benchmark=benchmark, split_coord=split_coord)
    end = datetime.now()
    print(score)
    print(f"Duration: {end - start}")


if __name__ == '__main__':
    import warnings

    warnings.simplefilter(action='ignore', category=FutureWarning)
    fire.Fire(command=FIRE_FLAGS)
