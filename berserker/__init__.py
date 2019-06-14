from berserker.utils import maybe_download_unzip
from pathlib import Path
import tensorflow as tf
import numpy as np
import pandas as pd
import time

ASSETS_PATH = str(Path(__file__).parent / 'assets')
_models_path = Path(__file__).parent / 'models'

from berserker.transform import batch_preprocess, batch_postprocess

MAX_SEQ_LENGTH = 512
SEQ_LENGTH = MAX_SEQ_LENGTH - 2
BATCH_SIZE = 80

def load_model(model_name=None, verbose=True, force_download=False):
    maybe_download_unzip(
        'https://github.com/Hoiy/berserker/releases/download/v0.1-alpha/1547563491.zip',
        _models_path,
        verbose,
        force_download
    )


def tokenize(text):
    load_model()
    texts = [text]
    bert_inputs, mappings, sizes = batch_preprocess(texts, MAX_SEQ_LENGTH, BATCH_SIZE)

    berserker = tf.contrib.predictor.from_saved_model(
        str(_models_path / '1547563491')
    )
    bert_outputs = berserker(bert_inputs)
    bert_outputs = [{'predictions': bo} for bo in bert_outputs['predictions']]

    return batch_postprocess(texts, mappings, sizes, bert_inputs, bert_outputs, MAX_SEQ_LENGTH)[0]


def process_me(manifest_path):
    load_model()
    berserker = tf.contrib.predictor.from_saved_model(
        str(_models_path / '1547563491')
    )
    df = pd.read_csv(manifest_path, encoding="utf-8")
    total_recs = df.shape[0]
    iter_count = int(total_recs/BATCH_SIZE) + 1
    for i in range(iter_count):
        if i == 0:
            print("Started batch", time.time())
        start_time = time.time()
        split = df[BATCH_SIZE*i: BATCH_SIZE*(i+1)]
        texts = [line for line in split["txt"].tolist() if type(line) == str and len(line) > 3]

        bert_inputs, mappings, sizes = batch_preprocess(texts, MAX_SEQ_LENGTH, BATCH_SIZE)
        bert_outputs = berserker(bert_inputs)
        bert_outputs = [{'predictions': bo} for bo in bert_outputs['predictions']]
        results = batch_postprocess(texts, mappings, sizes, bert_inputs, bert_outputs, MAX_SEQ_LENGTH)
        with open("/nfs/alldata/clients/dell/data/processed/tokens/splits-" + str(i) + ".txt", "w", encoding="utf-8") as f:
            f.write(str(results))
        print("Completed batch", str(i), str(time.time - start_time))


