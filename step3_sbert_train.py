from sentence_transformers import SentenceTransformer
import pickle
import os
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from omegaconf import OmegaConf
import joblib
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from statistics import mode
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC, LinearSVC

import tensorflow as tf
from tensorflow import keras
# import tensorflow_hub as hub
import tensorflow_decision_forests as tfdf

cfg = OmegaConf.load("conf/config.yaml")


df = pd.read_csv(
    cfg.catalog.translated
)


valid_index = (
    df
    [df.split!='test']
    .groupby('category')
    .sample(frac=0.2) # 80 20 split
    .index
)

valid_mask = df.index.isin(valid_index)
df.loc[valid_mask, 'split'] = 'valid'

cls2idx = {k:i for i, k in enumerate(sorted(df.category.unique()))}
labels = df.category.map(cls2idx).values
test_mask = df.split == 'test'


model = SentenceTransformer('all-distilroberta-v1', device='cuda')
model.max_seq_length = 500

encode_kwargs = dict(
        batch_size = 32,
        # output_value = 'token_embeddings',
        convert_to_numpy = True,
        show_progress_bar=True,
        device = 'cuda',
        normalize_embeddings=True
)

embeddings = model.encode(
    df.en_sentence.values,
    **encode_kwargs
    )

train_mask = np.logical_and(~valid_mask, ~test_mask)

# fit svc
model = LinearSVC()
model.fit(embeddings[train_mask], labels[train_mask])

valid_pred = model.predict(embeddings[valid_mask])
valid_acc = accuracy_score(
    labels[valid_mask], valid_pred
)
assert valid_acc > 0.7


# refit with train + valid
train_valid_mask = np.logical_or(train_mask, valid_mask)
model = LinearSVC()
model.fit(embeddings[train_valid_mask], labels[train_valid_mask])


test_pred = model.predict(embeddings[test_mask])
test_acc = accuracy_score(
    labels[test_mask], test_pred
)
assert test_acc > 0.7

OmegaConf.save(
    OmegaConf.create(
    {
    'accuracy': {
        'test': float(test_acc),
    }
    }), cfg.catalog.metric.en_base
    )


idx2cls = {v:k for k,v in cls2idx.items()}


df.loc[test_mask, 'result'] = pd.Series(test_pred).map(idx2cls).values
df.to_csv(
        cfg.catalog.output.en_base, index=False
    )
