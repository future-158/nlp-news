import pickle
import os
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from omegaconf import OmegaConf
import joblib
import tqdm
from statistics import mode
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

cfg = OmegaConf.load("conf/config.yaml")

for folder in cfg.dirs:
    Path(folder).mkdir(parents=True, exist_ok=True)

# load train, test
train_valid = pd.read_csv( 
    cfg.catalog.raw.train
)

test = pd.read_csv( 
    cfg.catalog.raw.test
)

# title, cleanBody, category
train_valid.category.value_counts()
test.category.value_counts()


# train test valid split
train = train_valid.groupby("category").sample(frac=0.8, random_state=1)
valid_mask = ~train_valid.index.isin(train.index)
valid = train_valid[valid_mask]

train['mergeBody'] = train.title.str.strip() + ' ' +  train.cleanBody.str.strip()
valid['mergeBody'] = valid.title.str.strip() + ' ' +  valid.cleanBody.str.strip()
test['mergeBody'] = test.title.str.strip() + ' ' +  test.cleanBody.str.strip()

# okt = Okt()
# train['nounBody'] = [ okt.nouns(sentence) for sentence in tqdm.tqdm(train['mergeBody'].tolist())]
# valid['nounBody'] = [okt.nouns(sentence) for sentence in tqdm.tqdm(valid['mergeBody'].tolist())]
# test['nounBody'] = [okt.nouns(sentence) for sentence in tqdm.tqdm(test['mergeBody'].tolist())]

# import cudf, cuml, cupy
# from cuml.feature_extraction.text import TfidfVectorizer
# from cuml.neighbors import NearestNeighbors


model = TfidfVectorizer(
    # preprocessor=,
    # stop_words=[],
    binary=True,
    )

train_embeddings = model.fit_transform(train.mergeBody).toarray()
valid_embeddings = model.transform(valid.mergeBody).toarray()
test_embeddings = model.transform(test.mergeBody).toarray()


# fit svc
model = LinearSVC()
model.fit(train_embeddings, train.category)

valid_pred = model.predict(valid_embeddings)
valid_acc = accuracy_score(
    valid.category, valid_pred
)
assert valid_acc > 0.7

# refit with train + valid
model = LinearSVC()
model.fit(
    np.concatenate([train_embeddings, valid_embeddings]),
    np.concatenate([train.category, valid.category])

)

test_pred = model.predict(test_embeddings)
test_acc = accuracy_score(
    test.category, test_pred
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

