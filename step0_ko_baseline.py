import pickle
import os
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from omegaconf import OmegaConf
import joblib
from konlpy.tag import Kkma, Okt
import tqdm


cfg = OmegaConf.load("conf/config.yaml")

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

okt = Okt()
train['nounBody'] = [ okt.nouns(sentence) for sentence in tqdm.tqdm(train['mergeBody'].tolist())]
valid['nounBody'] = [okt.nouns(sentence) for sentence in tqdm.tqdm(valid['mergeBody'].tolist())]
test['nounBody'] = [okt.nouns(sentence) for sentence in tqdm.tqdm(test['mergeBody'].tolist())]

train['nounBody'].str.join()


# import cudf, cuml, cupy
# from cuml.feature_extraction.text import TfidfVectorizer
# from cuml.neighbors import NearestNeighbors

from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer

model = TfidfVectorizer(
    # preprocessor=,
    # stop_words=[],
    binary=True,
    )

train_embeddings = model.fit_transform(train.mergeBody).toarray()
valid_embeddings = model.transform(valid.mergeBody).toarray()
test_embeddings = model.transform(test.mergeBody).toarray()

KNN = 10
model = NearestNeighbors(n_neighbors=KNN)
model.fit(train_embeddings)


from statistics import mode
from sklearn.metrics import accuracy_score

# evaluate valid set
distances, indices = model.kneighbors(valid_embeddings)
yhat = [mode(row) for row in train.category.values[indices]]
valid_score = accuracy_score(valid.category, yhat)
assert valid_score >= 0.7


# evaluate test set
distances, indices = model.kneighbors(test_embeddings)
yhat = [mode(row) for row in train.category.values[indices]]
test_score = accuracy_score(test.category, yhat)
assert test_score >= 0.7

