from sentence_transformers import SentenceTransformer
import pickle
import os
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from omegaconf import OmegaConf
import joblib

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


from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('distiluse-base-multilingual-cased-v1', device='cuda')
# model = SentenceTransformer('distilbert-base-nli-mean-tokens', device='cuda')
model.max_seq_length = 500



encode_kwargs = dict(
        batch_size = 32,
        # output_value = 'token_embeddings',
        convert_to_numpy = True,
        # convert_to_tensor = False,
        show_progress_bar=True,
        device = 'cuda',
        normalize_embeddings=True

)

train_embeddings = model.encode(
    train.mergeBody.values,
    **encode_kwargs
    )
valid_embeddings = model.encode(
    valid.mergeBody.values,
    **encode_kwargs
    )
test_embeddings = model.encode(
    test.mergeBody.values,
    **encode_kwargs
    )


# umap testing

import cupy as cp
from cuml.manifold import UMAP
from cuml.preprocessing import normalize
from cuml.cluster import HDBSCAN
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

train_embeddings_cp = normalize(cp.array(train_embeddings))
valid_embeddings_cp = normalize(cp.array(valid_embeddings))
test_embeddings_cp = normalize(cp.array(test_embeddings))

# smote_nc = SMOTENC(categorical_features=[0, 2], random_state=0)
# X_resampled, y_resampled = smote_nc.fit_resample(X, y)
# print(sorted(Counter(y_resampled).items()))
# print(X_resampled[-5:])

umap = UMAP(
    # n_neighbors=umap_n_neighbors, 
            random_state=1
            )

train_embeddings_umap = umap.fit_transform(train_embeddings_cp)
valid_embeddings_umap = umap.transform(valid_embeddings_cp)
test_embeddings_umap = umap.transform(test_embeddings_cp)

cls2idx = {k:i for i, k in enumerate(sorted(train.category.unique()))}

train_labels = train.category.map(cls2idx).values
valid_labels = valid.category.map(cls2idx).values
test_labels = test.category.map(cls2idx).values

x = cp.float32(train_embeddings_umap[:,0].get())
y = cp.float32(train_embeddings_umap[:,1].get())

plt.scatter(x,y, c=train_labels)
plt.legend()


# hdbscan

# hdbscan_min_samples=30
# hdbscan_min_cluster_size=5

clusterer = HDBSCAN(
    min_samples=5,
    min_cluster_size=15,
    output_type='numpy',
    metric = 'eucleadian'
                  )


X = cp.concatenate(
        [
            train_embeddings_cp,
            valid_embeddings_cp,
            test_embeddings_cp
        ]
)

y = cp.asarray(
    np.concatenate([train_labels, valid_labels, test_labels])
    )

cp.bincount(y)


clusterer.fit(
    X,y
    )


pd.Series(clusterer.labels_).value_counts()




from cuml.neighbors import NearestNeighbors
KNN = 10
model = NearestNeighbors(n_neighbors=KNN)
model.fit(train_embeddings)


from statistics import mode
from sklearn.metrics import accuracy_score

# evaluate valid set
distances, indices = model.kneighbors(valid_embeddings)
indices = cupy.float32(indices.get())
indices = np.asarray(indices)
indices = indices.astype(int)
yhat = [mode(row) for row in train.to_pandas()['category'].values[indices]]
valid_score = accuracy_score(valid.category.to_pandas(), yhat)
assert valid_score >= 0.7

