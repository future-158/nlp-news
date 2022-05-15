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


from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

model = SentenceTransformer('distilbert-base-nli-mean-tokens')
embeddings = model.encode(data, show_progress_bar=True)



norm_data = normalize(data, norm='l2')
clusterer = HDBSCAN() # euclidean distance is the default
clusterer.fit(norm_data)


from gensim import models
import cupy as cp

from cuml.manifold import UMAP
from cuml.preprocessing import normalize
from cuml.cluster import HDBSCAN

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

umap_n_neighbors=100
umap_min_dist=1e-3
umap_spread=2.0
umap_n_epochs=500
umap_random_state=42

hdbscan_min_samples=25
hdbscan_min_cluster_size=5
hdbscan_max_cluster_size=1000
hdbscan_cluster_selection_method="leaf"

%%time
X = normalize(cp.array(w.vectors))[:n_points]


%%time
umap = UMAP(n_neighbors=umap_n_neighbors, 
            min_dist=umap_min_dist, 
            spread=umap_spread, 
            n_epochs=umap_n_epochs, 
            random_state=umap_random_state)

embeddings = umap.fit_transform(X)

%%time
hdbscan = HDBSCAN(min_samples=hdbscan_min_samples, 
                  min_cluster_size=hdbscan_min_cluster_size, 
                  max_cluster_size=hdbscan_max_cluster_size,
                  cluster_selection_method=hdbscan_cluster_selection_method)

labels = hdbscan.fit_predict(X)


x = embeddings[:,0]
y = embeddings[:,1]

x = x[labels>-1]
y = y[labels>-1]

labels_nonoise = labels[labels>-1]

x_noise = embeddings[:,0]
y_noise = embeddings[:,1]

x_noise = x_noise[labels==-1]
y_noise = y_noise[labels==-1]

from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('distiluse-base-multilingual-cased-v1', device='cuda')
model.max_seq_length = 500

sentences = ['This framework generates embeddings for each input sentence',
    'Sentences are passed as a list of string.', 
    'The quick brown fox jumps over the lazy dog.']*100

query_embedding = model.encode(
    'How many people live in London?',
    batch_size = 32,
    output_value = 'token_embeddings',
    convert_to_numpy = True,
    # convert_to_tensor = False,
    device = 'cuda',
    normalize_embeddings=True
    )

#The passages are encoded as [ [title1, text1], [title2, text2], ...]
passage_embedding = model.encode([['London', 'London has 9,787,426 inhabitants at the 2011 census.']])

print("Similarity:", util.cos_sim(query_embedding, passage_embedding))
