import cudf, cuml, cupy
from cuml.feature_extraction.text import TfidfVectorizer
from cuml.neighbors import NearestNeighbors
print('RAPIDS',cuml.__version__)

train_gf = cudf.read_csv('../input/shopee-product-matching/train.csv')
print('train shape is', train_gf.shape )
train_gf.head()

model = TfidfVectorizer(stop_words='english', binary=True)
text_embeddings = model.fit_transform(train_gf.title).toarray()
print('text embeddings shape is',text_embeddings.shape)

KNN = 50
model = NearestNeighbors(n_neighbors=KNN)
model.fit(text_embeddings)
distances, indices = model.kneighbors(text_embeddings)


from sentence_transformers import SentenceTransformer
model = SentenceTransformer('distilbert-base-nli-mean-tokens')
embeddings = model.encode(data, show_progress_bar=True)

from sklearn.preprocessing import normalize

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
