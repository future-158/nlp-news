from sklearn import datasets
from cuml.cluster import HDBSCAN

X, y = datasets.make_moons(n_samples=50, noise=0.05)

model = HDBSCAN(min_samples=5, gen_min_span_tree=True)
y_hat = model.fit_predict(X)

model.minimum_spanning_tree_.plot()