# src/wisdom/clustering/factory.py
from sklearn.cluster import (
    KMeans, MiniBatchKMeans, AgglomerativeClustering, MeanShift,
    DBSCAN, OPTICS, Birch, SpectralClustering
)

NAME2CLS = {
    "kmeans": KMeans,
    "minibatchkmeans": MiniBatchKMeans,
    "agglomerativeclustering": AgglomerativeClustering,
    "meanshift": MeanShift,
    "dbscan": DBSCAN,
    "optics": OPTICS,
    "birch": Birch,
    "spectralclustering": SpectralClustering,
}

def make(name: str, **params):
    key = (name or "KMeans").lower()
    if key not in NAME2CLS:
        raise ValueError(f"Unknown clustering method: {name}. Available: {list(NAME2CLS)}")
    return NAME2CLS[key](**params)