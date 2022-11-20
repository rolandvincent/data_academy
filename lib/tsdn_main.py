import numpy as np
import pandas as pd
import scipy

# These are the visualization libraries. Matplotlib is standard and is what most people use.
# Seaborn works on top of matplotlib, as we mentioned in the course.
import matplotlib.pyplot as plt
import seaborn as sns
# sns.set()
# For standardizing features. We'll use the StandardScaler module.
from sklearn.preprocessing import StandardScaler
# Hierarchical clustering with the Sci Py library. We'll use the dendrogram and linkage modules.
from scipy.cluster.hierarchy import dendrogram, linkage
# Sk learn is one of the most widely used libraries for machine learning. We'll use the k means and pca modules.
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
# We need to save the models, which we'll use in the next section. We'll use pickle for that.
import pickle


class TSDN:
    def __init__(self) -> None:
        self.scaler = pickle.load(open('files/scaler.pickle', 'rb'))
        self.pca = pickle.load(open('files/pca.pickle', 'rb'))
        self.kmeans_pca = pickle.load(
            open('files/kmeans_pca.pickle', 'rb'))

    def predict_segment(self, data):
        x_scaled = self.scaler.transform(data)
        x_pca = self.pca.transform(x_scaled)

        segm_kmeans_pcanew = self.kmeans_pca.predict(x_pca)
        predict_values = dict()
        legend = {0: 'Prosperous Customer',  # 'highly-spender-woman-hgi',
                  1: 'Standard Customer',  # 'thrifty-spender-man',
                  2: 'Rare Customer',  # 'highly-spender-woman-lwi',
                  3: 'Active Customer',  # 'thrifty-spender-woman',
                  4: 'Top Customer'}  # 'highly-spender-man'}
        description = {
            0: [
                "Customer ini mengcover 18.5% dari total populasi customer.",
                "High income dan spending score",
                "Memiliki rasio spending dan Income 79.01%"
            ],
            1: [
                "Customer yang umum, mengcover 24% dari total populasi customer.",
                "Standard income dan spending score",
                "Memiliki rasio spending dan Income 46.7%"
            ],
            2: [
                "Customer yang paling langka, mengcover 12.5% dari total populasi customer.",
                "Low income dan low spending score",
                "Memiliki rasio spending dan Income 49.2%"
            ],
            3: [
                "Customer yang paling umum, mengcover 25% dari total populasi customer.",
                "Standard income dan spending score",
                "Memiliki rasio spending dan Income 62.5%"
            ],
            4: [
                "Customer ini mengcover 20% dari total populasi customer.",
                "High income dan spending score",
                "Memiliki Rasio spending dan Income 114%"
            ]}
        predict_values['predict_segment'] = int(segm_kmeans_pcanew[0])
        predict_values['legend'] = legend[segm_kmeans_pcanew[0]]
        predict_values['description'] = description[segm_kmeans_pcanew[0]]
        return predict_values

    def predict(self, gender, age, income, spending_score):
        return self.predict_segment([[gender, age, income, spending_score]])
