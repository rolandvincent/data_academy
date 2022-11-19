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
        legend = {0: 'Spender-Agressive 2',  # 'highly-spender-woman-hgi',
                  1: 'Thrifty-Spender',  # 'thrifty-spender-man',
                  2: 'Spender-Agressive 1',  # 'highly-spender-woman-lwi',
                  3: 'Thrifty-Spender',  # 'thrifty-spender-woman',
                  4: 'Spender-Agressive 1'}  # 'highly-spender-man'}
        predict_values['predict_segment'] = int(segm_kmeans_pcanew[0])
        predict_values['legend'] = legend[segm_kmeans_pcanew[0]]
        return predict_values

    def predict(self, gender, age, income, spending_score):
        return self.predict_segment([[gender, age, income, spending_score]])
