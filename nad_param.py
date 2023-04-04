from sklearn.cluster import DBSCAN
from sklearn.metrics import mean_squared_error, silhouette_score
from sklearn.neighbors import NearestNeighbors
from keras.models import load_model
from keras import backend as K
import tensorflow as tf
import numpy as np
import pandas as pd

import nthresh

def read_csv(path):
    datas = pd.read_csv(path, index_col=False)
    labels = datas['is_anomaly']
    datas = datas.drop('is_anomaly', axis=1)

    np_labels = labels.to_numpy()
    np_datas = datas.to_numpy()

    return np_datas, np_labels

def feed_forward(model, datas):

  n_layer = len(model.layers)
  inp = model.input
  outputs = [ layer.output for layer in model.layers ]
  functors = [ K.function([inp], [out]) for out in outputs ]

  layer_outs = [ func(datas) for func in functors ]
  latent_spaces = np.array(layer_outs[int(n_layer/2)][0])
  latent_spaces = latent_spaces.flatten()
  res = np.array(layer_outs[-1][0])

  REs = [ mean_squared_error(res[i], datas[i]) for i in range(len(datas)) ]
  REs = np.array(REs)

  return latent_spaces, REs

def getDistances(pos_instances):
  neighbors = NearestNeighbors(n_neighbors=3)
  neighbors_fit = neighbors.fit(pos_instances)
  distances, indices = neighbors_fit.kneighbors(pos_instances)

  distances = np.sort(distances, axis=0)
  distances = distances[:,1]
  return distances

def getMinSampleEPS(pos_instances, min_samples, eps):
  output = []

  with tf.device('/GPU:0'):
    for ms in min_samples:
      for ep in eps:
        labels = DBSCAN(min_samples=ms, eps=ep).fit(pos_instances).labels_
        score = silhouette_score(pos_instances, labels)
        output.append((ms, ep, score))
          
    min_samples, eps, score = sorted(output, key=lambda x:x[-1])[-1]
    print(f"Best silhouette_score: {score}")
    print(f"min_samples: {min_samples}")
    print(f"eps: {eps}")

    return min_samples, eps

def get_threshold(REs):
  thresh = nthresh.nthresh(REs, 2)
  print(f"threshold: {thresh[0]}")

  return thresh[0]
  
if __name__ == "__main__":
  main()