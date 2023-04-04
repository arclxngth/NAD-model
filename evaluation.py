from keras.models import load_model
from sklearn.cluster import DBSCAN
from sklearn.metrics import mean_squared_error, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from keras import backend as K
import numpy as np
import pandas as pd

ANOMALY_RATIO = 0.8

def read_csv(path):
  datas = pd.read_csv(path, index_col=False)
  labels = datas['is_anomaly']
  datas = datas.drop('is_anomaly', axis=1)

  np_labels = labels.to_numpy()
  np_datas = datas.to_numpy()

  return np_datas, np_labels

def batch_evaluate(inputs, actuals, model_path, params):
  def dbscan_clustering(pos_instances, params):
    def get_clusters(X, y):
      s = np.argsort(y)
      return np.split(X[s], np.unique(y[s], return_index=True)[1][1:])

    eps, min_samples, anomaly_threshold = params
    cluster_labels = DBSCAN(eps=eps, min_samples=min_samples).fit(pos_instances).labels_
    clusters = get_clusters(pos_instances, cluster_labels)
    clusters = clusters[1:]

    isAnomaly = ["anomaly"]

    for cluster in clusters:
      threshExeed = [ 1 if pos[1] >= anomaly_threshold else 0 for pos in cluster ]
      exeedRatio = sum(threshExeed) / len(cluster)
      if exeedRatio >= ANOMALY_RATIO:
          isAnomaly.append("anomaly")
      else:
          isAnomaly.append("normal")

    res = [ isAnomaly[i + 1] for i in cluster_labels ]

    return res

  def ae_predict(model, datas):
    inp = model.input
    outputs = [ layer.output for layer in model.layers ]
    functors = [ K.function([inp], [out]) for out in outputs ]

    layer_outs = [ func(datas) for func in functors ]
    
    return layer_outs

  def eval(predicts, actuals):
    return [
      accuracy_score(y_true=actuals, y_pred=predicts),
      precision_score(y_true=actuals, y_pred=predicts, average="binary", pos_label="anomaly"),
      recall_score(y_true=actuals, y_pred=predicts, average="binary", pos_label="anomaly"),
      f1_score(y_true=actuals, y_pred=predicts, average="binary", pos_label="anomaly"),
      confusion_matrix(y_true=actuals, y_pred=predicts)
    ]

  # Autoencoder
  model = load_model(model_path)
  n_layer = len(model.layers)
  layer_outs = ae_predict(model, inputs)      # each layer outputs
  latent_spaces = np.array(layer_outs[int(n_layer/2)][0])
  latent_spaces = latent_spaces.flatten()     # latent space
  outputs = np.array(layer_outs[-1][0])       # output layer

  # RE
  reconstruct_errors = [ mean_squared_error(outputs[i], inputs[i]) for i in range(len(inputs)) ]
  reconstruct_errors = np.array(reconstruct_errors)

  # combine latent spaces and REs to form pos of each record
  pos_instances = np.dstack((latent_spaces, reconstruct_errors))[0]
  res = dbscan_clustering(pos_instances, params)

  results = eval(res, actuals)
  
  return results

def real_time_evaluate(inputs, actuals, model_path, params):
  def ae_predict(model, datas):
    inp = model.input
    outputs = [ layer.output for layer in model.layers ]
    functors = [ K.function([inp], [out]) for out in outputs ]

    layer_outs = [ func(datas) for func in functors ]
    
    return layer_outs

  def eval(predicts, actuals):
    return [
      accuracy_score(y_true=actuals, y_pred=predicts),
      precision_score(y_true=actuals, y_pred=predicts, average="binary", pos_label="anomaly"),
      recall_score(y_true=actuals, y_pred=predicts, average="binary", pos_label="anomaly"),
      f1_score(y_true=actuals, y_pred=predicts, average="binary", pos_label="anomaly"),
      confusion_matrix(y_true=actuals, y_pred=predicts)
    ]
  
  def RT_detect(REs, threshold):
    return [ "anomaly" if e > threshold else "normal" for e in REs ]

  # Autoencoder
  model = load_model(model_path)
  layer_outs = ae_predict(model, inputs)      # each layer outputs
  outputs = np.array(layer_outs[-1][0])       # output layer

  # RE
  reconstruct_errors = [ mean_squared_error(outputs[i], inputs[i]) for i in range(len(inputs)) ]
  reconstruct_errors = np.array(reconstruct_errors)

  res = RT_detect(reconstruct_errors, params)

  results = eval(res, actuals)
  return results

def main():
  return 0

if __name__ == "__main__":
  main()