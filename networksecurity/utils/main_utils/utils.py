import yaml
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
import os, sys
import numpy as np
import dill
import pickle

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score


def read_yaml_file(file_path:str) -> dict:
  try:
    with open(file_path, 'rb') as yaml_file:
      return yaml.safe_load(yaml_file)
  except Exception as e:
    raise NetworkSecurityException(e, sys) from e
  

def write_yaml_file(file_path: str, content: object, replace: bool = False)-> None:
  try:
    if replace:
      if os.path.exists(file_path):
        os.remove(file_path)
      os.makedirs(os.path.dirname(file_path), exist_ok=True)
      with open(file_path, 'w') as yaml_file:
        yaml.dump(content, yaml_file)
  except Exception as e:
    raise NetworkSecurityException(e, sys)
  

def save_numpy_array_data(file_path: str, array: np.array) -> None:
  try:
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)
    with open(file_path, 'wb') as file_obj:
      np.save(file_obj, array)
    logging.info(f"Numpy array saved at: {file_path}")
  except Exception as e:
    raise NetworkSecurityException(e, sys)
  

def load_numpy_array_data(file_path: str) -> np.array:
  try:
    with open(file_path, 'rb') as file_obj:
      logging.info(f"Numpy array loaded from: {file_path}")
      return np.load(file_obj)
  except Exception as e:
    raise NetworkSecurityException(e, sys)

def save_object(file_path: str, obj: object) -> None:
  try:
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)
    with open(file_path, 'wb') as file_obj:
      pickle.dump(obj, file_obj)
    logging.info(f"Object saved at: {file_path}")
  except Exception as e:
    raise NetworkSecurityException(e, sys)
  

def load_object(file_path: str) -> object:
  try:
    if not os.path.exists(file_path):
      raise Exception(f"File not found: {file_path}")
    with open(file_path, 'rb') as file_obj:
      logging.info(f"Object loaded from: {file_path}")
      return pickle.load(file_obj)
  except Exception as e:
    raise NetworkSecurityException(e, sys)
  

def evaluate_models(X_train, y_train, X_test, y_test, models, params) -> dict:
  try:
    report = {}
    logging.info("Evaluating models...")
    for i in range(len(list(models))):
      model = list(models.values())[i]
      param = params[list(models.keys())[i]]

      gs = GridSearchCV(model, param, cv=3, n_jobs=-1)
      gs.fit(X_train, y_train)

      model.set_params(**gs.best_params_)
      model.fit(X_train, y_train)

      y_train_pred = model.predict(X_train)
      y_test_pred = model.predict(X_test)

      train_model_score = r2_score(y_train, y_train_pred)
      test_model_score = r2_score(y_test, y_test_pred)

      report[list(models.keys())[i]] = test_model_score
      logging.info(f"{list(models.keys())[i]}: {test_model_score}")

    return report
  except Exception as e:
    raise NetworkSecurityException(e, sys) from e