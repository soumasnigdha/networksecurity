import os
import sys

from networksecurity.exception.exception import NetworkSecurityException 
from networksecurity.logging.logger import logging

from networksecurity.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
from networksecurity.entity.config_entity import ModelTrainerConfig



from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.main_utils.utils import save_object,load_object
from networksecurity.utils.main_utils.utils import load_numpy_array_data,evaluate_models
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
  RandomForestClassifier,
  GradientBoostingClassifier,
  AdaBoostClassifier,
)
import mlflow
import dagshub


dagshub.init(repo_owner='soumasnigdha', repo_name='networksecurity', mlflow=True)



class ModelTrainer:
  def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
    try:
      self.model_trainer_config = model_trainer_config
      self.data_transformation_artifact = data_transformation_artifact
    except Exception as e:
      raise NetworkSecurityException(e, sys)
    
  def track_mlflow(self, best_model, classification_metric):
    with mlflow.start_run():
      f1_score = classification_metric.f1_score
      recall_score = classification_metric.recall_score
      precision_score = classification_metric.precision_score

      mlflow.log_metric("f1_score", f1_score)
      mlflow.log_metric("recall_score", recall_score)
      mlflow.log_metric("precision", precision_score)
      mlflow.sklearn.log_model(best_model, "model")

    
  def train_model(self, X_train, y_train, X_test, y_test):
    try:
      models = {
        "Random Forest": RandomForestClassifier(verbose=1),
        "Decision Tree": DecisionTreeClassifier(),
        "Logistic Regression": LogisticRegression(verbose=1),
        "Gradient Boosting": GradientBoostingClassifier(verbose=1),
        "AdaBoost": AdaBoostClassifier(),
        "KNeighbors": KNeighborsClassifier(),
      }

      params = {
        "Decision Tree": {
          "criterion": ["gini", "entropy", "log_loss"],
          "max_depth": [2, 4, 6, 8],
          "max_features": ["sqrt", "log2"],
          "splitter": ["best", "random"],
        },
        "Random Forest": {
          "criterion": ["gini", "entropy", "log_loss"],
          "max_depth": [2, 4, 6, 8],
          "max_features": ["sqrt", "log2"],
          "n_estimators": [8, 16, 32, 64, 128, 256],
        },
        "Gradient Boosting": {
          "loss": ["log_loss", "exponential"],
          "learning_rate": [0.001, 0.01, 0.05, 0.1],
          "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
          "criterion": ["friedman_mse", "squared_error"],
          "n_estimators": [8, 16, 32, 64, 128, 256],
        },
        "AdaBoost": {
          "n_estimators": [8, 16, 32, 64, 128, 256],
          "learning_rate": [0.001, 0.01, 0.05, 0.1],
        },
        "KNeighbors": {
          "n_neighbors": [3, 5, 7, 9],
          "weights": ["uniform", "distance"],
          "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
        },
        "Logistic Regression": {
          "penalty": ["l1", "l2", "elasticnet"],
          "C": [0.001, 0.01, 0.1, 1, 10],
          "solver": ["liblinear", "saga"],
        },
      }

      model_report:dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, params=params)
      best_model_score = max(model_report.values())
      best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
      best_model = models[best_model_name]
      logging.info(f"Best model found: {best_model_name} with score: {best_model_score}")

      y_train_pred = best_model.predict(X_train)
      classification_train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)
      self.track_mlflow(best_model, classification_train_metric)

      y_test_pred = best_model.predict(X_test)
      classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)
      self.track_mlflow(best_model, classification_test_metric)

      preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)

      model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
      os.makedirs(model_dir_path, exist_ok=True)

      Network_Model = NetworkModel(preprocessor=preprocessor, model=best_model)
      save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=Network_Model)

      save_object("final_model/model.pkl", best_model)

      ##Model Trainer Artifact
      model_trainer_artifact = ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                           train_metric_artifact=classification_train_metric,
                           test_metric_artifact=classification_test_metric)
      
      logging.info(f"Model trainer artifact: {model_trainer_artifact}")

      return model_trainer_artifact
    
    except Exception as e:
      raise NetworkSecurityException(e, sys)
    
  def initiate_model_trainer(self) -> ModelTrainerArtifact:
    try:
      logging.info("Model training started")
      train_file_path = self.data_transformation_artifact.transformed_train_file_path
      test_file_path = self.data_transformation_artifact.transformed_test_file_path
      # Load transformed train and test data
      train_arr = load_numpy_array_data(train_file_path)
      test_arr = load_numpy_array_data(test_file_path)

      #Split the data into features and target variable
      X_train, y_train, X_test, y_test = (
        train_arr[:,:-1], 
        train_arr[:,-1], 
        test_arr[:,:-1], 
        test_arr[:,-1]
        )
      
      model_trainer_artifact = self.train_model(X_train, y_train, X_test, y_test)
      return model_trainer_artifact

    except Exception as e:
      raise NetworkSecurityException(e, sys)
    

