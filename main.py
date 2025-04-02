from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.components.model_trainer import ModelTrainer
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import DataIngestionConfig, TrainingPipelineConfig, DataValidationConfig, DataTransformationConfig, ModelTrainerConfig

import sys

if __name__=="__main__":
  try:
    trainingpipelineconfig=TrainingPipelineConfig()
    dataingestionconfig=DataIngestionConfig(trainingpipelineconfig)
    dataingestion=DataIngestion(dataingestionconfig)
    logging.info("Initiating the data ingestion")
    dataingestionartifact=dataingestion.initiate_data_ingestion()
    logging.info("Data ingestion completed")
    print(dataingestionartifact)

    data_validation_config=DataValidationConfig(trainingpipelineconfig)
    data_validation=DataValidation(dataingestionartifact, data_validation_config)
    logging.info("Initiating the data validation")
    data_validation_artifact=data_validation.initiate_data_validation()
    logging.info("Data validation completed")
    print(data_validation_artifact)

    data_transformation_config=DataTransformationConfig(trainingpipelineconfig)
    data_transformation=DataTransformation(data_validation_artifact, data_transformation_config)
    logging.info("Initiating the data transformation")
    data_transformation_artifact=data_transformation.initiate_data_transformation()
    logging.info("Data transformation completed")
    print(data_transformation_artifact)

    logging.info("Initiating the model training")
    model_trainer_config = ModelTrainerConfig(trainingpipelineconfig)
    model_train = ModelTrainer(model_trainer_config=model_trainer_config, data_transformation_artifact=data_transformation_artifact)
    model_trainer_artifact = model_train.initiate_model_trainer()
    logging.info("Model training completed")
    print(model_trainer_artifact)

    
  except Exception as e:
    raise NetworkSecurityException(e,sys)


