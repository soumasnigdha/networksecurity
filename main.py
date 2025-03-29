from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import DataIngestionConfig, TrainingPipelineConfig, DataValidationConfig

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

    


  except Exception as e:
    raise NetworkSecurityException(e,sys)


