### Spam Filtering Network Security ML Project

This project implements an end-to-end machine learning pipeline for network security, specifically designed for potential phishing email detection, with a focus on deployment via a FastAPI API. The original raw data is first pushed to a MongoDB database and then ingested from that database. The pipeline's architecture is structured around distinct components, each with its configuration and artifacts:
 * Data Ingestion Component extracts data from the MongoDB database and produces Data Ingestion Artifacts.
 * Data Validation Component ensures data quality by validating the ingested data, generating Data Validation Artifacts, including reports on schema consistency and data drift.
 * Data Transformation Component transforms the validated data, handling missing values and feature scaling using a preprocessor, and creating Data Transformation Artifacts, including the preprocessor itself.
 * Model Trainer Component trains the machine learning model using the transformed data and also includes model evaluation and model pushing functionalities, producing Model Trainer Artifacts, which include the trained model, evaluation metrics, and deployment artifacts.
 * The project exposes this pipeline's functionality through a FastAPI application. The API provides endpoints for:
    - /train: Triggers the execution of the training pipeline to train, evaluate, and prepare the model for deployment.
    - /predict: Accepts a CSV file as input, uses the preprocessor generated during the Data Transformation stage and the trained model to predict phishing probabilities, and returns the results.


![Project Structure](https://github.com/soumasnigdha/networksecurity/blob/main/project_structure.png)
