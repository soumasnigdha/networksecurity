### Network Security Project for Phising Data

This project implements an end-to-end machine learning pipeline for network security, specifically designed for potential phishing email detection, with a focus on deployment.  Data is ingested from a MongoDB database, and the pipeline includes components for data ingestion, validation, and transformation. Data validation ensures schema and column consistency, as well as drift detection.  Data transformation involves handling missing values and feature scaling.  A machine learning model is trained and evaluated, and the project utilizes FastAPI to create an API for model training and prediction. The system is designed for deployment on AWS EC2, leveraging Docker, AWS ECR, and CI/CD pipelines.

![Project Structure](https://imgur.com/a/h6nYJQj)
