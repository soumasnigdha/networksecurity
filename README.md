# Network Security: Phishing Link Detection

## Overview

This project is a showcase of an end-to-end machine learning pipeline for network security, specifically designed for detecting potential phishing links. It demonstrates the process of data ingestion, validation, transformation, model training, and deployment. While not intended for direct production use, it highlights a modular design and modern ML practices. The trained model is made available for predictions via a FastAPI application. The project emphasizes a modular design, with each component handling a specific stage of the ML workflow. Experiments are tracked using MLflow and DagsHub. The dataset used is the Phishing Websites Dataset, sourced from the UCI Machine Learning Repository: [Phishing Websites Dataset](https://archive.ics.uci.edu/dataset/327/phishing+websites).

A demonstration of the application and its prediction functionality using the Swagger UI can be found here: ![Application Demo](https://github.com/soumasnigdha/networksecurity/blob/main/networksecurity_demo.mp4)

## Project Architecture

The pipeline architecture consists of the following components:

* **Data Ingestion Component**: Extracts data from a MongoDB database.
* **Data Validation Component**: Validates the ingested data to ensure data quality and schema consistency, and detects data drift.
* **Data Transformation Component**: Transforms the validated data by handling missing values and scaling features, preparing it for model training.
* **Model Trainer Component**: Trains a machine learning model using the transformed data, performs hyperparameter tuning, evaluates model performance, and pushes the trained model to a local directory, simulating a staging environment.

###   Pipeline Structure

![Project Structure/Pipeline Image](https://github.com/soumasnigdha/networksecurity/blob/main/project_structure.png)

Each component inherits its configuration and artifacts from the preceding component, except for the Data Ingestion Component, which only inherits its own configuration.

##   Data Flow

1.  **Data Ingestion**: Raw data is extracted from a MongoDB database.
2.  **Data Validation**: The ingested data is validated.
3.  **Data Transformation**: The validated data is transformed.
4.  **Model Trainer**: A model is trained, evaluated, and pushed to a local directory.
5.  **FastAPI**: The trained model is served via a FastAPI application.

##   Data Details

The dataset used for this project contains features designed to identify phishing websites. All features are integers, and there are no missing values. The columns are as follows:

| Feature                        | Description                                                                                             |
| ------------------------------ | ------------------------------------------------------------------------------------------------------- |
| `having_IP_Address`            | Indicates whether the URL has an IP address.                                                            |
| `URL_Length`                   | The length of the URL.                                                                                  |
| `Shortining_Service`           | Indicates whether a URL shortening service is used.                                                     |
| `having_At_Symbol`             | Indicates whether the URL contains an "@" symbol.                                                       |
| `double_slash_redirecting`     | Indicates the presence of "//" in the URL.                                                              |
| `Prefix_Suffix`                | Indicates the presence of a prefix or suffix in the domain name.                                        |
| `having_Sub_Domain`            | Indicates the number of subdomains.                                                                     |
| `SSLfinal_State`               | The state of the SSL certificate.                                                                       |
| `Domain_registeration_length`  | The length of the domain registration.                                                                  |
| `Favicon`                      | Indicates whether the site has a favicon.                                                               |
| `port`                         | The port used for the connection                                                                        |
| `HTTPS_token`                  | Indicates the presence of "HTTPS" token in URL.                                                         |
| `Request_URL`                  | URL of the resource requested                                                                           |
| `URL_of_Anchor`                | URL of the anchor text.                                                                                 |
| `Links_in_tags`                | Links within HTML tags.                                                                                 |
| `SFH`                          | Server Form Handler                                                                                     |
| `Submitting_to_email`          | Indicates whether the form submits data to an email.                                                    |
| `Abnormal_URL`                 | Indicates whether the URL is abnormal.                                                                  |
| `Redirect`                     | Number of redirects.                                                                                    |
| `on_mouseover`                 | Indicates whether there is an onMouseOver event.                                                        |
| `RightClick`                   | Indicates whether right-click is disabled.                                                              |
| `popUpWidnow`                  | Indicates if pop-up window appears.                                                                     |
| `Iframe`                       | Indicates whether the page uses an iframe.                                                              |
| `age_of_domain`                | The age of the domain.                                                                                  |
| `DNSRecord`                    | Indicates whether the DNS record exists.                                                                |
| `web_traffic`                  | The amount of web traffic to the site.                                                                  |
| `Page_Rank`                    | The PageRank of the site.                                                                               |
| `Google_Index`                 | Indicates whether the site is indexed by Google.                                                        |
| `Links_pointing_to_page`       | Number of links pointing to the page.                                                                   |
| `Statistical_report`           | Indicates whether a statistical report is available.                                                    |
| `result`                       | Target feature: Indicates whether the website is phishing (1) or not (-1).                              |

The data was converted from CSV to JSON format and then inserted into a MongoDB database.

## Components

###   1. Data Ingestion

* **Description**: This component extracts data from a MongoDB database, splits it into training and testing sets, and saves them as CSV files.

* **Input**: Data from MongoDB.

* **Output**: Training and testing CSV files.

* **Configuration**:

    * `DATA_INGESTION_COLLECTION_NAME`: Name of the collection in MongoDB ("NetworkData").

    * `DATA_INGESTION_DATABASE_NAME`: Name of the database in MongoDB ("soumasnigdha").

    * `DATA_INGESTION_DIR_NAME`: Name of the data ingestion directory ("data_ingestion").

    * `DATA_INGESTION_FEATURE_STORE_DIR`: Name of the feature store directory ("feature_store").

    * `DATA_INGESTION_INGESTED_DIR`: Name of the ingested data directory ("ingested").

    * `DATA_INGESTION_TEST_TRAIN_SPLIT_RATIO`: Ratio for splitting data into training and testing sets (0.2).

* **Artifacts**:

    * `trained_file_path`: Path to the training CSV file.

    * `test_file_path`: Path to the testing CSV file.

* **Process**:

    1.  Connects to the MongoDB database using the provided URL.

    2.  Retrieves data from the specified collection.

    3.  Splits the data into training and testing sets.

    4.  Saves the training and testing sets as CSV files.

###   2. Data Validation

* **Description**: This component validates the ingested data to ensure its quality and consistency. It checks for schema consistency and detects data drift.

* **Input**: Training and testing CSV files from the Data Ingestion Component.

* **Output**: Validated training and testing CSV files, invalid data files, and a drift report.

* **Configuration**:

    * `DATA_VALIDATION_DIR_NAME`: Name of the data validation directory ("data_validation").

    * `DATA_VALIDATION_VALID_DIR`: Name of the valid data directory ("validated").

    * `DATA_VALIDATION_INVALID_DIR`: Name of the invalid data directory ("invalid").

    * `DATA_VALIDATION_DRIFT_REPORT_DIR`: Name of the drift report directory("drift_report")

    * `DATA_VALIDATION_DRIFT_REPORT_FILE_NAME`: Name of the drift report file ("report.yaml").

* **Artifacts**:

    * `validation_status`: Boolean indicating whether the data is valid.

    * `valid_train_file_path`: Path to the valid training CSV file.

    * `valid_test_file_path`: Path to the valid testing CSV file.

    * `invalid_train_file_path`: Path to the invalid training CSV file.

    * `invalid_test_file_path`: Path to the invalid testing CSV file.

    * `drift_report_file_path`: Path to the drift report file.

* **Process**:

    1.  Loads the training and testing CSV files.

    2.  Validates the number of columns and data types.

    3.  Detects data drift between the training and testing sets.

    4.  Saves the valid and invalid data to separate directories.

    5.  Generates a drift report in YAML format.

###   3. Data Transformation

* **Description**: This component transforms the validated data to make it suitable for model training. It handles missing values using imputation and scales the features.

* **Input**: Valid training and testing CSV files from the Data Validation Component.

* **Output**: Transformed training and testing data in NumPy array format, and a preprocessor object.

* **Configuration**:

    * `DATA_TRANSFORMATION_DIR_NAME`: Name of the data transformation directory ("data_transformation").

    * `DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR`: Name of the transformed data directory ("transformed").

    * `DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR`: Name of the transformed object directory ("transformed_object").

    * `DATA_TRANSFORMATION_IMPUTER_PARAMS`: Parameters for the KNN Imputer used for handling missing values.

* **Artifacts**:

    * `transformed_object_file_path`: Path to the preprocessor object file (in pickle format).

    * `transformed_train_file_path`: Path to the transformed training data (in NumPy array format).

    * `transformed_test_file_path`: Path to the transformed testing data (in NumPy array format).

* **Process**:

    1.  Loads the valid training and testing CSV files.

    2.  Creates a preprocessing pipeline with a KNN Imputer.

    3.  Fits the pipeline on the training data and transforms both the training and testing data.

    4.  Saves the transformed data as NumPy arrays.

    5.  Saves the preprocessor object using pickle.

###   4. Model Trainer

* **Description**: This component trains a machine learning model using the transformed data. It supports multiple models, performs hyperparameter tuning using GridSearchCV, evaluates the trained model, and pushes the model to a local directory.

* **Input**: Transformed training and testing data from the Data Transformation Component.

* **Output**: Trained model, training and testing metrics.

* **Configuration**:

    * `MODEL_TRAINER_DIR_NAME`: Name of the model trainer directory ("model_trainer").

    * `MODEL_TRAINER_TRAINED_MODEL_DIR`: Name of the trained model directory ("trained_model").

    * `MODEL_TRAINER_TRAINED_MODEL_NAME`: Name of the trained model file ("model.pkl").

    * `MODEL_TRAINER_EXPECTED_SCORE`: The expected accuracy of the model (0.6).

    * `MODEL_TRAINER_OVER_FIITTING_UNDER_FITTING_THRESHOLD`: Threshold for overfitting/underfitting (0.05).

* **Artifacts**:

    * `trained_model_file_path`: Path to the trained model file (in pickle format).

    * `train_metric_artifact`: Metrics for the training data (f1_score, precision, recall).

    * `test_metric_artifact`: Metrics for the testing data (f1_score, precision, recall).

* **Process**:

    1.  Loads the transformed training and testing data.

    2.  Defines a set of machine learning models (Random Forest, Decision Tree, Logistic Regression, Gradient Boosting, AdaBoost, and KNeighbors).

    3.  Performs hyperparameter tuning for each model using GridSearchCV.

    4.  Trains and evaluates each model.

    5.  Selects the best-performing model based on the  $R^2$  score.

    6.  Logs the model and metrics using MLflow.

    7.  Saves the trained model and preprocessor.

    8.  Generates the model trainer artifact.

    The best-performing model was KNeighborsClassifier with an  $R^2$  score of 0.838.

###   5. FastAPI Application

* **Description**: This component deploys the trained machine learning model using FastAPI. It provides endpoints for training the model and making predictions.

* **Endpoints**:

    * `/train`: Triggers the training pipeline.

    * `/predict`: Accepts a CSV file, predicts the class labels, and returns the predictions as an HTML table.

* **Functionality**:

    * The `/train` endpoint starts the  `TrainingPipeline`  which executes all the components.

    * The `/predict` endpoint loads the pre-trained model and preprocessor, makes predictions on the uploaded data, and returns the results.

* **Accessing Swagger UI**: The API documentation can be accessed at  `/docs`  or  `/swagger`  when the application is running.

* **Templates**: Uses Jinja2 templates to render HTML.

* **CORS Middleware**: Implements Cross-Origin Resource Sharing (CORS) to allow requests from any origin.

##   Environment Setup

1.  **Python Version**: 3.12

2.  **Dependencies**:

    * Create an Anaconda environment:

        ```
        conda create -n venv python=3.12
        conda activate venv
        ```

    * Install the required packages:

        ```
        pip install -r requirements.txt
        ```

    * `requirements.txt`:

        ```text
        python-dotenv
        pandas
        numpy
        pymongo
        certifi
        pymongo[srv]==3.12
        scikit-learn
        dill
        mlflow
        dagshub
        pyaml
        fastapi
        uvicorn
        python-multipart
        ```

3.  **MongoDB Setup**:

    * Set up a MongoDB database (locally or in the cloud).

    * Ensure the MongoDB URL is set as an environment variable  `MONGO_DB_URL`. A  `.env`  file is used to load this variable.

    * The  `push_data.py`  script can be used to push the initial data to MongoDB.

##   Running the Application

1.  Ensure that MongoDB is running and the  `MONGO_DB_URL`  environment variable is correctly set.

2.  Navigate to the directory containing  `app.py`.

3.  Run the FastAPI application using Uvicorn:

    ```
    uvicorn app:app --reload --host 0.0.0.0 --port 8000
    ```

4.  The API will be accessible at  `http://localhost:8000`.

5.  The Swagger UI will be accessible at  `http://localhost:8000/docs`.

##   Model Deployment

The trained model is saved locally in the  `final_model`  directory. This simulates a staging environment. The  `final_model`  directory contains:

* `preprocessor.pkl`: The preprocessor object used to transform the data.

* `model.pkl`: The trained machine learning model.

##   MLflow and DagsHub Integration

* MLflow is used to track experiments, including parameters, metrics, and artifacts.

* DAGsHub is used as a remote MLflow server to store the tracking data.

* The  `dagshub.init()`  function is used to initialize the connection to the DagsHub repository.

* The training process is logged using  `mlflow.log_param()`,  `mlflow.log_metric()`, and  `mlflow.sklearn.log_model()`.

##   Next Steps (For Further Development)

* Implement Model Evaluation and Model Pusher as separate components.

* Implement cloud deployment (e.g., on AWS).

* Add more comprehensive unit and integration tests.

* Implement data versioning.

* Add support for A/B testing.
