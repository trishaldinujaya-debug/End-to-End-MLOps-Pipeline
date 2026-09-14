\# End-to-End Customer Churn MLOps Pipeline



An end-to-end Machine Learning Operations (MLOps) project for predicting customer churn and deploying the trained machine learning model as a production-ready API.



The project demonstrates the complete ML lifecycle, from data validation and feature engineering to experiment tracking, automated testing, API deployment, containerization, and CI/CD.



\## Project Overview



This project builds a customer churn prediction system using a Random Forest machine learning model and integrates modern MLOps practices.



\### Pipeline



```text

Raw Dataset

&#x20;    ↓

Data Ingestion

&#x20;    ↓

Data Validation

&#x20;    ↓

Feature Engineering

&#x20;    ↓

Model Training

&#x20;    ↓

Model Evaluation

&#x20;    ↓

MLflow Experiment Tracking

&#x20;    ↓

Production Pipeline

&#x20;    ↓

FastAPI Prediction API

&#x20;    ↓

Docker Container

&#x20;    ↓

Automated Testing

&#x20;    ↓

GitHub Actions CI/CD

```



\## Technologies Used



\* Python

\* Pandas

\* NumPy

\* Scikit-learn

\* Random Forest

\* MLflow

\* DVC

\* FastAPI

\* Pydantic

\* Uvicorn

\* Pytest

\* Docker

\* Git

\* GitHub Actions

\* YAML



\## Project Structure



```text

End-to-End-MLOps-Pipeline/

│

├── api/

│   ├── main.py

│   └── schemas.py

│

├── config/

│   └── config.yaml

│

├── data/

│   ├── raw/

│   └── processed/

│

├── models/

│

├── monitoring/

│

├── notebooks/

│

├── pipeline/

│

├── src/

│   ├── data\_ingestion.py

│   ├── data\_validation.py

│   ├── feature\_engineering.py

│   ├── train.py

│   ├── train\_mlflow.py

│   ├── evaluate.py

│   ├── build\_pipeline.py

│   ├── test\_pipeline.py

│   └── utils.py

│

├── tests/

│   └── test\_api.py

│

├── docker/

│

├── kubernetes/

│

├── .github/

│   └── workflows/

│       └── ci.yml

│

├── Dockerfile

├── .dockerignore

├── .gitignore

├── requirements.txt

└── README.md

```



\## Machine Learning Model



The project uses a \*\*Random Forest Classifier\*\* for customer churn prediction.



The production model is packaged together with preprocessing inside a single Scikit-learn pipeline.



The pipeline performs:



1\. Numerical feature scaling

2\. Categorical feature encoding

3\. Random Forest classification

4\. Churn probability prediction



\## Model Evaluation



The model was evaluated using a held-out test dataset.



Current evaluation results:



| Metric    |  Score |

| --------- | -----: |

| Accuracy  | 0.9982 |

| Precision | 0.9992 |

| Recall    | 0.9970 |

| F1 Score  | 0.9981 |

| ROC-AUC   | 1.0000 |



> Note: The unusually high performance will be investigated further as part of future model validation and dataset analysis.



\## MLflow



MLflow is used for experiment tracking.



The project tracks:



\* Model parameters

\* Training metrics

\* Test metrics

\* Model artifacts

\* Experiment runs

\* Model type

\* Dataset information



Experiment:



```text

Customer Churn Prediction

```



\## Data Version Control



DVC is used to version the raw dataset.



```text

data/raw/customer\_churn.csv

```



The dataset itself is not committed directly to Git.



\## FastAPI



The trained model is exposed through a REST API.



\### Health Check



```http

GET /health

```



Example response:



```json

{

&#x20; "status": "healthy",

&#x20; "model\_loaded": true

}

```



\### Prediction



```http

POST /predict

```



Example response:



```json

{

&#x20; "prediction": 0,

&#x20; "churn\_probability": 0.0002

}

```



\### API Documentation



FastAPI automatically provides interactive API documentation through:



```text

/docs

```



\## Docker



The FastAPI application is containerized using Docker.



Build the image:



```bash

docker build -t customer-churn-api:1.1 .

```



Run the container:



```bash

docker run -d -p 8000:8000 --name customer-churn-api-container customer-churn-api:1.1

```



The API can then be accessed at:



```text

http://localhost:8000

```



\## Automated Testing



Pytest is used for automated testing.



Current tests cover:



\* API root endpoint

\* Health endpoint

\* Prediction endpoint

\* Invalid input validation

\* Production pipeline prediction



Run tests:



```bash

python -m pytest -v

```



Current result:



```text

5 passed

```



\## CI/CD



GitHub Actions is configured to automatically:



1\. Check out the repository

2\. Set up Python

3\. Install dependencies

4\. Run the automated test suite



Workflow:



```text

.github/workflows/ci.yml

```



\## Future MLOps Roadmap



The project will continue to be extended with:



\* \[x] Data ingestion

\* \[x] Data validation

\* \[x] Feature engineering

\* \[x] Model training

\* \[x] Model evaluation

\* \[x] MLflow experiment tracking

\* \[x] DVC dataset versioning

\* \[x] FastAPI model serving

\* \[x] Automated API testing

\* \[x] Docker containerization

\* \[x] GitHub Actions CI

\* \[ ] Kubernetes deployment

\* \[ ] Prometheus monitoring

\* \[ ] Grafana dashboards

\* \[ ] Automated model retraining

\* \[ ] Model performance monitoring

\* \[ ] Cloud deployment



\## Learning Objectives



This project was created to gain practical experience with:



\* Machine Learning Engineering

\* MLOps

\* Model deployment

\* REST APIs

\* Docker

\* CI/CD

\* Data versioning

\* Experiment tracking

\* Kubernetes

\* Model monitoring



\## Author



\*\*Dinujaya Fernando\*\*



Bachelor of Computer Science (Honours) in Artificial Intelligence



GitHub: `trishaldinujaya-debug`



\---



This project is continuously being developed as a practical MLOps portfolio project.



