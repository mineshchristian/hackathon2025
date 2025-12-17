from typing import List
import numpy as np
import joblib
import asyncio
import logging
from importlib.metadata import version, PackageNotFoundError

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAIEmbeddings

try:
    if tuple(map(int, version("scikit-learn").split('.'))) < (1, 2, 0):
        raise ImportError("scikit-learn version 1.2.0 or higher is required. Please upgrade with 'pip install --upgrade scikit-learn'.")
except PackageNotFoundError:
    raise ImportError("scikit-learn is not installed. Please install it with 'pip install scikit-learn'.")

from sklearn.ensemble import IsolationForest 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from google.cloud import bigquery
from pydantic import BaseModel, Field

PROJECT_ID = "mineshproject101"
BIGQUERY_TABLE = "mineshproject101.bq_minesh_1.bq_tbl_1"

class AnomalyDetectionInput(BaseModel):
    """Input model for the anomaly detection tool."""
    text_to_check: str = Field(
        description="The text string to be checked for anomalies."
    )


def get_or_train_model() -> (IsolationForest, VertexAIEmbeddings):
    """
    Loads a pre-trained IsolationForest model or trains a new one if it doesn't exist.

    Training involves:
    1. Fetching all 'name' data from the BigQuery table.
    2. Generating embeddings for each record using VertexAIEmbeddings.
    3. Training an IsolationForest model on these embeddings.
    4. Saving the trained model to a file for future use.

    Returns:
        A tuple containing the trained model and the embedding instance.
    """
    embeddings = VertexAIEmbeddings(model_name=EMBEDDING_MODEL)
    try:
        logging.info(f"Attempting to load model from {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)
        logging.info("Model loaded successfully.")
        return model, embeddings
    except FileNotFoundError:
        logging.warning(f"Model file not found at {MODEL_PATH}. Training a new model.")
        client = bigquery.Client(project=PROJECT_ID)
        
        # Fetch all data for training
        query = f"SELECT name FROM `{BIGQUERY_TABLE}`"
        logging.info("Fetching all data from BigQuery for training...")
        query_job = client.query(query)
        training_data = [row.name for row in query_job]
        
        if not training_data:
            raise ValueError("No data found in BigQuery to train the model.")

        logging.info(f"Generating embeddings for {len(training_data)} records...")

        try:
            # Generate embeddings for the training data
            training_embeddings = embeddings.embed_documents(training_data)
            # Train the Isolation Forest model
            logging.info("Training IsolationForest model...")
            model = IsolationForest(contamination='auto', random_state=42)
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
               model
            ])
            
            #model.fit(training_embeddings)
            pipeline.fit(training_embeddings)
            
            predictions = pipeline.predict(X_test)
            print(f"Predictions: {predictions}")

            # Extract anomaly scores (lower scores indicate anomalies)
            anomaly_scores = pipeline.decision_function(X_test)
            print(f"Anomaly Scores: {anomaly_scores}")

            # Save the model for future use
            logging.info(f"Saving trained model to {MODEL_PATH}")
            joblib.dump(model, MODEL_PATH)
        except Exception as e:
            logging.error(f"An error occurred during model training: {e}")
            raise e
        
        return model, embeddings

async def detect_anomaly_with_isolation_forest(text_to_check: str, *, context: dict) -> str:
    """
    Uses a pre-trained Isolation Forest model to detect if the input text is an anomaly.
    The model was trained on embeddings of 'normal' data from the BigQuery table.
    Returns 'ANOMALY' if the text is an outlier, otherwise returns 'NORMAL'.
    """
    logging.info(f"Tool 'detect_anomaly_with_isolation_forest' called with text: '{text_to_check}'")
    try:
        model, embeddings = get_or_train_model()
        
        # Generate embedding for the input text
        input_embedding = embeddings.embed_query(text_to_check)
        
        # The model expects a 2D array, so we reshape
        prediction = model.predict(np.array(input_embedding).reshape(1, -1))
        
        # Prediction is -1 for anomalies (outliers) and 1 for normal (inliers)
        result = "ANOMALY" if prediction[0] == -1 else "NORMAL"
        logging.info(f"Anomaly detection result: {result}")
        return f"The analysis result is: {result}"
    except Exception as e:
        logging.error(f"An error occurred during anomaly detection: {e}")
        return f"Error: Could not perform anomaly detection. Details: {e}"
