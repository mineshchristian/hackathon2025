import asyncio
import logging
from typing import List
import numpy as np
import joblib
from google.cloud import aiplatform
from .forrest_iso import detect_anomaly_with_isolation_forest
from google.cloud import bigquery
from google.cloud import aiplatform, bigquery
from google.adk.agents import LlmAgent
from pydantic import BaseModel, Field
from vertexai.language_models import TextEmbeddingModel
#from .opik import track_adk_agent_recursive

MODEL_GEMINI="gemini-2.5-flash"
#import vertexai
#from vertexai.language_models import TextEmbeddingModel



# Load the textembedding-gecko model
# Use the latest stable version (e.g., textembedding-gecko@003 or @latest)
#model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")

# --- Configuration ---
# Set your Google Cloud project ID here.
# The BigQuery client and other services will use this.
PROJECT_ID = "mineshproject101"

# The full ID of your BigQuery table.
BIGQUERY_TABLE = "mineshproject101.bq_minesh_1.bq_tbl_1"

#vertexai.init(project=PROJECT_ID, location="us-central1")
# The region where your Vertex AI Endpoint is deployed.
LOCATION = "us-central1"

# --- ML Model Configuration ---
MODEL_PATH = "isolation_forest.joblib"
EMBEDDING_MODEL = "text-embedding-005"
# The ID of your deployed model endpoint in Vertex AI Model Registry.
ENDPOINT_ID = "your-vertex-ai-endpoint-id" # <--- IMPORTANT: UPDATE THIS
EMBEDDING_MODEL_NAME = "text-embedding-004"


# Configure logging to see the agent's thoughts.
#logging.basicConfig(level=logging.INFO)


# --- Agent Tool Definition ---

class BigQuerySearchInput(BaseModel):
    """Input model for the BigQuery search tool."""
    search_query: str = Field(
        description="The text string to search for in the BigQuery 'name' column."
    )

class AnomalyDetectionInput(BaseModel):
    """Input model for the anomaly detection tool."""
    text_to_analyze: str = Field(
        description="The input text to be analyzed for anomalies."
    )

async def detect_anomaly_with_vertex_model(text_to_analyze: str, *, context: dict) -> str:
    """
    Analyzes input text for anomalies using a deployed model on Vertex AI.
    It first generates a text embedding, then sends it to the anomaly detection
    endpoint. Returns 'ANOMALY' or 'NORMAL'.
    """
    logging.info(f"Tool 'detect_anomaly_with_vertex_model' called with text: '{text_to_analyze}'")
    try:
        # 1. Initialize Vertex AI
        aiplatform.init(project=PROJECT_ID, location=LOCATION)

        # 2. Generate embedding for the input text
        embedding_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL_NAME)
        embeddings = embedding_model.get_embeddings([text_to_analyze])
        # The model returns a list of embeddings; we need the first one.
        query_embedding = embeddings[0].values

        # 3. Get the Vertex AI Endpoint for the anomaly detection model
        endpoint = aiplatform.Endpoint(endpoint_name=ENDPOINT_ID)

        # 4. Get a prediction from the deployed model
        # The input format must match what your deployed model expects.
        # For scikit-learn models, it's often a list of instances.
        prediction_result = endpoint.predict(instances=[query_embedding])

        # 5. Interpret the result
        # Isolation Forest models typically return -1 for anomalies and 1 for normal.
        prediction = prediction_result.predictions[0]
        return "ANOMALY" if prediction == -1 else "NORMAL"

    except Exception as e:
        logging.error(f"An error occurred during anomaly detection: {e}")
        return f"Error: Could not perform anomaly detection. Details: {e}"


async def search_normal_data_in_bigquery(search_query: str, *, context: dict) -> str:
    """
    Queries a BigQuery table for records where the 'name' column is similar
    to the search_query. This tool finds examples of 'normal' data.
    If it returns an empty list, no similar normal data was found.
    """
    logging.info(f"Tool 'search_normal_data_in_bigquery' called with query: '{search_query}'")
    client = bigquery.Client(project=PROJECT_ID)

    # The user's request mentioned chunking and embedding, which is for semantic
    # search. However, the tool was specified to use a LIKE query.
    # For a direct SQL search, we format the query for a pattern match.
    # Note: Using f-strings for SQL values is unsafe. We use query parameters.
    query0 = f"""
        SELECT name
        FROM `{BIGQUERY_TABLE}`
    """

    query = f"""
            SELECT name
            FROM `{BIGQUERY_TABLE}`
            LIMIT 5
        """
        # Configure the query parameters to prevent SQL injection.
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter(
                "search_pattern", "STRING", f"%{search_query.lower()}%"
            ),
        ]
    )

    try:
        # Execute the query
        #query_job = client.query(query, job_config=job_config)
        query_job = client.query(query0)
        results = [row.name for row in query_job] # Collect results into a list

        logging.info(f"BigQuery query returned {len(results)} results.")

        if not results:
            return "No similar records found in the BigQuery knowledge base."
        else:
            # Format the results as a string for the agent's context.
            formatted_results = "\n- ".join(results)
            return (
                "Found the following similar records in the BigQuery knowledge base:\n"
                f"- {formatted_results}"
            )

    except Exception as e:
        logging.error(f"An error occurred during the BigQuery query: {e}")
        return f"Error: Could not query the BigQuery database. Details: {e}"


# --- Main Application Logic ---

root_agent = LlmAgent(
        name="Anomaly_Detection_Agent",
        model="gemini-2.5-flash",
        description="An AI Agent that detects anomalies in user input using a machine learning model.",
        instruction="""You are an expert anomaly detection system.
        Your goal is to determine if the user's input text is an ANOMALY or NORMAL.

        1.  You have a primary tool called `detect_anomaly_with_isolation_forest`. This tool uses a machine learning model to classify text as 'NORMAL' or 'ANOMALY'.
        2.  ALWAYS use this tool first with the user's input text to get a definitive analysis.
        2.  ALWAYS use the `detect_anomaly_with_vertex_model` tool first with the user's input text to get a definitive analysis.
        3.  You also have a secondary tool, `search_normal_data_in_bigquery`, which can find examples of normal data. Only use this if you need to provide the user with examples of what normal data looks like, but do not use it for the primary analysis.
        4.  Based on the output from the `detect_anomaly_with_isolation_forest` tool, state the result clearly.
        4.  Based on the output from the `detect_anomaly_with_vertex_model` tool, state the result clearly.
        5.  Conclude your response by stating 'FINAL ANSWER: ANOMALY' or 'FINAL ANSWER: NORMAL'.
        6.  Provide a brief explanation for your decision, mentioning that it was based on the machine learning model's analysis.""",
        tools=[detect_anomaly_with_vertex_model, search_normal_data_in_bigquery],

)

#track_adk_agent_recursive(root_agent=root_agent, tracer=basic_tracer) 

   