from google.cloud import bigquery
import pandas as pd
import numpy as np
from config.settings import PROJECT_ID


def detect_anomaly_with_bqml_pca(
    project_id: str,
    dataset_id: str,
    model_id: str,
    input_text: str,
    feature_columns: list, # List of column names expected by the PCA model
    threshold: float = None # Anomaly threshold for the score
) -> dict:
    """
    Detects anomalies in input text using a BigQuery ML PCA model.

    Args:
        project_id: Your GCP project ID.
        dataset_id: The BigQuery dataset ID where the PCA model is located.
        model_id: The BigQuery ML PCA model ID.
        input_text: The text string to analyze for anomalies.
        feature_columns: A list of column names that the PCA model expects as input.
                         These must match the features used during model training.
        threshold: An optional anomaly score threshold. If provided, the function
                   will also return a boolean indicating if the text is anomalous.

    Returns:
        A dictionary containing the PCA transformed features and an anomaly score.
        If a threshold is provided, it also includes an 'is_anomalous' boolean.
    """
    client = bigquery.Client(project=project_id)
    model_path = f"{project_id}.{dataset_id}.{model_id}"

    # Step 1: Preprocess the input text to generate numerical features.
    # This is a placeholder. You MUST replace this with the actual
    # text preprocessing and feature extraction logic used when training
    # your BigQuery ML PCA model.
    # Example: TF-IDF vectorization, word embeddings, etc.
    # The output should be a dictionary or list of values corresponding
    # to the 'feature_columns'.
    def text_to_features(text: str, expected_columns: list) -> dict:
        # Placeholder: In a real scenario, this would involve
        # tokenization, vectorization, etc., to produce numerical features.
        # For demonstration, let's assume it produces random features.
        # You need to ensure the number of features and their meaning
        # match your PCA model's training data.
        print(f"Simulating feature extraction for: '{text}'")
        # Example: If your PCA model was trained on 10 features,
        # this function should produce 10 numerical values.
        num_features = len(expected_columns)
        features = np.random.rand(num_features) * 100 # Example random features
        return dict(zip(expected_columns, features))

    features_dict = text_to_features(input_text, feature_columns)

    # Step 2: Create a temporary table or use a direct query for prediction.
    # For a single input, a direct query is often simpler.
    # We construct a SELECT statement that provides the features.
    feature_values_str = ", ".join([
        f"{features_dict[col]} AS {col}" for col in feature_columns
    ])

    # Step 3: Use ML.PREDICT with the BigQuery ML PCA model.
    # For PCA, ML.PREDICT returns the transformed components.
    # Anomaly detection often involves calculating reconstruction error
    # or distance from the principal components. This example assumes
    # we'll use the transformed components directly for a simple score.
    # A more robust anomaly detection would involve calculating
    # reconstruction error or Mahalanobis distance in the PCA space.
    query = f"""
    select uniq_id from ccibt-hack25ww7-752.uc3_dataset.mdl_mem_v2 
    where time_date >= TIMESTAMP_SUB('2025-12-16 00:44:38.923967', INTERVAL 1 MINUTE)

    """

    print(f"Executing BigQuery ML.PREDICT query:\n{query}")
    query_job = client.query(query)
    results = query_job.result()

    # Process results
    transformed_components = []
    for row in results:
        # The output columns will be named 'principal_component_1', 'principal_component_2', etc.
        # or similar, depending on the model's output.
        # We'll collect all numerical values from the row.
        transformed_components = [row[field.name] for field in row.keys() if isinstance(row[field.name], (int, float))]
        break # We expect only one row for a single input

    if not transformed_components:
        raise ValueError("ML.PREDICT did not return any transformed components.")

    # Step 4: Calculate an anomaly score based on the PCA output.
    # This is a placeholder. A common approach for PCA anomaly detection
    # is to calculate the reconstruction error (how well the original data
    # can be reconstructed from the principal components) or the squared
    # distance from the origin in the PCA space.
    # For simplicity, we'll use the squared sum of transformed components as a proxy.
    anomaly_score = np.sum(np.array(transformed_components)**2)

    response = {
        "input_text": input_text,
        "transformed_components": transformed_components,
        "anomaly_score": float(anomaly_score) # Ensure it's a standard float
    }

    if threshold is not None:
        response["is_anomalous"] = anomaly_score > threshold

    return response

# Example Usage (replace with your actual model details and features)
if __name__ == "__main__":
    # These are placeholder values. You need to replace them with your
    # actual project ID, dataset ID, model ID, and the feature columns
    # that your PCA model expects.
    my_model_id = "qry_mdl_PCA_EventTemplate95"
    
    # IMPORTANT: Replace with the actual feature column names your PCA model expects.
    # For text, these would typically be numerical features derived from the text.
    # Example: If your PCA model was trained on 10 TF-IDF features, list them here.
    # For demonstration, assuming generic 'feature_X' columns.
    my_feature_columns = [f"feature_{i}" for i in range(10)]

    test_text_normal = "This is a normal event log entry."
    test_text_anomalous = "ERROR: Critical system failure detected in module XYZ. Aborting all operations."

    # A hypothetical anomaly threshold. This would typically be determined
    # by analyzing the distribution of anomaly scores from normal data.
    my_anomaly_threshold = 500.0

    try:
        print(f"\n--- Analyzing Normal Text ---")
        result_normal = detect_anomaly_with_bqml_pca(
            project_id=PROJECT_ID,
            dataset_id=DATASET_ID,
            model_id=MODEL_MEM_ANAMOLY,
            input_text=test_text_normal,
        )
    except Exception as e:
        print(e)  
