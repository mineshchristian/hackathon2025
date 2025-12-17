from google.cloud import bigquery
import pandas as pd
import numpy as np
import json
import uuid # To generate unique temporary table names

def detect_anomaly_with_bqml_detect_anomalies(
    project_id: str,
    dataset_id: str,
    model_id: str,
    input_text: list,
    feature_columns: list, # List of column names expected by the PCA model
    contamination: float = 0.1 # Expected proportion of anomalies in the dataset (used by ML.DETECT_ANOMALIES)
) -> dict:
    """
    Detects anomalies in input text using a BigQuery ML PCA model and ML.DETECT_ANOMALIES.

    Args:
        project_id: Your GCP project ID.
        dataset_id: The BigQuery dataset ID where the PCA model is located.
        model_id: The BigQuery ML PCA model ID.
        input_text: The text string to analyze for anomalies.
        feature_columns: A list of column names that the PCA model expects as input.
                         These must match the features used during model training.
        contamination: The expected proportion of anomalies in the dataset.
                       This value is used by ML.DETECT_ANOMALIES to help determine
                       the anomaly threshold.

    Returns:
        A dictionary containing the input text and the anomaly detection result
        (is_anomaly, anomaly_score).
    """
    client = bigquery.Client(project=project_id)
    model_path = f"`{project_id}.{dataset_id}.{model_id}`" # Enclosed in backticks for SQL

    # --- CRITICAL STEP: Replace this with your actual text preprocessing and feature extraction ---
    # The 'text_to_features' function MUST transform your raw input_text
    # into the exact numerical feature format (number and meaning of features)
    # that your BigQuery ML PCA model was trained on.
    # If your model was trained on TF-IDF vectors, you need to use the
    # same TF-IDF vectorizer (with the same vocabulary) here.
    # If it was trained on word embeddings, you need to generate embeddings.
    #
    # Placeholder: This example uses random numbers.
    def text_to_features(text: str, expected_columns: list) -> dict:
        """
        Placeholder function for text preprocessing and feature extraction.
        You MUST replace this with the actual logic used during your
        BigQuery ML PCA model's training.

        Args:
            text: The raw input text string.
            expected_columns: A list of the expected numerical feature column names.

        Returns:
            A dictionary where keys are feature names and values are their numerical
            representations.
        """
        print(f"Simulating feature extraction for: '{text}'...")
        num_features = len(expected_columns)

        # --- IMPORTANT: Replace this with your actual feature generation logic ---
        # For demonstration, let's generate some pseudo-random features.
        # If the text contains "ERROR", make some features higher.
        if "ERROR" in text.upper() or "FAILURE" in text.upper():
            features = np.random.rand(num_features) * 100 + 50 # Higher values for "anomalous" words
        else:
            features = np.random.rand(num_features) * 100 # Lower values for "normal" words

        features = np.clip(features, 0, 200) # Example clipping
        feature_dict = dict(zip(expected_columns, features))
        print(f"Generated features: {feature_dict}")
        return feature_dict
    # --- END OF CRITICAL STEP ---

    #features_dict = text_to_features(input_text, feature_columns)

    # Convert the features dictionary into a Pandas DataFrame for easy upload
    # Ensure the DataFrame has the correct column order and types.
    # input_data_df = pd.DataFrame(dict(zip(feature_columns, input_text)), columns=feature_columns)

    # Generate a unique temporary table name
    # temp_table_id = f"{dataset_id}.temp_anomaly_input_{uuid.uuid4().hex}"
    # print(f"Uploading input data to temporary BigQuery table: {temp_table_id}")

    # job_config = bigquery.LoadJobConfig(
    #     schema=[bigquery.SchemaField(col, "FLOAT") for col in feature_columns],
    #      write_disposition="WRITE_TRUNCATE", # Overwrite if table exists
    # )

    for i, text in enumerate(input_text):
        # Apply the string replacement for each item in the list
        escaped_text = text.replace("'", "\\'")
        input_text[i] = escaped_text
    
    feature_column_str = f"( Select '{input_text[0]}' as {feature_columns[0]}, '{input_text[1]}' as  {feature_columns[1]}  )"

    # Upload the DataFrame to BigQuery
    # load_job = client.load_table_from_dataframe(
    #     input_data_df, temp_table_id, job_config=job_config
    # )
    # load_job.result() # Wait for the load job to complete

    print(f"Data uploaded successfully. Running ML.DETECT_ANOMALIES...")

    # Step 3: Execute ML.DETECT_ANOMALIES query
    # The 'options' struct specifies parameters like 'contamination'
    query = f"""
    SELECT
      *
    FROM
      ML.DETECT_ANOMALIES(
        MODEL {model_path},
        STRUCT({contamination} AS contamination),
        {feature_column_str}
      )
    """

    print(f"\nExecuting BigQuery ML.DETECT_ANOMALIES query:\n{query}")
    query_job = client.query(query)
    results = list(query_job.result())

    # Process results
    anomaly_result = {}
    for row in results:
        anomaly_result = {
            "input_text": input_text,
            "is_anomaly": row["is_anomaly"],
        }
        break # We expect only one row for a single input

    # Clean up the temporary table
    #client.delete_table(temp_table_id, not_found_ok=True)
    #print(f"Temporary table {temp_table_id} deleted.")

    if not anomaly_result:
        raise ValueError("ML.DETECT_ANOMALIES did not return any results.")

    return anomaly_result

# Example Usage
if __name__ == "__main__":
    # Ensure you have authenticated with GCP (e.g., `gcloud auth application-default login`)
    # and have the 'google-cloud-bigquery' library installed (`pip install google-cloud-bigquery pandas numpy`)

    my_project_id = "mineshproject101" # Your actual GCP project ID
    my_dataset_id = "bq_minesh_1"     # Your actual BigQuery dataset ID
    my_model_id = "qry_mdl_PCA_EventTemplate95" # Your actual BigQuery ML PCA model ID

    # IMPORTANT: Replace with the actual feature column names your PCA model expects.
    # For text, these would typically be numerical features derived from the text.
    # Example: If your PCA model was trained on 10 TF-IDF features, list them here.
    my_feature_columns = ['EventTemplate','ParameterList'] # Adjust this based on your model's input

    test_text_normal = ["Failed <*> for invalid user <*> from <*> port <*> ssh2","'password', 'monitor', '103.99.0.122', '59812'"]
    test_text_anomalous = ["CRITICAL ERROR: Unauthorized access attempt detected on database server from unknown IP 203.0.113.42. Immediate action required!","'password', 'monitor', '103.99.0.122', '59812'"]
    test_text_another_normal = ["System health check passed. All services running.","'password', 'monitor', '103.99.0.122', '59812'"]

    # Contamination parameter: This tells ML.DETECT_ANOMALIES what proportion
    # of your *training data* you expect to be anomalous. This helps the function
    # set the internal threshold.
    my_contamination_rate = 0.05 # For example, assume 5% of your historical data was anomalous

    try:
        print(f"\n--- Analyzing Normal Text 1 (Contamination: {my_contamination_rate}) ---")
        result_normal_1 = detect_anomaly_with_bqml_detect_anomalies(
            project_id=my_project_id,
            dataset_id=my_dataset_id,
            model_id=my_model_id,
            input_text=test_text_normal,
            feature_columns=my_feature_columns,
            contamination=my_contamination_rate
        )
        print(f"\nResult for normal text 1:\n{json.dumps(result_normal_1, indent=2)}")

        print(f"\n--- Analyzing Anomalous Text (Contamination: {my_contamination_rate}) ---")
        result_anomalous = detect_anomaly_with_bqml_detect_anomalies(
            project_id=my_project_id,
            dataset_id=my_dataset_id,
            model_id=my_model_id,
            input_text=test_text_anomalous,
            feature_columns=my_feature_columns,
            contamination=my_contamination_rate
        )
        print(f"\nResult for anomalous text:\n{json.dumps(result_anomalous, indent=2)}")

        print(f"\n--- Analyzing Normal Text 2 (Contamination: {my_contamination_rate}) ---")
        result_normal_2 = detect_anomaly_with_bqml_detect_anomalies(
            project_id=my_project_id,
            dataset_id=my_dataset_id,
            model_id=my_model_id,
            input_text=test_text_another_normal,
            feature_columns=my_feature_columns,
            contamination=my_contamination_rate
        )
        print(f"\nResult for normal text 2:\n{json.dumps(result_normal_2, indent=2)}")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please ensure:")
        print("1. Your BigQuery ML model exists and is accessible.")
        print("2. Your project_id, dataset_id, and model_id are correct.")
        print("3. The 'text_to_features' function accurately preprocesses text into the expected numerical features.")
        print("4. You have sufficient BigQuery permissions (bigquery.jobs.create, bigquery.tables.create, bigquery.tables.update, bigquery.tables.delete).")
        print("5. Your `contamination` value is reasonable for your dataset.")

