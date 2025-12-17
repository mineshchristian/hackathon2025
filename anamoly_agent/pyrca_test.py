# File: rca_analysis.py
import pandas as pd
import re
from datetime import datetime
from pyrca.analyzers.bayesian import BayesianNetwork
from pyrca.graphs.causal.pc import PC


def parse_log_file(file_path: str, service_name: str) -> pd.DataFrame:
    """
    Parses a log file to count ERROR entries per minute.
    
    Args:
        file_path: Path to the log file.
        service_name: Name of the service to use in the column header.
        
    Returns:
        A DataFrame with 'timestamp' and error count columns.
    """
    print(f"Parsing log file: {file_path}...")
    log_pattern = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \[ERROR\].*")
    error_timestamps = []
    with open(file_path, 'r') as f:
        for line in f:
            match = log_pattern.match(line)
            if match:
                # Truncate to the minute
                ts = datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S').replace(second=0, microsecond=0)
                error_timestamps.append(ts)
    
    if not error_timestamps:
        return pd.DataFrame(columns=['timestamp', f'{service_name}_error_count'])

    error_df = pd.DataFrame(error_timestamps, columns=['timestamp'])
    # Group by minute and count occurrences
    error_counts = error_df.groupby('timestamp').size().reset_index(name=f'{service_name}_error_count')
    return error_counts

def run_rca():
    """
    Main function to load data, merge it, and run PyRCA analysis.
    """
    # --- 1. Load Server Metrics ---
    print("Loading server metrics from metrics.csv...")
    metrics_df = pd.read_csv('metrics.csv')
    metrics_df['timestamp'] = pd.to_datetime(metrics_df['timestamp'])
    
    # --- 2. Parse Log Files ---
    service_a_errors_df = parse_log_file('service_a.log', 'service_a')
    service_b_errors_df = parse_log_file('service_b.log', 'service_b')

    # --- 3. Create the Unified DataFrame (KEY TASK) ---
    print("Creating unified DataFrame by merging metrics and log data...")
    
    # Start with the base metrics DataFrame
    df = metrics_df
    
    # Merge error counts from each log file
    if not service_a_errors_df.empty:
        df = pd.merge(df, service_a_errors_df, on='timestamp', how='left')
    if not service_b_errors_df.empty:
        df = pd.merge(df, service_b_errors_df, on='timestamp', how='left')
        
    # Fill NaN values for minutes with no errors with 0
    df.fillna(0, inplace=True)
    
    # Set timestamp as the index, which is good practice for time-series data
    df.set_index('timestamp', inplace=True)
    
    print("\n--- Unified DataFrame Head ---")
    print(df.head())
    print("\n--- Unified DataFrame (Anomaly Period) ---")
    print(df.loc['2023-10-27 10:14:00':'2023-10-27 10:20:00'])
    #df.to_csv('unified_df.csv')

    # --- 4. Perform Root Cause Analysis with PyRCA ---
    print("\nRunning Root Cause Analysis with PyRCA's BayesianNetwork...")
    
    # Initialize the Bayesian Network analyzer
    # The `structure_algorithm` 'chow-liu' is efficient for building the tree structure.
    
    model = PC(PC.config_class())
    graph_df = model.train(df)

    # dot_graph = "digraph { X -> Y; Y -> Z; }"

    # # Create a CausalModel in DoWhy
    # model = CausalModel(
    #     data=data,
    #     treatment="X",
    #     outcome="Z",
    #     graph=dot_graph
    # )

    # # Visualize the model
    # model.view_model()

    bn_analyzer = BayesianNetwork(config=BayesianNetwork.config_class(graph=graph_df))
    
    # Train the model on our historical data
    # PyRCA's BayesianNetwork learns the dependency graph from the data.
    bn_analyzer.train(df)
    
    model.save("model")

    # Define the anomaly we want to investigate.
    # We observed errors in 'service_a', so we set that as the symptom.
    # The anomaly occurred at '2023-10-27 10:17:00'.
    anomaly_timestamp = pd.to_datetime('2023-10-27 10:17:00')
    symptom_nodes = ['service_a_error_count','service_b_error_count']
    
    # Find the root causes for the specified symptom at the given time.
    # The `find_root_causes` method traverses the learned graph to find the most probable causes.
    root_causes = bn_analyzer.find_root_causes(
       symptom_nodes
    )
    
    print("\n--- Root Cause Analysis Results ---")
    if root_causes:
        print(f"Symptom: High '{symptom_nodes[0]}' at {anomaly_timestamp}")
        # The results are sorted by their contribution score, with the highest score being the most likely root cause.
        print(root_causes.to_dict())
     
    else:
        print("No root causes found.")


if __name__ == '__main__':
    # First, ensure sample data exists
    try:
        pd.read_csv('metrics.csv')
        open('service_a.log', 'r')
        open('service_b.log', 'r')
    except FileNotFoundError:
        print("One or more data files not found. Please run generate_sample_data.py first.")
    else:
        run_rca()
