# File: generate_sample_data.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("Generating sample data files...")

# --- 1. Generate metrics.csv ---
data = []
base_time = datetime(2023, 10, 27, 10, 0, 0)
for i in range(30):  # 30 minutes of data
    timestamp = base_time + timedelta(minutes=i)
    
    # Normal operations
    service_a_cpu = np.random.uniform(10, 20)
    service_a_memory = np.random.uniform(50, 95)
    service_b_cpu = np.random.uniform(10, 95)
    database_cpu = np.random.uniform(10, 95)
    network_latency = np.random.uniform(50, 80)

    # --- Anomaly Injection between 10:15 and 10:18 ---
    if 15 <= i <= 18:
        # Root Cause: Database CPU spikes
        database_cpu = np.random.uniform(85, 95) 
        # Effect: Service A memory increases due to connection pooling/retries
        service_a_memory = np.random.uniform(70, 85)
        # Effect: Network latency increases
        network_latency = np.random.uniform(200, 300)
        # Effect: Service B CPU also slightly increases
        service_b_cpu = np.random.uniform(40, 50)

    data.append([timestamp, service_a_cpu, service_a_memory, service_b_cpu, database_cpu, network_latency])

metrics_df = pd.DataFrame(data, columns=['timestamp', 'service_a_cpu', 'service_a_memory', 'service_b_cpu', 'database_cpu', 'network_latency'])
metrics_df.to_csv('metrics.csv', index=False)
print("-> Created metrics.csv")


# --- 2. Generate Log Files ---
def generate_log_file(filename, normal_msgs, error_msgs, error_minutes):
    with open(filename, 'w') as f:
        for i in range(30): # 30 minutes
            current_time = base_time + timedelta(minutes=i)
            # Anomaly period
            if i in error_minutes:
                for _ in range(np.random.randint(5, 10)): # More errors during anomaly
                    log_time = current_time + timedelta(seconds=np.random.randint(0, 59))
                    f.write(f"{log_time.strftime('%Y-%m-%d %H:%M:%S')} [ERROR] {np.random.choice(error_msgs)}\n")
            # Normal period
            for _ in range(np.random.randint(1, 3)): # Fewer messages normally
                log_time = current_time + timedelta(seconds=np.random.randint(0, 59))
                f.write(f"{log_time.strftime('%Y-%m-%d %H:%M:%S')} [INFO] {np.random.choice(normal_msgs)}\n")
    print(f"-> Created {filename}")

# Service A logs (directly affected by DB)
generate_log_file(
    'service_a.log',
    normal_msgs=['User authentication successful.', 'Request processed in 50ms.'],
    error_msgs=['Database connection timeout.', 'Failed to execute query.'],
    error_minutes=range(16, 19) # Errors appear a minute after DB CPU spike
)

# Service B logs (indirectly affected)
generate_log_file(
    'service_b.log',
    normal_msgs=['Payment processed.', 'Inventory check complete.'],
    error_msgs=['Upstream service_a unavailable.', 'Request to service_a timed out.'],
    error_minutes=range(17, 20) # Errors appear after service_a starts failing
)

print("Sample data generation complete.")
