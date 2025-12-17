from google.cloud import bigquery
import pandas as pd
import numpy as np
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from config.settings import PROJECT_ID,DATASET_ID,MODEL_MEM_ANAMOLY, MODEL_GEMINI,MODEL_CPU_ANAMOLY
from google.adk.agents import LlmAgent

def get_memory_anamolies(
    filter_criteria : str,
   
) -> dict:
    """
    Detects anomalies in input text using a BigQuery ML PCA model.

    Args:
        filter_criteria {str}: filter criteria to get anamolies 

    Returns:
        A dictionary containing the data point is key value pair.
    """
    client = bigquery.Client(project=PROJECT_ID)

    query = f"""
    select  uniq_id,maximum_usage_memory_scaled from {PROJECT_ID}.{DATASET_ID}.{MODEL_CPU_ANAMOLY}
    where {filter_criteria}
    """

    print(f"Executing data anamolie look up :\n{query}")
    query_job = client.query(query)

    transformed_components = []
    transformed_components = [{"uniq_id":row.uniq_id,"maximum_usage_memory": row.maximum_usage_memory_scaled} for row in query_job] # Collect results into a list

    return transformed_components

def get_cpu_anamolies(
    filter_criteria : str,
   
) -> dict:
    """
    Detects anomalies in CPU usage in input text using a BigQuery ML PCA model.

    Args:
        filter_criteria {str}: filter criteria to get anamolies 

    Returns:
        A dictionary containing the data point is key value pair.
    """
    client = bigquery.Client(project=PROJECT_ID)

    query = f"""
    select  uniq_id,maximum_usage_cpu_scaled from {PROJECT_ID}.{DATASET_ID}.{MODEL_MEM_ANAMOLY}
    where {filter_criteria}
    """

    print(f"Executing data anamolie look up :\n{query}")
    query_job = client.query(query)

    transformed_components = []
    transformed_components = [{"uniq_id":row.uniq_id,"maximum_usage_cpu":row.maximum_usage_cpu_scaled} for row in query_job] # Collect results into a list

    return transformed_components


# root_agent = LlmAgent(
#     name="Memory_Anomaly_Detection_Agent",
#     model=MODEL_GEMINI,
#     description="An AI Agent that find anomalies in memory consumption",
#     instruction="""You are an expert anomaly detection system. 
#     Your goal provide a list of anamolies in memory consumption.

#     1.  You have a call get_memory_anamolies with input as a creation of where clause on an BiqQuery sql syntax 
#     2.  User will provide time period to look up the anamolies for.
#     2.  You will use that create where clause on the column time_date
#     3.  Example of where clause for 'Anamolies in memory consumption in last 1 minute '  is time_date >= TIMESTAMP_SUB('2025-12-16 00:44:38.923967', INTERVAL 1 MINUTE). 
#     4.  Use  the fix timestamp '2025-12-16 00:23:38.923967'  
#     5.  If no anamolies found return saying "No anamolies detected"
#     6.  If anamolies found, return back well formatted list of anamolies from a dictionary object
#     """,
#     tools=[get_memory_anamolies,get_cpu_anamolies]
# )

root_agent = LlmAgent(
    name="Anomaly_Detection_Agent",
    model=MODEL_GEMINI,
    description="An AI Agent that find anomalies in memory consumption",
    instruction="""You are an expert anomaly detection system. 
    Your goal provide a list of anamolies in either cpu or memory consumption or both.

    1.  You have a call respective tool based on what user asks. If user specfically ask CPU anamolies then call get_cpu_anamolies. 
    If user asks memory anamolies then call get_memory_anamolies.If ask just anamolies, call both the tools and combile the response.
    2. Call the tools with input as a creation of where clause on an BiqQuery sql syntax 
    3.  User will provide time period to look up the anamolies for.
    4.  You will use that create where clause on the column time_date
    5.  Example of where clause for 'Anamolies in memory consumption in last 1 minute '  is time_date >= TIMESTAMP_SUB('2025-12-16 00:44:38.923967', INTERVAL 1 MINUTE). 
    6.  Use  the fix timestamp '2025-12-16 00:44:38.923967'  
    7.  If no anamolies found return saying "No anamolies detected"
    8.  If anamolies found, return back well formatted list of anamolies from a dictionary object. Show upto 5 anamolies.
    """,
    tools=[get_memory_anamolies, get_cpu_anamolies]
)
