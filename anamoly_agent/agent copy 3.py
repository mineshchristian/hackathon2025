from google.cloud import bigquery
import pandas as pd
import numpy as np
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from config.settings import PROJECT_ID,DATASET_ID,MODEL_MEM_ANAMOLY, MODEL_GEMINI,MODEL_CPU_ANAMOLY
from google.adk.agents import LlmAgent

def get_memory_anomalies(
    filter_criteria : str,
   
) -> dict:
    """
    Detects anomalies in input text using a BigQuery ML PCA model.

    Args:
        filter_criteria {str}: filter criteria to get anomalies 

    Returns:
        A dictionary containing the data point is key value pair.
    """
    client = bigquery.Client(project=PROJECT_ID)

    query = f"""
    select  uniq_id,maximum_usage_memory_scaled from {PROJECT_ID}.{DATASET_ID}.{MODEL_CPU_ANAMOLY}
    where {filter_criteria}
    """

    print(f"Executing data anomalies look up :\n{query}")
    query_job = client.query(query)

    transformed_components = []
    transformed_components = [{"uniq_id":row.uniq_id,"maximum_usage_memory": row.maximum_usage_memory_scaled} for row in query_job] # Collect results into a list

    return transformed_components

def get_cpu_anomalies(
    filter_criteria : str,
   
) -> dict:
    """
    Detects anomalies in CPU usage in input text using a BigQuery ML PCA model.

    Args:
        filter_criteria {str}: filter criteria to get anomalies 

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
#     Your goal provide a list of anomalies in memory consumption.

#     1.  You have a call get_memory_anomalies with input as a creation of where clause on an BiqQuery sql syntax 
#     2.  User will provide time period to look up the anomalies for.
#     2.  You will use that create where clause on the column time_date
#     3.  Example of where clause for 'anomalies in memory consumption in last 1 minute '  is time_date >= TIMESTAMP_SUB('2025-12-16 00:44:38.923967', INTERVAL 1 MINUTE). 
#     4.  Use  the fix timestamp '2025-12-16 00:23:38.923967'  
#     5.  If no anomalies found return saying "No anomalies detected"
#     6.  If anomalies found, return back well formatted list of anomalies from a dictionary object
#     """,
#     tools=[get_memory_anomalies,get_cpu_anomalies]
# )

root_agent = LlmAgent(
    name="Anomaly_Detection_Agent",
    model=MODEL_GEMINI,
    description="An AI Agent that find anomalies in memory consumption",
    instruction="""
    Your goal is to provide anomalies in  CPU and memory consumption.

    1.  You have to call respective tool based on  which type of anomalies user asks. CALL the tools only ONCE.  Show both type of anomalies if user does not specify.
    2.  Call the tool with by providing WHERE CLAUSE for an BiqQuery sql syntax based the user input for the time period requested.
    3.  User will provide time period to look up the anomalies for. 
    4.  The WHERE CLAUSE needs to be on the column time_date.
    5.  Example of WHERE CLAUSE for 'anomalies in memory consumption in last 1 minute'  is time_date >= TIMESTAMP_SUB('2025-12-16 00:44:38.923967', INTERVAL 1 MINUTE). 
    6.  Use  the fix timestamp '2025-12-16 00:44:38.923967'  
    7.  If no anomalies found return saying "No anomalies detected"
    8.  If anomalies found, return back well formatted list of anomalies from a dictionary object. 

    AGAIN, ***VERY IMPORTANT***  call the tools only ONCE.
    """,
    tools=[get_memory_anomalies, get_cpu_anomalies]

)

# root_agent = LlmAgent(
#     name="Anomaly_Detection_Agent",
#     model=MODEL_GEMINI,
#     description="An AI Agent that find anomalies in memory consumption",
#     instruction="""
#     Your goal is to provide anomalies in  CPU and  memory consumption.

#     1.  You have to call respective tool based on what which type of anomalies user asks. CALL the tools only ONCE.  
#     2.  If user specfically ask CPU anomalies then call get_cpu_anomalies. If user asks memory anomalies then call get_memory_anomalies.If asks just anomalies, call both the tools and combile the response.
#     3.  Call the tools with input as a creation of where clause on an BiqQuery sql syntax 
#     4.  User will provide time period to look up the anomalies for.
#     5.  You will use that create where clause on the column time_date
#     6.  Example of where clause for 'anomalies in memory consumption in last 1 minute '  is time_date >= TIMESTAMP_SUB('2025-12-16 00:44:38.923967', INTERVAL 1 MINUTE). 
#     7.  Use  the fix timestamp '2025-12-16 00:44:38.923967'  
#     8.  If no anomalies found return saying "No anomalies detected"
#     9.  If anomalies found, return back well formatted list of anomalies from a dictionary object. Show upto 5 anomalies.

#     AGAIN, ***VERY IMPORTANT***  call the tools only ONCE.
#     """,
#     tools=[get_memory_anomalies, get_cpu_anomalies]

# )
