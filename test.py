import asyncio
import logging
from typing import List

from google.cloud import bigquery
from google.adk.agents import LlmAgent
from pydantic import BaseModel, Field
from adk_service import *
from config.settings import APP_NAME_FOR_ADK, USER_ID, INITIAL_STATE, ADK_SESSION_KEY



async def createsession():
    session_id = f"api_adk_session_{int(time.time())}_{os.urandom(4).hex()}"
    await create_session(session_id)
    #user_input_anomaly = "error: this is a fatal error"
    user_input_anomaly = "Show me memory anamolies in last 1 minute"
    print(f"\n--- Checking for anomaly in: '{user_input_anomaly}' ---\n")
    response_anomaly = await run_adk_async(session_id,  user_input_anomaly )
    print(f"Agent Response:\n{response_anomaly}\n")

    user_input_anomaly = "Show me CPU anamolies in last 1 minute"
    print(f"\n--- Checking for anomaly in: '{user_input_anomaly}' ---\n")
    response_anomaly = await run_adk_async(session_id,  user_input_anomaly )
    print(f"Agent Response:\n{response_anomaly}\n")

    user_input_anomaly = "Show me anamolies in last 1 minute"
    print(f"\n--- Checking for anomaly in: '{user_input_anomaly}' ---\n")
    response_anomaly = await run_adk_async(session_id,  user_input_anomaly )
    print(f"Agent Response:\n{response_anomaly}\n")


    # #user_input_anomaly = "completely off track"
    # user_input_anomaly = "[error] factory reset failed"
    # print(f"\n--- Checking for anomaly in: '{user_input_anomaly}' ---\n")
    # response_anomaly = await run_adk_async(session_id,  user_input_anomaly )
    # print(f"Agent Response:\n{response_anomaly}\n")





asyncio.run(
createsession()
)

#