"""Application settings and helpers.

This module centralizes small defaults and helper functions used across the project.
It intentionally uses simple environment-based configuration so the code works without
extra dependencies.
"""
import os
from typing import Any, Dict
from dotenv import load_dotenv
load_dotenv()

# Streamlit session keys
MESSAGE_HISTORY_KEY = "chat_message_history"
ADK_SESSION_KEY = "adk_session_id"

# ADK / agent defaults
APP_NAME_FOR_ADK = "hackathonv1"
#APP_NAME_FOR_ADK = "agents"
USER_ID = "default_user"
INITIAL_STATE: Dict[str, Any] = {}

# Model setting used by the greeting agent. Update to a real model identifier in production.
MODEL_GEMINI = "gemini-2.5-pro"

PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID")
DATASET_ID = "uc3_dataset"
MODEL_MEM_ANAMOLY = "mdl_mem_v2"
MODEL_CPU_ANAMOLY = "mdl_cpu_v2"

MODEL_MEM_RCA = "mdl_rca_mem_1"
MODEL_CPU_RCA = "mdl_rca_cpu_1"

def get_api_key() -> str | None:
    """Return Google API key from environment (or None).

    This will attempt to read GOOGLE_API_KEY from the environment. If you use a .env file
    loader (python-dotenv or others) in your runtime, this will pick it up from the env.
    """
    return os.environ.get("GOOGLE_API_KEY")
