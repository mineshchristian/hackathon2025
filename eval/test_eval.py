
import pathlib

import dotenv
import pytest
from google.adk.evaluation.agent_evaluator import AgentEvaluator
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from eval import agent

pytest_plugins = ("pytest_asyncio",)


@pytest.fixture(scope="session", autouse=True)
def load_env():
    dotenv.load_dotenv()


@pytest.mark.asyncio
async def test_all():
    """Test the agent's basic ability on a few examples."""

    eval_data_path = str(
        pathlib.Path(__file__).parent / "eval_data/anam.test.json"
    )

    print(eval_data_path)

    await AgentEvaluator.evaluate(
        "eval.agent",
        eval_data_path,
        num_runs=1,
    )
