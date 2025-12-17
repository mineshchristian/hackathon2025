from fastapi import FastAPI
from pydantic import BaseModel
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from adk_service import *
import json
import time
import os

app = FastAPI()

class Query(BaseModel):
    session_id: str
    user_input: str | None = None
    
@app.get("/")
async def root():
    create_session(f"api_adk_session_{int(time.time())}_{os.urandom(4).hex()}")
    return {"message": "Hello World 2"}

@app.get("/test")
async def test():
    create_session(f"api_adk_session_{int(time.time())}_{os.urandom(4).hex()}")
    return {"message": "This is test endpoint"}

@app.get("/createsession")
async def createsession():
    session_id = f"api_adk_session_{int(time.time())}_{os.urandom(4).hex()}"
    await create_session(session_id)
    return {"message": session_id}

@app.post("/query")
async def createsession(query: Query):
    response = await run_adk_async(query.session_id, 
        query.user_input 
    )
    return {"message": response}
