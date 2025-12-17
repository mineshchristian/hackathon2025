import uvicorn
from fastapi import FastAPI

# Create the FastAPI application instance
app = FastAPI()

# Define a simple route
@app.get("/")
def read_root():
    return {"Hello": "World111"}