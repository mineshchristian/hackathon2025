from opik.integrations.adk import OpikTracer, track_adk_agent_recursive
from config.settings import MODEL_GEMINI

basic_tracer: OpikTracer = None
basic_tracer = OpikTracer( 
    name="Anamoly_Detection_Agent_Tracer", 
    tags=["basic", "anamoly",  "single-agent"], 
    metadata={ 
        "environment": "development", 
        "model": MODEL_GEMINI, 
        "framework": "google-adk", 
        "example": "basic", 
    }, 
    project_name="hackathonv2"
, 
) 