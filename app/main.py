from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(
    title="AI-Based Drift Detection System",
    description="Detect concept drift in real-time using LSTM Autoencoders",
    version="1.0"
)

app.include_router(router, prefix="/api")
