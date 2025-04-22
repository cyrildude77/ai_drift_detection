import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    MODEL_PATH = os.getenv("MODEL_PATH", r"C:\ai_drift_detection\models\lstm_autoencoder.pth")
    THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", 0.004))

settings = Settings()
