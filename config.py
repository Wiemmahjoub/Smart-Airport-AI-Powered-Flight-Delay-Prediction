import os
from dataclasses import dataclass

@dataclass
class AOIPConfig:
    """AOIP Configuration"""
    # Server
    HOST: str = os.getenv('AOIP_HOST', '0.0.0.0')
    PORT: int = int(os.getenv('AOIP_PORT', 8050))
    DEBUG: bool = os.getenv('AOIP_DEBUG', 'False').lower() == 'true'
    
    # Data paths
    DATA_DIR: str = os.getenv('DATA_DIR', 'data/processed')
    MODEL_DIR: str = os.getenv('MODEL_DIR', 'model')
    
    # Model
    MODEL_PATH: str = os.path.join(MODEL_DIR, 'delay_model.pkl')
    
    # Performance
    CACHE_TIMEOUT: int = int(os.getenv('CACHE_TIMEOUT', 300))
    MAX_WORKERS: int = int(os.getenv('MAX_WORKERS', 4))

config = AOIPConfig()