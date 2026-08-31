from functools import cache
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os

load_dotenv()

@cache
def get_qdrant_client() -> QdrantClient:
    qdrant_host = os.getenv("QDRANT_HOST", default="localhost")
    qdrant_port = os.getenv("QDRANT_PORT", default=6333) 
    return QdrantClient(host=qdrant_host, port=int(qdrant_port))