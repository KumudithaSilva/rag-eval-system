from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from container.rag_container import RagEvalContainer
from dependencies.dependencies import get_mongo_client
from logs.logger_singleton import Logger

logger = Logger(name="fastapi-routes")

router = APIRouter()
container = RagEvalContainer()


# --- pydantic  ---
class MongoDataResponse(BaseModel):
    response: list[dict]


# --- Routes ---
@router.get("/rag/mongo", response_model=MongoDataResponse)
async def mongo_data(
    mongo_client=Depends(get_mongo_client),
):
    """
    Handle Mongo data return response.
    """
    logger.info("Received request to /rag/mongo")
    try:
        mongo_client_data = mongo_client.fetch_data()
        return MongoDataResponse(response=mongo_client_data)

    except Exception as e:
        logger.exception("Error in /rag/mongo")
        raise HTTPException(
            status_code=500, detail=f"Error fetching response: {str(e)}"
        )
