import json
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from dependencies.dependencies import get_mongo_client, get_rag_container
from logs.logger_singleton import Logger

logger = Logger(name="fastapi-routes")

router = APIRouter()


# --- pydantic  ---
class MongoDataResponse(BaseModel):
    response: list[dict]


class FileUploadResponse(BaseModel):
    response: str


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


@router.post("/rag/user_upload", response_model=FileUploadResponse)
async def user_upload(
    container=Depends(get_rag_container),
    file: UploadFile = File(...),
    document: str = Form(...),
):
    """
    Handle file upload and configuration.
    """
    logger.info("Received request to /rag/user_upload")
    try:

        knowledge_base = container.knowledge_base_service()
        knowledge_base_path = knowledge_base.generate(source=file.file)

        document_data = json.loads(document)

        pipeline = container.create_pipeline(
            document_config=document_data, path=knowledge_base_path
        )
        pipeline.process()

        logger.debug(f"Received request json load: {document_data}")
        return FileUploadResponse(
            response="File uploaded successfully. Processing has started and may take some time."
        )
    except Exception as e:
        logger.exception("Error in /rag/user_upload")
        raise HTTPException(
            status_code=500, detail=f"Error fetching response: {str(e)}"
        )
