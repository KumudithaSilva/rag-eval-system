from contextlib import asynccontextmanager
from fastapi import FastAPI
from container.application_container import ApplicationContainer
from container.rag_container import RagEvalContainer
from logs.logger_singleton import Logger
from routes import rag_route
import uvicorn

logger = Logger(name="fastapi")


# --- FastAPI app with lifespan context manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initializes the Mongo client at startup.
    """
    try:
        logger.info("Initialize ApplicationContainer once at startup...")
        app_container = ApplicationContainer()
        rag_container = RagEvalContainer(app_container=app_container)

        app.state.app_container = app_container
        app.state.rag_container = rag_container
        logger.info("Containers initialized successfully.")

        logger.info("Starting Mongo server: initializing Mongo client...")
        mongo_service = rag_container.create_mongo_connection_service()
        app.state.mongo_client = mongo_service.connect()
        logger.info("Mongo client initialized successfully.")

        yield

    except Exception:
        logger.exception("Error during FastAPI startup")
        raise
    finally:
        logger.info("FastAPI server shutting down...")


app = FastAPI(title="Retrival Augmentation Evaluation System API", lifespan=lifespan)
app.include_router(rag_route.router)


# --- Run Uvicorn ---
if __name__ == "__main__":
    uvicorn.run(host="127.0.0.1", port=8000, app="main:app", reload=True)
