from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from app.routes import auth, home, depth
from app.config.database import test_connection
import logging

# Set up logging for development
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup - test database connection
    logger.info("Starting FastAPI application...")
    
    # Test MongoDB connection on startup
    connection_ok = await test_connection()
    if connection_ok:
        logger.info("Database connection established")
    else:
        logger.error("Database connection failed - but continuing anyway")
    
    yield
    
    # Shutdown
    logger.info("Shutting down FastAPI application...")

# Create FastAPI app with lifespan events
app = FastAPI(
    title="D3MSD Development API",
    description="API for development environment",
    lifespan=lifespan
)

# Enable CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routes
app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
app.include_router(home.router, prefix="", tags=["Home"])
app.include_router(depth.router, prefix="/depth", tags=["Depth"])

# Serve static files
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

@app.get("/")
def root():
    return {"message": "FastAPI is running in development mode!"}

@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    try:
        from app.config.database import client
        await client.admin.command('ping')
        return {
            "status": "healthy",
            "database": "connected",
            "environment": "development"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "database": f"disconnected: {str(e)}",
            "environment": "development"
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app", 
        host="127.0.0.1", 
        port=8000, 
        reload=True,  # Auto-reload for development
        log_level="info"
    )