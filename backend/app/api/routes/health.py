from fastapi import APIRouter

router = APIRouter(prefix="/api", tags=["health"])

@router.get("/health")
def health_check():
    return {
        "status": "healthy",
        "services": {
            "sqlite": "healthy",
            "filesystem": "healthy"
        }
    }

@router.get("/version")
def version():
    return {
        "apiVersion": "0.1.0",
        "detectorContractVersion": "1.0.0"
    }
