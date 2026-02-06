from fastapi import APIRouter

router = APIRouter()

@router.get("/")
def home():
    return {"status": "API running"}

@router.get("/health")
def health():
    return {"status": "ok"}

@router.get("/ready")
def ready():
    return {"status": "ready"}
