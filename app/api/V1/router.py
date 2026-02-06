from fastapi import APIRouter
from app.api.V1.endpoints import health, coref


api_router = APIRouter(prefix="/api/v1")
api_router.include_router(health.router, tags=["health"])
api_router.include_router(coref.router, tags=['coref'])

