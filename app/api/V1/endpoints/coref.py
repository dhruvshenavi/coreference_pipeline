from fastapi import APIRouter
from starlette.concurrency import run_in_threadpool
from app.schema.coref import Article
from app.services.coref.coreference import ready_data

router = APIRouter()

@router.post("/coref")
async def coreference(data: Article):
    return await run_in_threadpool(ready_data, data)


