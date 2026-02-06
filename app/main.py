from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.V1.router import api_router

def create_app() -> FastAPI:
    app = FastAPI(title="Coreference News Bias", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "https://your-frontend-domain.vercel.app",
            "https://cors-test-vercel-news-scrape.vercel.app"
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router)
    return app

app = create_app()
