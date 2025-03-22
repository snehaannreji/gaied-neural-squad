from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from api.controllers import routes

app = FastAPI(
    title="Demo API"
)
app.include_router(routes)
