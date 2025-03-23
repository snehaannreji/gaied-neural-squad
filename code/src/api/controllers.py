from fastapi import APIRouter
from pydantic import BaseModel

from llm import classify_email
from llm.models import all_models

routes = APIRouter()

@routes.get('/')
async def index():
    return 'Welcome to the api'

class EmailText(BaseModel):
    email: str

@routes.post('/api/analyzeEmail')
def analyze_email(email_text: EmailText):
    data = None
    error = None
    
    try:
        data = {name: classify_email(email_text.email, model) for name, model in all_models}
    except Exception as err:
        error = f"An error occurred: {err}"
        print(err)
    
    return {
        "data": data,
        "error": error
    }