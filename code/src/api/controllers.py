from fastapi import APIRouter
from pydantic import BaseModel

from model.basic import classify_email

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
        data = classify_email(email_text.email)
    except Exception as err:
        error = f"An error occurred: {err}"
    
    return {
        "data": data,
        "error": error
    }