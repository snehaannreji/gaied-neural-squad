import os

from huggingface_hub import login, InferenceClient

HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACE_TOKEN')

login(HUGGINGFACEHUB_API_TOKEN)

repo_ids = [
    'mistralai/Mistral-7B-Instruct-v0.2',
    # 'google/flan-t5-small'
]

all_models = [
    (id, InferenceClient(id, token=HUGGINGFACEHUB_API_TOKEN)) for id in repo_ids
]