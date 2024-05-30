from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import os
import src.config as config
from src.classifier import Classifier

### Download the models
category_table_url = os.environ.get('CATEGORY_TABLE_URL', None)
classify_model_url = os.environ.get('CLASSIFY_MODEL_URL', None)
language_model = os.environ.get('LANGUAGE_MODEL', 'distiluse-base-multilingual-cased-v2')
category_table, classify_model, embedding_model = config.download_models(category_table_url, classify_model_url)
classifier = Classifier(
  classify_model  = classify_model,
  embedding_model = embedding_model,
  category_table  = category_table
)

### App related variables
app = FastAPI()
origins = ["*"]
methods = ["*"]
headers = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = methods,
    allow_headers = headers
)

### API Design
@app.get('/')
async def health_checking():
  '''
  Health checking API.
  '''
  return dict(message="Health check for mesh-categorize-news")

@app.post('/categorize')
async def categorize(data: dict):
  '''
  Categorize the input text.
  '''
  content = data.get("text", None)
  if content is None:
    return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content=dict(error="Input text is required."))
  category_id = classifier.predict([content])[0]
  category_name = classifier.category_mapping(category_id)
  return {'category': category_name}