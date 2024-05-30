from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from src.classifier import ClassifierSingleton

classifier_singleton = ClassifierSingleton()

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
  classifier = classifier_singleton.get_instance()
  if content is None:
    return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content=dict(error="Input text is required."))
  if classifier is None:
    return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content=dict(error="No classifier exists."))
  category_id = classifier.predict([content])[0]
  category_name = classifier.category_mapping(category_id)
  return {'category': category_name}