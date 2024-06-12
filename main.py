from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from src.classifier import ClassifierSingleton
from src.request_body import CategoryRequestBody, ClusterRequestBody
from src.gql import gql_query, gql_query_stories, gql_story_update
from src.tools import preprocess_text
import src.config as config
import os
from sentence_transformers import util


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

@app.post('/test')
async def test(data: dict):
  '''
  Test categorizing the input text.
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

@app.post('/categorize')
async def categorize(data: CategoryRequestBody):
  '''
  Cronjob to classify the stories without category.
  '''
  gql_endpoint = os.environ['MESH_GQL_ENDPOINT']
  take = data.take
  if take<=config.MIN_TAKE_CATEGORIZATION or take>config.MAX_TAKE_CATEGORIZATION:
    return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content=dict(error="Number of take is not valid."))
  
  ### get classifier model
  classifier = classifier_singleton.get_instance()
  if classifier is None:
    return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content=dict(error="No classifier exists."))
  
  ### get cms stories
  gql_stories_string = gql_query_stories.format(take=take)
  stories, error_message = gql_query(gql_endpoint, gql_stories_string)
  if error_message:
    return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content=dict(error="Query stories failed."))
  stories = stories.get('stories', [])
  if len(stories)==0:
    return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content=dict(error="Empty stories."))
  
  ### predict category
  contents = [
    (story['title'] + preprocess_text(story['summary']) + preprocess_text(story['content'])) for story in stories
  ]
  category_ids  = classifier.predict(contents)
  model_category_names = [classifier.category_mapping(id) for id in category_ids]
  
  ### update category
  response, error_message = gql_story_update(gql_endpoint, stories, model_category_names)
  if error_message:
    return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content=dict(error="Update category for stories failed."))
  return response

@app.post('/cluster')
async def cluster(data: ClusterRequestBody):
  '''
  Cronjob to cluster the stories as topic.
  '''
  gql_endpoint = os.environ['MESH_GQL_ENDPOINT']
  take = data.take
  if take<=config.MIN_TAKE_CATEGORIZATION or take>config.MAX_TAKE_CATEGORIZATION:
    return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content=dict(error="Number of take is not valid."))
  
  ### get classifier model
  classifier = classifier_singleton.get_instance()
  if classifier is None:
    return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content=dict(error="No classifier exists."))
  
  ### get cms stories
  gql_stories_string = gql_query_stories.format(take=take)
  stories, error_message = gql_query(gql_endpoint, gql_stories_string)
  if error_message:
    return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content=dict(error="Query stories failed."))
  stories = stories.get('stories', [])
  if len(stories)==0:
    return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content=dict(error="Empty stories."))
  
  ### cluster
  contents = [
    (story['title'] + preprocess_text(story['summary']) + preprocess_text(story['content'])) for story in stories
  ]
  text_embeddings  = classifier.embedding(contents)
  clusters = util.community_detection(text_embeddings, min_community_size=3, threshold=0.6)
  
  # Print for all clusters the top 3 and bottom 3 elements
  print("Number of clusters: ", len(clusters))
  for i, cluster in enumerate(clusters):
      print("\nCluster {}, #{} Elements ".format(i + 1, len(cluster)))
      for sentence_id in cluster[0:3]:
          print("\t", stories[sentence_id]['title'])
      print("\t", "...")
      for sentence_id in cluster[-3:]:
          print("\t", stories[sentence_id]['title'])
  return "ok"