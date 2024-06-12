from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from src.classifier import ClassifierSingleton
from src.request_body import CategoryRequestBody
from src.gql import gql_query, gql_query_stories_without_category, gql_query_latest_stories, gql_story_update
from src.tools import preprocess_text
import src.config as config
import os
from sentence_transformers import util
from datetime import datetime, timedelta
import pytz


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
  gql_stories_string = gql_query_stories_without_category.format(take=take)
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
async def cluster():
  '''
  Cronjob to cluster the stories as topic.
  '''
  gql_endpoint = os.environ['MESH_GQL_ENDPOINT']
  MIN_COMMUNITY_SIZE = int(os.environ.get('MIN_COMMUNITY_SIZE', config.DEFAULT_MIN_COMMUNITY_SIZE))
  SIMILARITY_THRESHOLD = float(os.environ.get('SIMILARITY_THRESHOLD', config.DEFAULT_SIMILARITY_THRESHOLD))
  COMMUNITY_DAYS = int(os.environ.get('COMMUNITY_DAYS', config.DEFAULT_COMMUNITY_DAYS))
  
  current_time = datetime.now(pytz.timezone('Asia/Taipei'))
  start_time = current_time - timedelta(days=COMMUNITY_DAYS)
  formatted_start_time = start_time.isoformat()
  
  ### get classifier model
  classifier = classifier_singleton.get_instance()
  if classifier is None:
    return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content=dict(error="No classifier exists."))
  
  ### get cms stories
  gql_stories_string = gql_query_latest_stories.format(START_PUBLISHED_DATE=formatted_start_time)
  stories, error_message = gql_query(gql_endpoint, gql_stories_string)
  if error_message:
    return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content=dict(error="Query stories failed."))
  stories = stories.get('stories', [])
  if len(stories)==0:
    return JSONResponse(status_code=status.HTTP_200_OK, content=dict(error="Empty stories."))
  
  ### categorize
  categorized_stories = {}
  for story in stories:
    category_id = story['category']['id']
    story_list = categorized_stories.setdefault(category_id, [])
    story_list.append(story)
  
  ### cluster
  for category_id, story_list in categorized_stories.items():
    contents = [
      (story['title'] + preprocess_text(story['summary']) + preprocess_text(story['content'])) for story in story_list
    ]
    text_embeddings  = classifier.embedding(contents)
    clusters = util.community_detection(
      text_embeddings, 
      min_community_size=MIN_COMMUNITY_SIZE, 
      threshold=SIMILARITY_THRESHOLD
    )
    print(f"Number of clusters for category_id {category_id}: ", len(clusters))
    for i, cluster in enumerate(clusters):
        print("\nCluster {}, #{} Elements ".format(i + 1, len(cluster)))
        for sentence_id in cluster:
            print("\t", stories[sentence_id]['title'])
    print("--------------------------------")
  return "ok"