from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from src.classifier import ClassifierSingleton
from src.request_body import CategoryRequestBody
from src.gql import gql_query, gql_query_stories_without_category, gql_query_latest_stories, gql_story_update
from tool import preprocess_text, upload_blob, save_file
import src.config as config
import os
from sklearn.cluster import DBSCAN
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
  CLUSTER_EPS = float(os.environ.get('CLUSTER_EPS', config.DEFAULT_CLUSTER_EPS))
  MIN_SAMPLES = int(os.environ.get('MIN_SAMPLES', config.DEFAULT_MIN_SAMPLES))
  GROUP_DAYS = int(os.environ.get('GROUP_DAYS', config.DEFAULT_GROUP_DAYS))
  
  current_time = datetime.now(pytz.timezone('Asia/Taipei'))
  start_time = current_time - timedelta(days=GROUP_DAYS)
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
    category_name = story['category']['slug']
    story_list = categorized_stories.setdefault(category_name, [])
    story_list.append(story)
  
  ### cluster: you should remove noise by restricting the length of text
  groups = {}
  for category_name, story_list in categorized_stories.items():
    contents = [
      preprocess_text(story['title']+story['summary']) for story in story_list
    ]
    text_embeddings  = classifier.embedding(contents)
    clustering = DBSCAN(eps=CLUSTER_EPS, min_samples=MIN_SAMPLES, metric='euclidean').fit(text_embeddings)
    labels = clustering.labels_ # Note: noisy samples will be labelled -1
    
    # categorize group
    category_group = groups.setdefault(category_name, {})
    for idx, label in enumerate(labels):
        label = 0 if label<=0 else label # noisy samples is the same as no-group
        if label>0:
            group_section = category_group.setdefault('groups', {})
            group_list = group_section.setdefault(str(label), [])
            group_list.append(story_list[idx]['title'])
        else:
            other_list  = category_group.setdefault('others', [])
            other_list.append(story_list[idx]['title'])
  
  ### save and upload
  for category_name, group_data in groups.items():
    filename = os.path.join('data', f"group_{category_name}.json")
    save_file(filename, group_data)
    upload_blob(filename, cache_contro="cache_control_long")
  return {"message": "upload groups successfully"}