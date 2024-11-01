from bs4 import BeautifulSoup as bs
import regex as re
import string
from google.cloud import storage
import os
import json
from datetime import datetime
from src.classifier import classifier_singleton
from sklearn.preprocessing import StandardScaler
import src.config as config
import numpy as np

def remove_html(content):
    soup = bs(content, 'html.parser')
    for a_tag in soup.find_all('a'):
        a_tag.extract()
    return soup.get_text()

def remove_punctuation(content):
    if content==None:
        return ''
    punctuation_pattern = re.escape(string.punctuation)
    content_filtered = re.sub(f'[{punctuation_pattern}]', '', content)
    return content_filtered

def preprocess_text(content):
    if content==None:
        return ''
    return remove_punctuation(remove_html(content))

### cache-control
upload_configs = {
    "real_time": "no-store",
    "cache_control_short": 'max-age=30',
    "cache_control_long": 'max-age=50',
    "cache_control": 'max-age=86400',
    "content_type_json": 'application/json',
}

### upload
def upload_blob(dest_filename, cache_control: str):
    ### with service account attached to the service
    storage_client = storage.Client()
    bucket = storage_client.bucket(os.environ['BUCKET'])
    blob = bucket.blob(dest_filename)
    blob.upload_from_filename(dest_filename)
    
    print("File {} uploaded to {}.".format(dest_filename, dest_filename))
    blob.cache_control = upload_configs[cache_control]
    blob.patch()

### files operations
def save_file(dest_filename, data):
    if data:
        dirname = os.path.dirname(dest_filename)
        if len(dirname)>0 and not os.path.exists(dirname):
            os.makedirs(dirname)
        with open(dest_filename, 'w', encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False))
        print(f'save {dest_filename} successfully')

def open_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        file = json.load(f)
    return file

def df_timestamp(time_str):
    return datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S.%fZ").timestamp()

def embed_stories(stories):
    ### get classifier model
    classifier = classifier_singleton.get_instance()
    if classifier is None:
        return []
    
    ### precalculate embedding and standarization
    scaler = StandardScaler()
    contents = [
      (story['title']*config.DEFAULT_TITLE_WEIGHT+story['og_description']) for story in stories
    ]
    text_embeddings  = classifier.embedding(contents)
    scaled_embeddings = scaler.fit_transform(text_embeddings)
    return scaled_embeddings

def get_highlight_group(groups_data):
    '''
        groups data should be the following format:
        {
            "id": STORIES_LIST [
                STORY1{
                    published_date
                }
                STORY2{
                    published_date
                }
            ]
        }
        return the hightlight id which has the highest score based on timestamp and len(STORIES_LIST)
    '''
    ### ranking: calcuate score of each group and rank
    if not groups_data:
        return config.NO_HIGHLIGHT_GROUP, None
    
    # rank by distinct media number
    distinct_media_count = {}
    for idx, data in groups_data.items():
        sources = [story['source']['id'] for story in data]
        distinct_media_count[idx] = len(set(sources))
    
    sorted_media_number = sorted(
        distinct_media_count.items(), key=lambda item: item[1]
    )
    rank_media_number = {
        group[0]: idx+1 for idx, group in enumerate(sorted_media_number)
    }
    
    # rank by timestamp
    timestamp_table = {}
    for group_id, group in groups_data.items():
        for story in group:
            timestamp_list = timestamp_table.setdefault(group_id, [])
            timestamp_list.append(df_timestamp(story['published_date']))
    for group_id, timestamp_list in timestamp_table.items():
        min_timestamp = np.min(timestamp_list)
        timestamp_table[group_id] = min_timestamp
    
    sorted_timestamp = sorted(
        timestamp_table.items(),
        key=lambda item: item[1],
    )
    rank_timestamp = {
        group[0]: idx+1 for idx, group in enumerate(sorted_timestamp)
    }

    # weight the score
    score_table = {}
    for group_id, score in rank_media_number.items():
        score_table[group_id] = score*rank_timestamp[group_id]
    sorted_score_table = sorted(
        score_table.items(),
        key=lambda item: item[1],
        reverse=True
    )
    # select the highlight group
    highlight_group_id = sorted_score_table[0][0]
    return highlight_group_id, sorted_score_table