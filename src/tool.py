from bs4 import BeautifulSoup as bs
import regex as re
import string
from google.cloud import storage
import os
import json

def remove_html(content):
    soup = bs(content, 'html.parser')
    for a_tag in soup.find_all('a'):
        a_tag.extract()
    return soup.get_text()

def remove_punctuation(content):
    punctuation_pattern = re.escape(string.punctuation)
    content_filtered = re.sub(f'[{punctuation_pattern}]', '', content)
    return content_filtered

def preprocess_text(content):
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