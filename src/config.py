from sentence_transformers import SentenceTransformer
import requests
import joblib
import json

MIN_TAKE_CATEGORIZATION = 0
MAX_TAKE_CATEGORIZATION = 100
DEFAULT_MIN_COMMUNITY_SIZE = 3
DEFAULT_SIMILARITY_THRESHOLD = 0.6
DEFAULT_COMMUNITY_DAYS = 3

def download_models(category_table_url: str, classify_model_url: str, language_model: str='distiluse-base-multilingual-cased-v2'):
    if category_table_url is None or classify_model_url is None:
        print('You should provide category table and classify model urls...')
        return None, None, None    
    print("downloading models...")
    embedding_model = SentenceTransformer(language_model)
    ### load classify model
    r = requests.get(classify_model_url) # create HTTP response object 
    with open("classify_model_svm.joblib",'wb') as f: 
        f.write(r.content) 
    classify_model = joblib.load("classify_model_svm.joblib")
    ### load category table
    r = requests.get(category_table_url)
    with open("category_table.json",'wb') as f: 
        f.write(r.content)
    with open("category_table.json", "r") as f:
        category_table = json.load(f)
    return category_table, classify_model, embedding_model