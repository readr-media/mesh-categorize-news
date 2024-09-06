import os
from src.gql import *
from src.tool import preprocess_text, upload_blob, save_file, remove_punctuation
from src.classifier import ClassifierSingleton
import src.config as config

from datetime import datetime, timedelta
import pytz
import statistics
from sklearn.cluster import DBSCAN

def newpage_clustering(classifier_singleton: ClassifierSingleton):
    error_message = None
    gql_endpoint = os.environ['MESH_GQL_ENDPOINT']
    CLUSTER_EPS = float(os.environ.get('CLUSTER_EPS', config.DEFAULT_CLUSTER_EPS_NEWPAGE))
    MIN_SAMPLES = int(os.environ.get('MIN_SAMPLES', config.DEFAULT_MIN_SAMPLES_NEWPAGE))
    GROUP_DAYS = int(os.environ.get('GROUP_DAYS', config.DEFAULT_GROUP_DAYS))

    current_time = datetime.now(pytz.timezone('Asia/Taipei'))
    start_time = current_time - timedelta(days=GROUP_DAYS)
    formatted_start_time = start_time.isoformat()

    ### get classifier model
    classifier = classifier_singleton.get_instance()
    if classifier is None:
        error_message = "No classifier exists."
        return error_message

    ### get cms stories
    gql_stories_string = gql_query_latest_stories.format(START_PUBLISHED_DATE=formatted_start_time)
    stories, error_message = gql_query(gql_endpoint, gql_stories_string)
    if error_message:
        return str(error_message)
    stories = stories.get('stories', [])
    if len(stories)==0:
        error_message = "Empty stories."
        return error_message

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
            remove_punctuation(story['title']+story['summary'])+preprocess_text(story['content']) for story in story_list
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
            group_list.append(story_list[idx])
        else:
            other_list  = category_group.setdefault('others', [])
            other_list.append(story_list[idx])

    ### save and upload
    for category_name, group_data in groups.items():
        filename = os.path.join('data', f"group_{category_name}.json")
        save_file(filename, group_data)
        upload_blob(filename, cache_control="cache_control_long")
    return error_message

def hotpage_clustering(classifier_singleton: ClassifierSingleton):
    error_message = None
    gql_endpoint = os.environ['MESH_GQL_ENDPOINT']
    CLUSTER_EPS = float(os.environ.get('CLUSTER_EPS_HOTPAGE', config.DEFAULT_CLUSTER_EPS_HOTPAGE))
    MIN_SAMPLES = int(os.environ.get('MIN_SAMPLES_HOTPAGE', config.DEFAULT_MIN_SAMPLES_HOTPAGE))
    HOTPAGE_STORIES_NUM = int(os.environ.get('HOTPAGE_STORIES_NUM', config.DEFAULT_HOTPAGE_STORIES_NUM))

    ### get classifier model
    classifier = classifier_singleton.get_instance()
    if classifier is None:
        error_message = "No classifier exists."
        return error_message

    gql_stories_string = gql_stories_hotpage.format(TAKE=HOTPAGE_STORIES_NUM)
    stories, error_message = gql_query(gql_endpoint, gql_stories_string)
    if error_message:
        return str(error_message)
    stories = stories['stories']

    ### pre-processing stories
    contents = [story['title']+story['og_description'] for story in stories]
    text_embeddings  = classifier.encode(contents)

    ### cluster stories
    clustering = DBSCAN(eps=CLUSTER_EPS, min_samples=MIN_SAMPLES, metric='euclidean').fit(text_embeddings)
    labels = clustering.labels_ # Note: noisy samples will be labelled -1

    ### classify labels
    hotpage_group = {}
    hotpage_no_group = []
    for idx, label in enumerate(labels):
        label = 0 if label==-1 else label # noisy samples is the same as no-group
        if label>0:
            group_list = hotpage_group.setdefault(str(label), [])
            group_list.append(stories[idx])
        else:
            hotpage_no_group.append(stories[idx])
      
    ### post-filtering: remove the abnormal group which have too many stories
    group_lens = [len(group) for _, group in hotpage_group.items()]
    median_len = statistics.median(group_lens)
    threshold_len = 2*median_len
    print('threshold length is: ', threshold_len)
    
    remove_ids = []
    for id, group in hotpage_group.items():
        if len(group) > threshold_len:
            remove_ids.append(id)
            break
    for id in remove_ids:
        print('remove group id: ', id)
        hotpage_group.pop(id)      
          
    ### save and upload
    # upload group
    filename = os.path.join('data', f"hotpage_group.json")
    save_file(filename, hotpage_group)
    upload_blob(filename, cache_control="cache_control_long")
    # upload no group
    filename = os.path.join('data', f"hotpage_no_group.json")
    save_file(filename, hotpage_no_group)
    upload_blob(filename, cache_control="cache_control_long")

    return error_message