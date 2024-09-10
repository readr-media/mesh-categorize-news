import os
from src.gql import *
from src.tool import upload_blob, save_file, df_timestamp
from src.classifier import ClassifierSingleton
import src.config as config

from datetime import datetime, timedelta
import pytz
import statistics
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

scaler = StandardScaler()

def category_clustering(classifier_singleton: ClassifierSingleton):
    '''
        Cluster the stories based on different category
    '''
    error_message = None
    gql_endpoint = os.environ['MESH_GQL_ENDPOINT']
    EPS_SIMILARITY_DIST = float(os.environ.get('EPS_SIMILARITY_DIST', config.HOTPAGE_CATEGORY_EPS_SIMILARITY))
    MIN_SAMPLES = int(os.environ.get('CATEGORY_MIN_SAMPLES', config.HOTPAGE_CATEGORY_MIN_SAMPLES))
    GROUP_DAYS = int(os.environ.get('GROUP_DAYS', config.HOTPAGE_GROUP_DAYS))

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

    ### precalculate embedding and standarization
    contents = [
      (story['title']*2+story['og_description']) for story in stories
    ]
    text_embeddings  = classifier.embedding.encode(contents)
    scaled_embeddings = scaler.fit_transform(text_embeddings)

    ### categorize the stories and embedding
    categorized_stories = {}
    for idx, story in enumerate(stories):
        category_name = story['category']['slug']
        category_info = categorized_stories.setdefault(category_name, {})
        story_list = category_info.setdefault('stories', [])
        story_list.append(story)
        embedding_list = category_info.setdefault('embeddings', [])
        embedding_list.append(scaled_embeddings[idx])

    ### clustering
    groups = {}
    for category_name, category_info in categorized_stories.items():
        story_list = category_info.get('stories', [])
        embedding_list = category_info.get('embeddings', [])

        cos_sim_matrix = cosine_similarity(embedding_list)
        cos_dist_matrix = 1 - np.clip(cos_sim_matrix, 0, 1) # clip negative to 0
        clustering = DBSCAN(eps=EPS_SIMILARITY_DIST, min_samples=MIN_SAMPLES, metric='precomputed').fit(cos_dist_matrix)
        labels = clustering.labels_

        # classify labels
        group_cluster = groups.setdefault(category_name, {})
        other_section = group_cluster.setdefault('others', [])
        for idx, label in enumerate(labels):
            label = 0 if label==-1 else label # noisy samples is the same as no-group
            if label>0:
                group_section = group_cluster.setdefault('groups', {})
                group_list = group_section.setdefault(str(label), [])
                group_list.append(story_list[idx])
            else:
                other_section.append(story_list[idx])

    ### save and upload
    for category_name, group_data in groups.items():
        filename = os.path.join('data', f"group_{category_name}.json")
        save_file(filename, group_data)
        upload_blob(filename, cache_control="cache_control_long")
    return error_message

def all_clustering(classifier_singleton: ClassifierSingleton):
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
    text_embeddings  = classifier.embedding(contents)

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
    threshold_len = config.DEFAULT_CLUSTER_MARGIN*median_len
    print('threshold length is: ', threshold_len)
    
    remove_ids = []
    for id, group in hotpage_group.items():
        if len(group) > threshold_len:
            remove_ids.append(id)
            break
    for id in remove_ids:
        print('remove group id: ', id)
        hotpage_group.pop(id)
    
    ### ranking: calcuate score of each group and rank
    # rank by media number: idx as the rank score, and larger group with higher score
    sorted_media_number = sorted(
        hotpage_group.items(), key=lambda item: len(item[1])
    )
    rank_media_number = {
        group[0]: idx+1 for idx, group in enumerate(sorted_media_number)
    }
    
    # rank by timestamp: idx as the rank score, and newer story with higher score
    timestamp_table = {}
    for group_id, group in hotpage_group.items():
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

    # select the topic
    topic_group_id = sorted_score_table[0][0]
    topic_group = hotpage_group[topic_group_id]
    other_groups = [
        hotpage_group[group_id][0] for group_id, _ in sorted_score_table[1:]
    ]
    if len(other_groups)<6:
        other_groups.extend(
            hotpage_no_group[:6-len(other_groups)]
        )
            
    ### save and upload
    # upload group
    filename = os.path.join('data', f"hotpage_group.json")
    save_file(filename, topic_group)
    upload_blob(filename, cache_control="cache_control_long")
    # upload no group
    filename = os.path.join('data', f"hotpage_no_group.json")
    save_file(filename, other_groups)
    upload_blob(filename, cache_control="cache_control_long")

    return error_message