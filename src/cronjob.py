import os
from src.gql import *
from src.tool import upload_blob, save_file, embed_stories, get_highlight_group, remove_nonprintable, remove_html
import src.config as config

from datetime import datetime, timedelta
import pytz
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from src.keyword_extract import kw_model

scaler = StandardScaler()

def category_clustering():
    '''
        Cluster the stories based on different category
    '''
    error_message = None
    gql_endpoint = os.environ['MESH_GQL_ENDPOINT']
    EPS_SIMILARITY_DIST = float(os.environ.get('EPS_SIMILARITY_DIST', config.HOTPAGE_CATEGORY_EPS_SIMILARITY))
    MIN_SAMPLES = int(os.environ.get('CATEGORY_MIN_SAMPLES', config.HOTPAGE_CATEGORY_MIN_SAMPLES))
    GROUP_HOURS = config.HOTPAGE_GROUP_HOURS

    current_time = datetime.now(pytz.timezone('Asia/Taipei'))
    start_time = current_time - timedelta(hours=GROUP_HOURS)
    formatted_start_time = start_time.isoformat()

    ### get cms stories
    gql_stories_string = gql_latest_stories.format(START_PUBLISHED_DATE=formatted_start_time)
    stories, error_message = gql_query(gql_endpoint, gql_stories_string)
    if error_message:
        return str(error_message)
    stories = stories.get('stories', [])
    if len(stories)==0:
        error_message = "Empty stories."
        return error_message

    ### embed and cateorize stories
    scaled_embeddings = embed_stories(stories)
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

    ### find hightlight group and organize files
    category_files = {}
    for category_name, category_data in groups.items():
        category_groups_data = category_data.get('groups', {})
        highlight_group_id, _ = get_highlight_group(category_groups_data)
        category_file = category_files.setdefault(category_name, {})
        # topic
        if highlight_group_id != config.NO_HIGHLIGHT_GROUP:
            topic_data = category_data.get('groups', {}).get(highlight_group_id, [])
            if topic_data:
                category_file['group'] = topic_data
        # others
        others_list = category_file.setdefault('others', [])
        for id, group_data in category_groups_data.items():
            if id!=highlight_group_id:
                others_list.extend(group_data)
        others_list.extend(category_data['others'][:config.HOTPAGE_CATEGORY_OTHERS_ADDITION])         
    
    ### save and upload
    for category_name, category_data in category_files.items():
        filename = os.path.join('data', f"group_{category_name}.json")
        save_file(filename, category_data)
        upload_blob(filename, cache_control="cache_control_long")
    return error_message

def all_clustering():
    error_message = None
    gql_endpoint = os.environ['MESH_GQL_ENDPOINT']
    EPS_SIMILARITY_DIST = float(os.environ.get('EPS_SIMILARITY_DIST', config.HOTPAGE_CATEGORY_EPS_SIMILARITY))
    MIN_SAMPLES = int(os.environ.get('HOTPAGE_ALL_MIN_SAMPLES', config.HOTPAGE_ALL_MIN_SAMPLES))
    GROUP_HOURS = config.HOTPAGE_GROUP_HOURS

    current_time = datetime.now(pytz.timezone('Asia/Taipei'))
    start_time = current_time - timedelta(hours=GROUP_HOURS)
    formatted_start_time = start_time.isoformat()

    ### get cms stories
    gql_stories_string = gql_latest_stories.format(START_PUBLISHED_DATE=formatted_start_time)
    stories, error_message = gql_query(gql_endpoint, gql_stories_string)
    if error_message:
        return str(error_message)
    stories = stories['stories']

    ### cluster stories
    scaled_embeddings = embed_stories(stories)
    cos_sim_matrix = cosine_similarity(scaled_embeddings)
    cos_dist_matrix = 1 - np.clip(cos_sim_matrix, 0, 1) # clip negative to 0
    clustering = DBSCAN(eps=EPS_SIMILARITY_DIST, min_samples=MIN_SAMPLES, metric='precomputed').fit(cos_dist_matrix)
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

    # get the highlight group
    highlight_group_id, sorted_score_table = get_highlight_group(hotpage_group)
    if highlight_group_id != config.NO_HIGHLIGHT_GROUP:
        topic_group = hotpage_group[highlight_group_id]
        other_groups = [
            hotpage_group[group_id][0] for group_id, _ in sorted_score_table[1:]
        ]
        if len(other_groups)<config.HOTPAGE_ALL_OTHERS_NUM:
            other_groups.extend(
                hotpage_no_group[:config.HOTPAGE_ALL_OTHERS_NUM-len(other_groups)]
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

def keyword_labelling(story_num: int=config.KEYWORD_LABELLING_NUM, least_ngram: int=2, top_n: int=4):
    gql_endpoint = os.environ['MESH_GQL_ENDPOINT']
    error_msg = None
    try:
        # get data
        data, _ = gql_query(gql_endpoint, gql_story_tags.format(NUM=story_num))
        stories = data['stories']
        content = [
            story['title'] + remove_nonprintable(remove_html(story['summary'])) + remove_nonprintable(remove_html(story['content'])) for story in stories
        ]
        
        # get keywords, which is the array of (phrase, score)
        keywords = kw_model.get_keyword(content)
        mutation_data = []
        all_keywords = []
        for idx in range(len(stories)):
            story_id       = stories[idx]['id']
            story_keywords = keywords[idx]

            # filter keywords which n-gram is less than 2
            filtered_keywords = []
            try:
                for keyword, _ in story_keywords:
                    if len(keyword) >= least_ngram:
                        filtered_keywords.append({
                            "name": keyword
                        })

                mutation_data.append({
                    "where": {
                        "id": story_id
                    },
                    "data": {
                        "tag": {
                            "create": filtered_keywords[:top_n]
                        }
                    }
                })
                all_keywords.extend(filtered_keywords[:top_n])
            except Exception as e:
                print(f"keyword labelled failed for story_id: {story_id}. error:", e)
        
        mutation_create_tags = {
            "data": all_keywords
        }
        mutation_story_tags = {
            "data": mutation_data
        }
        
        # create tags and update story.tags
        _, _ = gql_query(gql_endpoint, gql_create_tags, mutation_create_tags)
        _, error_msg = gql_query(gql_endpoint, gql_update_stories, mutation_story_tags)
        if error_msg:
            raise Exception(error_msg)
    except Exception as e:
        print("cronjob: keyword labelling error: ", e)
    return error_msg