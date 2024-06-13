from gql.transport.requests import RequestsHTTPTransport
from gql import gql, Client

def gql_query(gql_endpoint, gql_string: str, gql_variables: str=None, operation_name: str=None):
  '''
      gql_fetch is used to retrieve data
  '''
  json_data, error_message = None, None
  try:
    gql_transport = RequestsHTTPTransport(url=gql_endpoint)
    gql_client = Client(transport=gql_transport,
                        fetch_schema_from_transport=True)
    json_data = gql_client.execute(gql(gql_string), variable_values=gql_variables, operation_name=operation_name)
  except Exception as e:
    print("GQL query error:", e)
    error_message = e
  return json_data, error_message

def gql_story_update(gql_endpoint, stories, model_category_names):    
    ### transform model category names to CMS category ids
    print('Query categories...')
    categories, error_message = gql_query(gql_endpoint, gql_query_categories)
    categories = categories['categories']
    cms_category_mapping = {category['slug']: category['id'] for category in categories}
    
    ### build gql variable
    print('Build gql variable...')
    story_update_args = []
    for idx, story in enumerate(stories):
      story_id = story['id']
      category_name = str(model_category_names[idx]).lower()
      category_id = cms_category_mapping[category_name]
      update_arg = {
        "where": {
          "id": story_id
        },
        "data": {
          "category": {
            "connect": {
              "id": category_id
            }
          }
        }
      }
      story_update_args.append(update_arg)
    story_update_variable = {
      "data": story_update_args
    }
    ### Send update stories query
    print('Send update stories query...')
    json_data, error_message = gql_query(gql_endpoint, gql_update_category, story_update_variable)
    return json_data, error_message

### GQL Queries
gql_query_stories_without_category = """
query Stories{{
    stories(
        where: {{
            category: null
        }},
        orderBy: {{
            id: desc
        }},
        take: {take}
    )
    {{
        id
        title
        summary
        content
    }}
}}
"""

gql_query_latest_stories = '''
query Stories{{
  stories(
    where: {{
      published_date: {{
        gte: "{START_PUBLISHED_DATE}"
      }},
      category: {{
        id: {{
          gt: 0
        }}
      }}
    }},
    orderBy: {{
      published_date: desc
    }},
  ){{
    id
    url
    title
    category{{
      slug
    }}
    source{{
      title
      logo
      official_site
    }}
    published_date
    summary
    content
    pickCount
    commentCount
    og_title
    og_image
    og_description
    full_content
    paywall
  }}
}}
'''

gql_query_categories = """
query Categories{
    categories
    {
        id
        slug
    }
}
"""

gql_update_category = """
mutation UpdateStories($data: [StoryUpdateArgs!]!) {
  updateStories(data: $data) {
    id
    title
    category{
      id
      title
    }
  }
}
"""