from bs4 import BeautifulSoup as bs
import regex as re
import string

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