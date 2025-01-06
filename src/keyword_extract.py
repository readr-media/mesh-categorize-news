from ckip_transformers.nlp import CkipWordSegmenter
from keybert import KeyBERT
from sklearn.feature_extraction.text import CountVectorizer
import src.config as config 

class KeywordModel:
    def __init__(self, ckip_ws_base="bert-base", keybert_base: str=config.LANGUAGE_MODEL, top_n: int=8):
        '''
            ckip_ws_base: the base model to do word segmentation
            keybert_base: the base model to do keyword extraction
            top_n: the number of keywords that keybert produces
        '''
        print("KeywordModel: Download ws_driver...")
        self.ws_driver = CkipWordSegmenter(model=ckip_ws_base)
        self.vectorizer = CountVectorizer(tokenizer=lambda text: self.ws_driver([text])[0])
        print("KeywordModel: Download keybert...")
        self.kw_model = KeyBERT(model=keybert_base)
        self.top_n = top_n
    def tokenizer(self, text: str):
        ws = self.ws_driver([text])
        return ws
    def get_keyword(self, docs: list[str]):
        keywords = self.kw_model.extract_keywords(docs,vectorizer=self.vectorizer, top_n=self.top_n)
        return keywords

# singleton
kw_model = KeywordModel()