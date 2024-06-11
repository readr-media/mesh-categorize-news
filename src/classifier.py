import os
import src.config as config
import regex as re
import string

class Classifier():
    def __init__(self, classify_model, embedding_model, category_table:dict):
        '''
        SentenceTransformer is loaded from Huggingface, and SVC is loaded from bucket.
        '''
        self.embedding_model = embedding_model
        self.classify_model  = classify_model
        self.category_table  = category_table
    def embedding(self, sentences: list):
        punctuation_pattern = re.escape(string.punctuation)
        sentences_filtered = [re.sub(f'[{punctuation_pattern}]', '', text) for text in sentences]
        embeddings  = self.embedding_model.encode(sentences_filtered)
        return embeddings
    def predict(self, sentences: list):
        embeddings = self.embedding(sentences)
        predictions = self.classify_model.predict(embeddings)
        return predictions
    def category_mapping(self, id):
        '''
        Map the category id(int) to category name(str)
        '''
        return self.category_table[str(id)]
    def validate(self, X_test, y_test):
        predictions = self.classify_model.predict(X_test)
        category_mislabel = {}
        for true_label in y_test:
            category_info = category_mislabel.setdefault(self.category_table[true_label], {})
            category_info['count'] = category_info.get('count', 0) + 1
        for idx, pred in enumerate(predictions):
            true_label = y_test[idx]
            category = self.category_table[true_label]
            if pred!=true_label:
                category_mislabel[category]['miss'] = category_mislabel[category].get('miss', 0) + 1
        return category_mislabel

### Singleton design pattern
class ClassifierSingleton():
    def __init__(self):
        self.classifier = None
    def get_instance(self):
        if self.classifier is None:
            category_table_url = os.environ.get('CATEGORY_TABLE_URL', "https://storage.googleapis.com/statics-mesh-tw-dev/ai-models/category_table.json")
            classify_model_url = os.environ.get('CLASSIFY_MODEL_URL', "https://storage.googleapis.com/statics-mesh-tw-dev/ai-models/classify_model_svm.joblib")
            language_model = os.environ.get('LANGUAGE_MODEL', 'distiluse-base-multilingual-cased-v2')
            category_table, classify_model, embedding_model = config.download_models(category_table_url, classify_model_url, language_model)
            self.classifier = Classifier(
                classify_model  = classify_model,
                embedding_model = embedding_model,
                category_table  = category_table
            )
        return self.classifier
