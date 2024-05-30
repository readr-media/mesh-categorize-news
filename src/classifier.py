class Classifier():
    def __init__(self, classify_model, embedding_model, category_table:dict):
        '''
        SentenceTransformer is loaded from Huggingface, and SVC is loaded from bucket.
        '''
        self.embedding_model = embedding_model
        self.classify_model  = classify_model
        self.category_table  = category_table
    def predict(self, sentences: list):
        embeddings  = self.embedding_model.encode(sentences)
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