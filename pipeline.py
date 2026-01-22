from sklearn.metrics import classification_report, confusion_matrix
from preprocessing import TextCleaner
from encoders import Word2VecModel, TfIdfModel
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score

class SpamDetectionPipeline:
    def __init__(self, model_type='xgboost', embedding_type='tfidf'):
        self.cleaner = TextCleaner()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.df = None
        
        if embedding_type == 'word2vec':
            self.embedder = Word2VecModel()
        else:
            self.embedder = TfIdfModel()

        if model_type == 'xgboost':
            self.model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def preprocess(self):
        self.df = self.cleaner.clean(self.df)
        print("Data cleaned.")
        
    def split_data(self, test_size=0.2, random_state=42):
        X = self.df['sms']
        y = self.df['is_spam']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    def vectorize(self):
        self.embedder.df = self.df
        self.embedder.fit()
        self.X_train = self.embedder.transform(self.X_train)
        self.X_test = self.embedder.transform(self.X_test)

    def train_model(self):
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        self.y_pred = self.model.predict(self.X_test)

    def evaluate_model(self):
        print("Confusion Matrix:")
        print(confusion_matrix(self.y_test, self.y_pred))
        print("\nClassification Report:")
        print(classification_report(self.y_test, self.y_pred))
        # Print recall
        recall = recall_score(self.y_test, self.y_pred)
        print(f"Recall: {recall:.4f}")
        return recall
    
if __name__ == "__main__":
    recalls = {}
    for model_type in ['xgboost', 'random_forest']:
        for embedding_type in ['tfidf', 'word2vec']:
            print(f"\nEvaluating model: {model_type} with embedding: {embedding_type}")

            pipeline = SpamDetectionPipeline(model_type=model_type, embedding_type=embedding_type)
            print("Pipeline initialized.")
            # Check if we already have a cleaned dataset
            try:
                cleaned_df = pd.read_csv("dataset/cleaned/cleaned_spam.csv")
                pipeline.df = cleaned_df
                print("Loaded cleaned dataset.")
            except FileNotFoundError:
                print("Cleaned dataset not found. Starting preprocessing...")
                df = pd.read_csv("dataset/raw/spam.csv", encoding='latin-1')[['v1', 'v2']]
                df.columns = ['is_spam', 'sms']
                df['is_spam'] = df['is_spam'].map({'ham': 0, 'spam': 1})
                pipeline.df = df
                pipeline.preprocess()
                pipeline.cleaner.export_to_csv(pipeline.df)

            print("Pipeline initialized.")

            pipeline.split_data()
            print('X shapes: ', pipeline.X_train.shape, pipeline.X_test.shape)

            pipeline.vectorize()
            print('After vectorization: ', pipeline.X_train.shape, pipeline.X_test.shape)

            pipeline.train_model()
            print('Training done. Starting predictions...')

            pipeline.predict()
            print('Predictions done. Evaluating model...')
            
            recall = pipeline.evaluate_model()
            recalls[f"{model_type}_{embedding_type}"] = recall

    print("\nSummary of Recalls (ranking):")
    for key, value in sorted(recalls.items(), key=lambda item: item[1], reverse=True):
        print(f"{key}: {value:.4f}")
    
    