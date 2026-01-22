import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import *
import os
import pickle

class NaiveWord2Vec(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.weight1 = tf.keras.layers.Dense(embedding_dim, input_shape=(vocab_size,), activation='linear', name='embedding_layer')
        self.weight2 = tf.keras.layers.Dense(vocab_size, input_shape=(embedding_dim,), activation='softmax')
    
    def call(self, inputs):
        x = self.weight1(inputs)
        x = self.weight2(x)
        return x

class Word2VecModel:
    def __init__(self, window_size=2, vector_dim=250, epochs=10):
        self.df = None
        self.window_size = window_size
        self.vector_dim = vector_dim
        self.epochs = epochs
        self.model = None
        self.unique_words = []
        self.embeddings_lookup = None

    def generate_cbows(self):
        """Generate Continuous Bag of Words (CBOW) pairs from text."""
        text = get_corpus(self.df)
        words = text.split()
        cbows = []
        
        for i, target_word in enumerate(words):
            context_words = words[max(0, i - self.window_size):i] + words[i + 1:i + self.window_size + 1]
            if len(context_words) == self.window_size * 2:
                cbows.append((context_words, target_word))
        
        return cbows
    
    def prepare_cbow(self):
        # Create Cbows non-encoded
        cbows = self.generate_cbows()

        # One-hot-encode words
        self.unique_words = list(get_unique_words(self.df))

        one_hot_encodings = {
            word: self.one_hot_encoding(word) for word in self.unique_words
        }

        # Convert CBOW pairs to vector pairs
        cbow_vector_pairs = [([one_hot_encodings[word] for word in context_words], one_hot_encodings[target_word]) for context_words, target_word in cbows]

        # Sum the context vectors to get a single context vector
        cbow_vector_pairs = [(tf.reduce_sum(tf.stack(context_vectors), axis=0), target_vector) for context_vectors, target_vector in cbow_vector_pairs]

        return cbow_vector_pairs
    
    def one_hot_encoding(self, word: str) -> tf.Tensor:
        if word not in self.unique_words:
            print(f"Word '{word}' not found in unique_words.")
            return None
        
        vector = np.zeros(len(self.unique_words))
        vector[self.unique_words.index(word)] = 1
        return tf.convert_to_tensor(vector)

    def fit(self):
        if os.path.exists('word2vec_lookup.pkl'):
            print('Loading existing Word2Vec embeddings lookup table...')
            self.embeddings_lookup = pickle.load(open('word2vec_lookup.pkl', 'rb'))
            return self
        cbows = self.prepare_cbow()
        vocab_size = len(self.unique_words)
        
        self.model = NaiveWord2Vec(vocab_size, self.vector_dim)
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        print('Training Word2Vec model...')

        self.model.fit(
            x=tf.stack([pair[0] for pair in cbows]),
            y=tf.stack([pair[1] for pair in cbows]),
            epochs=self.epochs,
            batch_size=32,
            validation_split=0.2,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)]
        )
        embeddings = self.model.layers[0].get_weights()[0]
        self.embeddings_lookup = {word: embeddings[idx] for idx, word in enumerate(self.unique_words)}
        # Export lookup table to file for reuse
        with open('word2vec_lookup.pkl', 'wb') as f:
            pickle.dump(self.embeddings_lookup, f)

        return self
    
    def transform(self, X):
        embeddings_lookup = self.embeddings_lookup
        X_embedded = []
        for sms in X:
            words = sms.split()
            sms_embeddings = []
            for word in words:
                if word in embeddings_lookup:
                    sms_embeddings.append(embeddings_lookup[word])
                final_sms_embedding = np.mean(sms_embeddings, axis=0) if sms_embeddings else np.zeros(self.vector_dim)
            X_embedded.append(final_sms_embedding)
        return np.array(X_embedded)

        


class TfIdfModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.df = None

    def fit(self):
        self.vectorizer.fit(self.df.sms)
        return self

    def transform(self, X):
        return self.vectorizer.transform(X).toarray()

    def fit_transform(self, X):
        X = self.vectorizer.fit_transform(X).toarray()
        return X