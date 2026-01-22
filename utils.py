import pandas as pd

def get_corpus(df):
    return ' '.join(' '.join(df.sms.tolist()).split())

def count_unique(corpus):
    return len(set(corpus.split()))

def count_unique_in_df(df):
    corpus = get_corpus(df)
    return count_unique(corpus)

def get_unique_words(df):
    corpus = get_corpus(df)
    unique = set()
    output = []
    for word in corpus.split():
        if word not in unique:
            unique.add(word)
            output.append(word)
    return output