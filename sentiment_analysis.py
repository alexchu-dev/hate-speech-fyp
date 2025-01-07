import torch
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
if __name__ == '__main__':
    df = pd.read_csv('datasets\labeled_data.csv')
    df = df.drop('class',axis=1)
    tokenizer = BertTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    model = BertForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    def sentiment_score(movie_review):
        token = tokenizer.encode(movie_review, return_tensors = 'pt')
        result = model(token)
        return int(torch.argmax(result.logits))+1
    df['class'] = df['tweet'].apply(lambda x: sentiment_score(x[:512]))
    print(df)
    print(sentiment_score("I love you"))
    print(sentiment_score("I fuck you"))
    print(sentiment_score("facewithtearsofjoy middlefinger you"))
