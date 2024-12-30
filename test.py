import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
print(classifier("I love you"))