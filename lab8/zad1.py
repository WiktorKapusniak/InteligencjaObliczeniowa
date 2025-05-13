# import libraries
import pandas as pd

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize 
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords



with open('article.txt', 'r') as file:
    text = file.read()
# print(text)

# print(sent_tokenize(text)) 
print(word_tokenize(text))