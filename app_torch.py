
import tensorflow as tf
 
import transformers as trans
import torch
from transformers import DistilBertTokenizer, BertConfig
from transformers import AdamW, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch.nn as nn
import json
from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import pickle
import sqlite3
import os
import numpy as np



app = Flask(__name__)

# Preparing the Classifier
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)
model_load = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
model_load.load_state_dict(torch.load('torch_weights', map_location='cpu'))
def classify(Message):
    label = {0: 'Ham', 1: 'Spam'}
    X= tokenizer(Message, max_length=200, padding=True, truncation=True, return_tensors="pt") 
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X.to(device)
    model_load.to(device)
    result=model_load(**X)
    result_list=list(result[0][0].cpu().detach().numpy())
    max_value=max(zip(result_list, [0,1]))
    proba=max(nn.Softmax(dim=-1)(result[0][0]).cpu().detach().numpy())
    #proba = np.exp(max_value[0])/(1+np.exp(max_value[0]))
    return label[max_value[1]], proba


# Preparing the SentenceRank
import re
import nltk
import ssl
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.summarization import summarize
from gensim.summarization import keywords
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')

def document_process(text):
    pattern = r'\s|\ufeff'
    remove=re.sub(pattern, ' ', text)
    lmtzr=WordNetLemmatizer()
    text_tokenized=nltk.word_tokenize(text)
    holder=[]
    for i in text_tokenized:
        holder.append(lmtzr.lemmatize(i.lower(), wn.NOUN))
    Lemmatized=' '.join(holder)
    
    return remove, Lemmatized


# Preparing Document Similarity Comparision
import pandas as pd
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity
sbert_model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
def document_prep(doc1, doc2):
    documents_df=pd.DataFrame([doc1, doc2], columns=['documents'])
    stop_words_l=stopwords.words('english')
    doc_list=documents_df.documents.apply(lambda x: " ".join(re.sub(r'[^a-zA-Z]',' ',w).lower() for w in x.split() if re.sub(r'[^a-zA-Z]',' ',w).lower() not in stop_words_l) )
    return doc_list

def sentiment_similarity(doc_list):
    
    document_embeddings = sbert_model.encode(doc_list)
    sentiment_similarities=cosine_similarity(document_embeddings)[0][1]
    return sentiment_similarities


def word_choice_similarity(doc_list):
    tfidfvectoriser=TfidfVectorizer()
    tfidf_vectors=tfidfvectoriser.fit_transform(doc_list)
    words_similarities=cosine_similarity(tfidf_vectors)[0][1]
    return words_similarities





# Flask

@app.route('/', methods=['POST', 'GET'])  # '/'是个URL request, 可以在任何网页上被提出
def index():
    return render_template('Home.html')

# Spam_detect and sub pages


@app.route('/Spam_detect', methods=['POST', 'GET'])
def Spam():
    return render_template('Spam_detect.html')


@app.route('/Spam_detect/Results', methods=['POST', 'GET'])
def Result():
    if request.method=='POST':
        message=request.form['content']
        if len(message)>0:
            #message=[message]
            y, proba = classify([message])
            return render_template('Results.html',
                       content=message,
                       prediction=y,
                       probability=round(proba*100, 2))
    return render_template('Spam_detect.html')


@ app.route('/Document Summary', methods=['POST', 'GET'])
def Summary():
    return render_template('Summary_main.html')



@app.route('/Document Summary/Summary_result', methods=['POST', 'GET'])
def Summary_result():
    if request.method=='POST':
        message=request.form['content']
        rate=request.form['rate'] # form requested info are text, need to con
        count=request.form['count']
        if len(message)>0:
            remove, lemmatized=document_process(message)
            if rate!='' and 0<float(rate)<=100:
                try:
                    summary=summarize(remove, ratio=float(rate)/100)
                    rate=int(float(rate))
                    if len(summary)==0:
                        summary='Unable to generate proper summary based on inputs. \
                        This may due to the fact that the \
                        document being entered for summary is too short \
                        or the value for % of original sentences used for summary was set too low.'
                
                except ValueError:
                    summary='Input must have more than one sentence.'

            else:
                summary='Not Requested'
            if count!='':
                keyword=(keywords(lemmatized, words=int(count)).replace('\n', ', '))
                count=int(count)
            else:
                keyword='Not Requested'
            return render_template('Summary_result.html',
                        summary=summary,
                        keyword=keyword,
                        rate=rate,
                        count=count)
        return render_template('Summary_main.html')


@ app.route('/Document Compare', methods=['POST', 'GET'])
def Compare():
    return render_template('Compare_main.html')

@ app.route('/Document Compare/Compare_result', methods=['POST', 'GET'])
def Compare_result():
    if request.method=='POST':
        doc1=request.form['content1']
        doc2=request.form['content2']
        if len(doc1)>0 and len(doc2)>0:
            doc_list=document_prep(doc1, doc2)
            sentiment_similarities=sentiment_similarity(doc_list)*100
            words_similarities=word_choice_similarity(doc_list)*100
            Feedback="The two entered documents have {}% similarity in terms of 'Content' and {}% similarity \
            in their 'Word Choice'. " \
            .format( round(sentiment_similarities), round(words_similarities))
        elif doc1==' ' and doc2==' ':
            Feedback='The entered documents only contain stop words excluded from analysis'
        else:
            Feedback='Please enter both documents for comparing similarities.'

        return render_template('Compare_result.html', Feedback=Feedback)
    return render_template('Compare_main.html')


@ app.route('/stop_words', methods=['POST', 'GET'])
def Stopwords():
    Stopwords=tuple(stopwords.words('english'))
    return render_template('Nltk_Stopwords.html', stopwords=Stopwords)


if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True)
