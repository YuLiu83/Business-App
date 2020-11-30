
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
            if rate!='' and 0<float(rate)<=1:
                summary=summarize(remove, ratio=float(rate))
                rate=int(float(rate)*100)
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



if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True)
