from tensorflow.keras.models import Model
from official.nlp.bert import configs, tokenization, bert_models
from official.nlp import bert
from official import nlp
import tensorflow as tf
import json
from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import pickle
import sqlite3
import os
import numpy as np
import plotly


app = Flask(__name__)

# Preparing the Classifier
bert_config_file = os.path.join('bert_model', "bert_config.json")
config_dict = json.loads(tf.io.gfile.GFile(bert_config_file).read())
bert_config = bert.configs.BertConfig.from_dict(config_dict)
tokenizer = bert.tokenization.FullTokenizer(
    vocab_file=os.path.join('bert_model', "vocab.txt"),
    do_lower_case=True)
tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]'])
def encode_sentence(s):
    tokens = list(tokenizer.tokenize(s))
    tokens.append('[SEP]')
    return tokenizer.convert_tokens_to_ids(tokens)
def bert_encode(input, tokenizer):
    num_examples = len(input)
    sentence = tf.ragged.constant([
        encode_sentence(s)
        for s in input])
    cls = [tokenizer.convert_tokens_to_ids(['[CLS]'])]*sentence[:, :200].shape[0]
    input_word_ids = tf.concat([cls, sentence[:, :200]], axis=-1)
    input_mask = tf.ones_like(input_word_ids).to_tensor()
    type_cls = tf.zeros_like(cls)
    type_s = tf.zeros_like(sentence[: , :200])
    input_type_ids = tf.concat(
        [type_cls, type_s], axis=-1).to_tensor()
    inputs = {
    'input_word_ids': input_word_ids.to_tensor(),
    'input_mask': input_mask,
    'input_type_ids': input_type_ids}
    return inputs
bert_classifier, bert_encoder =bert.bert_models.classifier_model(bert_config, num_labels=2)
bert_classifier.load_weights("model_weights.h5")

def classify(Message):
    label = {0: 'Ham', 1: 'Spam'}
    X= bert_encode(Message, tokenizer)
    max_value=max(zip(bert_classifier.predict(X)[0], [0,1]))
    proba = np.exp(max_value[0])/(1+np.exp(max_value[0]))
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
    app.run(debug=True)
