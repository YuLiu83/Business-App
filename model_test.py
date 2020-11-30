import tensorflow as tf
import transformers as trans
import torch
from transformers import DistilBertTokenizer, BertConfig
from transformers import AdamW, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch.nn as nn
import numpy as np

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
model.load_state_dict(torch.load('torch_weights', map_location='cpu'))

def classify(Message):
    label = {0: 'Ham', 1: 'Spam'}
    X= tokenizer(Message, max_length=200, padding=True, truncation=True, return_tensors="pt") 
    device=torch.device("cpu")
    X.to(device)
    model.to(device)
    result=model(**X)
    result_list=list(result[0][0].cpu().detach().numpy())
    max_value=max(zip(result_list, [0,1]))
    proba1=max(nn.Softmax(dim=-1)(result[0][0]).cpu().detach().numpy())
    proba=nn.Softmax(dim=-1)(result[0][0]).cpu().detach().numpy()
    #proba = np.exp(max_value[0])/(1+np.exp(max_value[0]))
    return result_list, max_value, label[max_value[1]], proba1, proba

Message="hi Bob,It was nice meeting you today. Looking forward to meeting you again soon.cheers,Da"


print(classify(Message))