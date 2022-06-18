#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 16:52:12 2022

@author: navyareddy
"""

import pandas as pd
import streamlit as st
import docx2txt
import pdfplumber
import re
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot  as plt
nltk.download('wordnet')
nltk.download('stopwords')
import plotly.express as px
stop=set(stopwords.words('english'))
import pickle
vectors = pickle.load(open(r'/Users/navyareddy/Desktop/project_env/vect.pkl','rb'))
model = pickle.load(open(r'/Users/navyareddy/Desktop/project_env/rf_model.pkl','rb'))

resume = []

def display(doc_file):
    if doc_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        resume.append(docx2txt.process(doc_file))
    else :
        with pdfplumber.open(doc_file) as pdf:
            pages=pdf.pages[0]
            resume.append(pages.extract_text())
    return resume
def preprocess(sentence):
    sentence=str(sentence)
    sentence = sentence.lower()
    sentence=sentence.replace('{html}',"")
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence)
    rem_url=re.sub(r'http\S+', '',cleantext)
    rem_num = re.sub('[0-9]+', '', rem_url)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)
    filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    lemma_words=[lemmatizer.lemmatize(w) for w in filtered_words]
    return " ".join(lemma_words)
def main():
    st.title('DOCUMENT CLASSIFICATION')
    upload_file = st.file_uploader('Hey,Upload Your Resume ',
                                type= ['docx','pdf'],accept_multiple_files=True)
    if st.button('process'):
        st.write('upload_file')
        for doc_file in upload_file:
            if doc_file is not None:
                file_details = {'filename':[doc_file.name],
                               'filetype':doc_file.type.split('/')[-1],
                               'filesize':str(doc_file.size)+' KB'}
                file_type=pd.DataFrame(file_details)
                st.write(file_type.set_index('filename'))
                
                displayed=display(doc_file)
                #st.write(displayed)

                cleaned=preprocess(display(doc_file))
                #st.write(vectors.transform([cleaned]))
                #st.write(cleaned)
                predicted= model.predict(vectors.transform([cleaned]))
                st.write(predicted)
                st.header(string)

        
        
        
        
        
        
        
        
        
        
if __name__ == '__main__':
    main()