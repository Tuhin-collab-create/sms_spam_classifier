import pickle
import streamlit as st


from nltk.tokenize import word_tokenize
import re
from nltk.corpus import stopwords
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    text= y[:]
    y.clear()
    for j in text:
        if j not in stopwords.words('english') and j not in string.punctuation:
            y.append(j)
    text=y[:]
    y.clear()
    for k in text:
        y.append(ps.stem(k))
    return y

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title('SMS Spam Classifier')
input_sms = st.text_input('Enter the sms')

if st.button("Predict"):

    # 1. preprocess
    transform_sms = transform_text(input_sms)

    # 2. join tokens
    transform_sms = " ".join(transform_sms)

    # 3. vectorize
    vector_input = tfidf.transform([transform_sms])

    # 4. predict
    result = model.predict(vector_input)[0]

    # 5. display
    if result == 1:
        st.header("SPAM")
    else:
        st.header("NOT SPAM")



