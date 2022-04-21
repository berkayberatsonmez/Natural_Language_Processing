# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 20:05:29 2022

@author: berkayberatsonmez
"""
import numpy as np
import pandas as pd
import re

yorumlar = pd.read_csv('Restaurant_Reviews.txt') 

#stop word temizleme(bir anlam ifade etmeyen kelimeler)
import nltk
nltk.download('stopwords')

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

from nltk.corpus import stopwords



#Veri Temizleme (PreProcessing)
derleme = []
for i in range(1000):
    yorum = re.sub('[^a-zA-z]',' ',yorumlar['Review'][i]) #küçük harfleri yada büyük harfleri içermeyenleri boşluk ile değiştir
    yorum = yorum.lower() 
    yorum = yorum.split() 
    yorum = [ps.stem(kelime) for kelime in yorum if not kelime in set(stopwords.words('english'))] 
    yorum =  ' '.join(yorum)
    derleme.append(yorum)



#Feature Extraction(Bag of Words(BOW))
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features= 2000) 

X = cv.fit_transform(derleme).toarray() #bağımsız değişken
y = yorumlar.iloc[:,1].values #bağımlı değişken



#Machine Learning
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 123)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(x_train,y_train)

y_pred = gnb.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)







