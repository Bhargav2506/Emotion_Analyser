import numpy as np
import pandas as pd
import seaborn as sns

import neattext.functions as nfx

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from sklearn.feature_extraction.text import TfidfTransformer

df = pd.read_csv("emotion_dataset_2.csv")
df.head()
df.info()
df['Emotion'].value_counts()
dir(nfx)
df['Clean_Text'] = df['Text'].apply(nfx.remove_userhandles)
df['Clean_Text'] = df['Text'].apply(nfx.remove_stopwords)
df['Clean_Text'] = df['Text'].apply(nfx.remove_punctuations)
print(df['Clean_Text'])
print(df['Text'])
X = df['Clean_Text']
y = df['Emotion']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
from sklearn.pipeline import Pipeline
pipe = Pipeline(steps = [('cv',CountVectorizer()),
                        ('lg',LogisticRegression())
                         ])
pipe.fit(X_train,y_train)                         
pipe
pipe.score(X_test,y_test)
exl = "I had bad day at school"
pipe.predict([exl])
pipe.predict_proba([exl])
pipe.classes_
import joblib
pipe_file = open("emotion_classifier_pipe_09_sept_2022.pkl","wb")
joblib.dump(pipe,pipe_file)
pipe_file.close()
