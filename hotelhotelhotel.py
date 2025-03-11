#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib
import matplotlib.patches as mpatches
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
import xgboost as xgb

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional, Activation, GRU, BatchNormalization
from tensorflow.keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text

nltk.download('stopwords')
stop_words = stopwords.words('english')


# In[ ]:


# module_url
# use = load(module_url)


# In[2]:


data = pd.read_csv("Hotel_Reviews.csv")
data.head()


# In[3]:


data.describe().T


# In[4]:


data.isna().sum()


# In[5]:


data.dropna(inplace=True,axis=0)


# In[6]:


data.isna().sum()


# In[7]:


data["Hotel_Address"].head(10)


# In[8]:


print("Duplicated rows before: ",data.duplicated().sum())
data.drop_duplicates(inplace=True)
print("Duplicated rows after: ",data.duplicated().sum())


# In[9]:


data["Hotel_Address"]=data["Hotel_Address"].str.replace("United Kingdom","UK")


# In[10]:


data.head()


# ### EDA

# In[12]:


import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot


# ### Visualization

# In[13]:


hotel_trace = go.Scattermapbox(
    lat=data['lat'],
    lon=data['lng'],
    mode='markers',
    marker=dict(
        size=4,
        color='cyan',
        opacity=0.7,
    ),
    text=data['Hotel_Name'],
    hoverinfo='text',
)


layout = go.Layout(
    autosize=False,
    hovermode='closest',
    mapbox= dict(accesstoken="pk.eyJ1Ijoic2hhejEzIiwiYSI6ImNqYXA3NjhmeDR4d3Iyd2w5M2phM3E2djQifQ.yyxsAzT94VGYYEEOhxy87w",
        center=dict(lat=48.8566, lon=2.3522),
        zoom=12,
    ),
    width=900, 
    height=600,title = "Drop off locations of hotels" 
)


fig = go.Figure(data=[hotel_trace], layout=layout)


iplot(fig)


# In[14]:


text = " ".join(data[data["Reviewer_Score"] >= 8]["Positive_Review"])
wordcloud = WordCloud(width=800, height=400).generate(text)
fig = px.imshow(wordcloud, title="Word Cloud of Positive Reviews")
fig.show()


# In[15]:


text = " ".join(data[data["Reviewer_Score"] <= 4]["Negative_Review"])
wordcloud = WordCloud(width=800, height=400).generate(text)
fig = px.imshow(wordcloud, title="Word Cloud of Negative Reviews")
fig.show()


# In[16]:


data.Reviewer_Nationality.nunique()


# In[17]:


nationality = data["Reviewer_Nationality"].value_counts(dropna=False)[:10]

fig = px.bar(x=nationality.index, y=nationality.values, color=nationality.index,
             title="Top 10 Nationalities of Reviewers")
fig.update_layout(xaxis_title="Nationality", yaxis_title="Review Count", font=dict(size=14))
fig.show()


# In[ ]:




