#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import hist
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing 
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.metrics import confusion_matrix,accuracy_score


# In[4]:


df_athlete = pd.read_csv('athlete_events.csv')
df_regions = pd.read_csv('noc_regions.csv')


# In[6]:


df_athlete.head()


# In[8]:


df_athlete.info()


# In[9]:


df_athlete.describe()


# In[10]:


df_athlete.isna().any()


# In[12]:


sns.heatmap(data=df_athlete.isna())


# In[7]:


df_regions.head()


# In[13]:


df_regions.info()


# In[14]:


df_regions.describe()


# In[18]:


df_regions.isna().any()


# In[20]:


sns.heatmap(data=df_regions.isna())


# In[21]:


merge_2DF = pd.merge(df_athlete, df_regions, how='left', on = 'NOC')


# In[22]:


merge_2DF


# In[25]:


merge_2DF.info()
merge_2DF.describe()


# In[26]:


sns.heatmap(data=merge_2DF.isna())


# In[205]:


merge_2DF_1=merge_2DF[(merge_2DF['Medal']=='Gold') | (merge_2DF['Medal']=='Silver') | (merge_2DF['Medal']=='Bronze')]


# In[206]:


merge_2DF_1


# In[54]:


merge_2DF_1.isnull().any()


# In[55]:


merge_2DF_1.info()


# In[89]:


merge_2DF_1


# In[209]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(merge_2DF_1['Medal'])
merge_2DF_1['Medal_1'] = le.transform(merge_2DF_1['Medal'])
keys = le.classes_
values = le.transform(le.classes_)
dictionary = dict(zip(keys, values))
print(dictionary)


# In[210]:


sns.heatmap(merge_2DF_1.corr())


# In[ ]:


#Exploring counts


# In[134]:


import seaborn as sns
sns.set(rc={'figure.figsize':(8,4)})
sns.countplot("Sex", data=merge_2DF_1)


# In[82]:


merge_2DF_1[merge_2DF_1['Sex']=='M']['Medal'].count()


# In[83]:


merge_2DF_1[merge_2DF_1['Sex']=='F']['Medal'].count()


# In[97]:


sns.catplot(y="Sex",hue="Medal", data=merge_2DF_1, kind="count")


# In[108]:


GM=merge_2DF_1[(merge_2DF_1['Sex']=='M') & (merge_2DF_1['Medal']=='Gold')]['Medal'].count()
print('Count of (Male) Gold medals-',GM)


# In[109]:


SM=merge_2DF_1[(merge_2DF_1['Sex']=='M') & (merge_2DF_1['Medal']=='Silver')]['Medal'].count()
print('Count of (Male) Silver medals-',SM)


# In[110]:


BM=merge_2DF_1[(merge_2DF_1['Sex']=='M') & (merge_2DF_1['Medal']=='Bronze')]['Medal'].count()
print('Count of (Male) Bronze medals-',BM)


# In[111]:


GM=merge_2DF_1[(merge_2DF_1['Sex']=='F') & (merge_2DF_1['Medal']=='Gold')]['Medal'].count()
print('Count of (Female) Gold medals-',GM)


# In[112]:


SM=merge_2DF_1[(merge_2DF_1['Sex']=='F') & (merge_2DF_1['Medal']=='Silver')]['Medal'].count()
print('Count of (Female) Silver medals-',SM)


# In[113]:


BM=merge_2DF_1[(merge_2DF_1['Sex']=='F') & (merge_2DF_1['Medal']=='Bronze')]['Medal'].count()
print('Count of (Female) Bronze medals-',BM)


# In[181]:


#Exploring Ages

import seaborn as sns
sns.set(rc={'figure.figsize':(19,10)})
sns.countplot(merge_2DF_1['Age'])


# In[ ]:


# age of medalists (Male)


# In[127]:


A_GM=merge_2DF_1[(merge_2DF_1['Sex']=='M') & (merge_2DF_1['Medal']=='Gold')]['Age'].hist()
print('Mean age of gold medalists-',merge_2DF_1[(merge_2DF_1['Sex']=='M') & (merge_2DF_1['Medal']=='Gold')]['Age'].mean())
print('Median age of gold medalists-',merge_2DF_1[(merge_2DF_1['Sex']=='M') & (merge_2DF_1['Medal']=='Gold')]['Age'].median())


# In[124]:


A_GM=merge_2DF_1[(merge_2DF_1['Sex']=='M') & (merge_2DF_1['Medal']=='Silver')]['Age'].hist()
print('Mean age of silver medalists-',merge_2DF_1[(merge_2DF_1['Sex']=='M') & (merge_2DF_1['Medal']=='Silver')]['Age'].mean())
print('Median age of silver medalists-',merge_2DF_1[(merge_2DF_1['Sex']=='M') & (merge_2DF_1['Medal']=='Silver')]['Age'].median())


# In[125]:


A_GM=merge_2DF_1[(merge_2DF_1['Sex']=='M') & (merge_2DF_1['Medal']=='Bronze')]['Age'].hist()
print('Mean age of bronze medalists-',merge_2DF_1[(merge_2DF_1['Sex']=='M') & (merge_2DF_1['Medal']=='Bronze')]['Age'].mean())
print('Median age of bronze medalists-',merge_2DF_1[(merge_2DF_1['Sex']=='M') & (merge_2DF_1['Medal']=='Bronze')]['Age'].median())


# In[ ]:


# age of  medalists (Female)


# In[129]:


A_GM=merge_2DF_1[(merge_2DF_1['Sex']=='F') & (merge_2DF_1['Medal']=='Gold')]['Age'].hist()
print('Mean age of gold medalists-',merge_2DF_1[(merge_2DF_1['Sex']=='F') & (merge_2DF_1['Medal']=='Gold')]['Age'].mean())
print('Median age of gold medalists-',merge_2DF_1[(merge_2DF_1['Sex']=='F') & (merge_2DF_1['Medal']=='Gold')]['Age'].median())


# In[130]:


A_GM=merge_2DF_1[(merge_2DF_1['Sex']=='F') & (merge_2DF_1['Medal']=='Silver')]['Age'].hist()
print('Mean age of silver medalists-',merge_2DF_1[(merge_2DF_1['Sex']=='F') & (merge_2DF_1['Medal']=='Silver')]['Age'].mean())
print('Median age of silver medalists-',merge_2DF_1[(merge_2DF_1['Sex']=='F') & (merge_2DF_1['Medal']=='Silver')]['Age'].median())


# In[136]:


A_GM=merge_2DF_1[(merge_2DF_1['Sex']=='F') & (merge_2DF_1['Medal']=='Bronze')]['Age'].hist()
print('Mean age of bronze medalists-',merge_2DF_1[(merge_2DF_1['Sex']=='F') & (merge_2DF_1['Medal']=='Bronze')]['Age'].mean())
print('Median age of bronze medalists-',merge_2DF_1[(merge_2DF_1['Sex']=='F') & (merge_2DF_1['Medal']=='Bronze')]['Age'].median())


# In[ ]:


# stats/medals team - Russia


# In[178]:


A_GM=merge_2DF_1[(merge_2DF_1['Team']=='Russia') & (merge_2DF_1['Medal']=='Gold')]['Age'].hist()
A_GM=merge_2DF_1[(merge_2DF_1['Team']=='Russia') & (merge_2DF_1['Medal']=='Gold')]['ID'].count()
print('Count of Russia gold medalists-',A_GM)


# In[196]:


# trends
merge_2DF_1_M = merge_2DF_1[(merge_2DF_1.Sex == 'M') & (merge_2DF_1.Team == 'Russia')]
sns.countplot(x='Year', data=merge_2DF_1_M)


# In[199]:


merge_2DF_1_F = merge_2DF_1[(merge_2DF_1.Sex == 'F') & (merge_2DF_1.Team == 'Russia')]
sns.countplot(x='Year', data=merge_2DF_1_F)


# In[201]:


sns.countplot(x='Year', data=merge_2DF_1)


# In[ ]:


# clasterization


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


стата по тем у кого нет медалей

