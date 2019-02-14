import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import re

df = pd.read_csv('amazon_jobs_dataset.csv')
df.info()
df.isna().any()
sns.heatmap(df.isna())
df.describe()
df.columns
pd.set_option('display.max_columns', None)
df.head()
df.dropna(inplace=True)

#freq
df['Title'].value_counts().head(10)
df['location'].value_counts().head(10)
df['Posting_date'].value_counts().head(30)

#parsing
languages_list = ['swift','matlab','mongodb','hadoop','cosmos', 'mysql','spark', 'pig', 'python', 'java', 'c++', 'php', 'javascript', 'objectivec', 'ruby', 'perl','c','c#']

qualifications = df['BASIC QUALIFICATIONS'].tolist()+df['PREFERRED QUALIFICATIONS'].tolist()
qualifications_string = "".join(re.sub('[·,-/’()]', '', str(v)) for v in qualifications).lower()

wordcount = dict((x,0) for x in languages_list)
# + и # это не regex операторы это символы которые надо искать в тексте [] 
#указано как поиск одного из символов которые внутри
# ' как бы разделдение на 2 части ищем и то и то
wordcount = dict((x,0) for x in languages_list)
for w in re.findall(r"[\w'+#]+", qualifications_string):
    if w in wordcount:
        wordcount[w] += 1
# print
print(wordcount)

# items() сортирует пары значений если не указано аргументов функции items()
# sorted() сортирует то что присвоили, но при этом создает копию отсортрованного. т.е. wordcount первоначальный
# не отсортирован, сортировка присвоена к programming_language_popularity
programming_language_popularity = sorted(wordcount.items(), key=lambda kv: kv[1], reverse=True)

# make a new dataframe from programming languages and their popularity
df_popular_programming_lang = pd.DataFrame(programming_language_popularity,columns=['Language','Popularity'])
# Capitalize each programming language first letter
df_popular_programming_lang['Language'] = df_popular_programming_lang.Language.str.capitalize()

df_popular_programming_lang
df_popular_programming_lang.plot.bar(x='Language',y='Popularity',figsize=(15,15), legend=False)

# parse стран и городов через запятую
df['new']=df['location'].str.extract(r'(^\w+)', expand=False)
df['new2']=df['location'].str.extract(r'( \w+)')
df['new3']=df['location'].str.extract(r'([^,]+$)', expand=False)


