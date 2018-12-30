import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('who_suicide_statistics.csv')
df.head()
df.shape
df.info()
df.describe()
df.columns

#we have to change the data in the columns by this way.but without 2 new columns
df1=df.replace({'age': {'15-24 years': 1}})
df2=df1.replace({'age': {'5-14 years': 0}})
age_coder=df2.replace({'age': {'25-34 years': 2, '35-54 years':3, '55-74 years':4, '75+ years':5}})
 
gender_coder=age_coder.replace({'female':0,'male':1})         



# but we can change this by another way. :
#suicide_analysis['age_encoder'] = suicide_analysis['age'].map(age_coder)
#suicide_analysis['sex_encoder'] = suicide_analysis['sex'].map(gender_coder) 

gender_coder.head()
gender_coder.fillna(0,inplace=True)
age1=gender_coder.groupby('age')['suicides_no'].sum()
age1.head()
sex2=gender_coder.groupby('sex')['suicides_no'].sum()
sex2.head()
country=gender_coder.groupby('country')['suicides_no'].sum()
country.head()
year=gender_coder.groupby('year')['suicides_no'].sum()
year.head()

female_year=gender_coder[gender_coder['sex']==0][['country','year', 'age', 'suicides_no']]
female_year2=female_year.groupby('year')['suicides_no'].sum()
female_age=gender_coder[(gender_coder['age']>0) & (gender_coder['sex']==0)][['country','year', 'age', 'suicides_no']]
female_age2=female_year.groupby('age')['suicides_no'].sum()


#графики
color=['red','green','blue','orange','gray','#222111']
plt.figure(figsize=(16,12))
sns.swarmplot(x='year',y='suicides_no',hue='age',data=gender_coder,palette=color)
plt.title("Suicide Based On The Year And Age Group")
plt.xticks(rotation=90)
plt.ylabel("Suicide Number")


en = {0:'5-14 years',
      1:'15-24 years',
      2:'25-34 years',
      3:'35-54 years',
      4:'55-74 years',
      5:'75+ years'}
gen = {0:'female',1:'male'}

plt.figure(figsize=(12,5))
sns.barplot(x=age1.index.map(en.get),y=age1.suicides_no)
plt.title("Total Suicide based in Age group")
plt.xlabel("Age Group")
plt.ylabel("Number of Suicide")



suicide_country_age = gender_coder.groupby(['country','age']).sum()['suicides_no'].reset_index()
suicide_country_age.head()







# черновик
pd.set_option('display.max_columns', 10)
pd.options.display.max_rows

