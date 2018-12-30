import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv('StudentsPerformance.csv')
df.info()
df.describe()

#encoding values
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(df['lunch'])
df['lunch_enc'] = le.transform(df['lunch'])

le1 = preprocessing.LabelEncoder()
le1.fit(df['gender'])
df['gender_enc'] = le1.transform(df['gender'])

le = preprocessing.LabelEncoder()
le.fit(df['race/ethnicity'])
df['race/ethnicity_enc'] = le.transform(df['race/ethnicity'])

le = preprocessing.LabelEncoder()
le.fit(df['parental level of education'])
df['parental level of education_enc'] = le.transform(df['parental level of education'])

le = preprocessing.LabelEncoder()
le.fit(df['test preparation course'])
df['test preparation course_enc'] = le.transform(df['test preparation course'])


#вывод перекодированных значений
keys = le.classes_
values = le.transform(le.classes_)
dictionary = dict(zip(keys, values))
print(dictionary)

keys1 = le1.classes_
values1 = le1.transform(le1.classes_)
dictionary1 = dict(zip(keys1, values1))
print(dictionary1)


# passed >80
df1=df[df['math score']>80]
sns.countplot(x='math score', hue = 'gender', data = df1)

df2=df[(df['math score']>80) & (df['reading score']>80) & (df['writing score']>80)]
sns.countplot(x='reading score', hue = 'gender', data = df2)
sns.countplot(x='math score', hue = 'gender', data = df2)
sns.countplot(x='writing score', hue = 'gender', data = df2)

#create new column with LAMBDA - IF
df2['allpassed'] = df2.apply(lambda x : 'F' if x['math score'] >= 80 
   and x['reading score'] >= 80 
   and x['writing score'] >= 80 
   else 'P', axis =1)

df2['allpassed'].value_counts()

''' Percentage задали в начале число, OverAll_PassStatus создали новый столбец.
#def GetGrade(Percentage, OverAll_PassStatus):
    #if ( OverAll_PassStatus == 'F'):
        #return 'F'    
    #if ( Percentage >= 80 ):
        #return 'A'
    #if ( Percentage >= 70):
        #return 'B'
    #if ( Percentage >= 60):
        #return 'C'
    #if ( Percentage >= 50):
        #return 'D'
    #if ( Percentage >= 40):
        #return 'E'
    #else: 
        #return 'F'
#df['Grade'] = df.apply(lambda x : GetGrade(x['Percentage'], x['OverAll_PassStatus']), axis=1)
#df.Grade.value_counts()
        '''

# distplots
sns.distplot(df2['math score'])
sns.distplot(df2['reading score'])
sns.distplot(df2['writing score'])

# total mean values 
df2['Total_Marks'] = df2['math score']+df2['reading score']+df2['writing score']
df2['Percentage'] = df2['Total_Marks']/3
df2['Percentage'].hist()
df2[df2['gender']=='female']['Percentage'].hist()
df2[df2['gender']=='male']['Percentage'].hist()

# mean values 'writing'
df2[df2['gender']=='female']['writing score'].mean()
df2[df2['gender']=='male']['writing score'].mean()

#count of female 'writing score'
df2[df2['gender']=='female']['writing score'].value_counts().sum()
df2[df2['gender']=='male']['writing score'].value_counts().sum()

#AB
sns.countplot(x='Percentage', hue = 'gender', data = df2)
from scipy.stats import mannwhitneyu
mannwhitneyu(df2[df2['gender']== 'female']['Percentage'][:32], df2[df2['gender']== 'female']['Percentage'][33:])
