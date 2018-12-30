import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib as mlp
import matplotlib.pyplot as plt

df= pd.read_csv('BlackFriday.csv')
df.info()
df.dtypes
df.describe()

# null values
sns.heatmap(df.isnull(),yticklabels=False,cbar=True,cmap='viridis')
df.isna().any()

df['Product_Category_2'].unique()
df['Product_Category_3'].unique()
df.fillna(value=0,inplace=True)
# np.unique(df['Product_Category_2'])

sns.heatmap(df.corr())
sns.heatmap(df.isnull(),yticklabels=False,cbar=True,cmap='viridis')
df.isna().any()

#change the type of columns
df[['Product_Category_2', 'Product_Category_3']]=df[['Product_Category_2', 'Product_Category_3']].astype(dtype=np.int64)
df.info()

# df - drop
df.drop(columns = ["User_ID","Product_ID"],inplace=True)

# main graphics
df.hist()
# Age/Gender Graphics
sns.countplot(x="City_Category", hue="Age", data=df)
sns.catplot(x="City_Category", hue="Age", col="Gender", data=df, kind="count", height=4, aspect=1);
sns.countplot(df['Age'],hue=df['Gender'])

#check marital status
df["marital_combine"] = df["Gender"].map(str) + df["Marital_Status"].map(str)
# or we can use next template:
#df2['Age_Encoded'] = df2['Age'].map({'0-17':0,'18-25':1,
                          #'26-35':2,'36-45':3,
                          #'46-50':4,'51-55':5,
#check Gender                          #'55+':6})  
def map_gender(gender):
    if gender == 'M':
        return 1
    else:
        return 0                          
df['Gender_10'] = df['Gender'].apply(map_gender)
# Age/Gender Graphics №2
sns.countplot(df['Age'],hue=df['marital_combine'])
sns.catplot(x="City_Category", hue="Age", col="marital_combine", data=df, kind="count", height=4, aspect=1);

#check gender - pie chart
fig1, ax1 = plt.subplots(figsize=(10,7))
ax1.pie(df['Gender'].value_counts(),labels=['Male','Female'], autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  
plt.tight_layout()
plt.legend()
plt.show()

#check gender/Purchase - bar chart
df[['Gender','Purchase']].groupby('Gender').mean().plot.bar()
sns.barplot('Gender', 'Purchase', data = df)
plt.show()

#check Age/Purchase - bar chart
sns.boxplot('Age','Purchase', data = df)
plt.show()

#check User_ID/Purchase - lmplot chart
df1= pd.read_csv('BlackFriday.csv')
sns.lmplot('User_ID','Purchase',data=df1,fit_reg=False,hue='Gender',aspect=2.5)
sns.lmplot('User_ID','Purchase',data=df1,fit_reg=False,hue='Age',aspect=2.5)

#check Occupation/Purchase - lmplot chart
plt.figure(figsize=(12,6))
prod_by_occ = df.groupby(by='Occupation').nunique()['Purchase']
sns.barplot(x=prod_by_occ.index,y=prod_by_occ.values)
plt.title('Unique Products by Occupation')
plt.show()

# check TOP 10 Product_ID Purchases
fig1, ax1 = plt.subplots(figsize=(12,7))
df1.groupby('Product_ID')['Purchase'].sum().nlargest(10).sort_values().plot('barh')
