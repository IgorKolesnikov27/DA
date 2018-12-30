import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
import seaborn as sns

#describe the data
df = pd.read_csv('avocado.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.head()
df.info()
df.describe()

# nullable values
df.isna().sum()
sns.heatmap(df.isnull(),yticklabels=False,cbar=True,cmap='viridis')

#distribution graphics
pl.figure(figsize=(12,5))
pl.title('AveragePrice')
sns.distplot(df["AveragePrice"], color = 'b')

# boxplot graphics
for i in df['region'].unique():
    sns.boxplot(y="region", x="AveragePrice", data=df);
for i in pd.unique(df['region']):
    sns.boxplot(y="region", x="AveragePrice", data=df);
   
sns.boxplot(y="region", x="AveragePrice", data=df[(df['region']=='Albany')])
#sns.boxplot(y="year", x="AveragePrice", data=df)
sns.boxplot(y="type", x="AveragePrice", data=df)

# factorplots
pl.figure(figsize=(120,50))
sns.catplot(x='AveragePrice', y='region', hue="year", data=df[df['type']=='organic']);
sns.boxenplot(data=df.loc[:, df.columns != 'Date'], orient="h", palette="Set2")

df1=df.iloc[:,1:11]
sns.heatmap(df1.corr())
df2 = df1.drop(['XLarge Bags','Date'], axis = 1)


mask = df['type']=='conventional'
g = sns.factorplot('AveragePrice','region',data=df[mask],
                   hue='year',
                   size=8,
                   aspect=0.6,
                   palette='Blues',
                   join=False,
              )
# new sort
regions = ['PhoenixTucson', 'Chicago']
mask1 = (df['region'].isin(regions) & (df['type']=='conventional'))
# find the months
df['Month'] = df['Date'].dt.month
df[mask1].head()
# plot prices over time # factorplots
g = sns.factorplot('Month','AveragePrice',data=df[mask1], hue='year', row='region', aspect=2,palette='Blues')
