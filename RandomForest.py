# учебный прогон
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('winemag-data-130k-v2.csv')
df

# duplicates
df1=df[df.duplicated('description',keep='first')].sort_values(by=['points'], ascending = False)
df1

df2 = df1.drop('description', axis = 1)
df2

# Exploratory Analysis - correlation замена типа данных на INT
df2['price']=df2['price'].fillna(value=df['price'].mean())
df2['price']=df2['price'].astype(int)

from scipy.stats import pearsonr
print("Pearson Correlation:", pearsonr(df2.price, df2.points))
sns.heatmap(df2.corr())

ax = sns.boxplot(x="price", y="points", data=df2)
ax = sns.swarmplot(x="price", y="points", data=df2, color=".25")
plt.show()

# delete observations which consists less than 200 observations
l4=df2.country.value_counts()[:15]
l4

# графики
country=df2.groupby('country').filter(lambda x: len(x) >50)
df3 = pd.DataFrame({col:vals['points'] for col,vals in country.groupby('country')})
meds = df3.median()
meds.sort_values(ascending=False, inplace=True)

fig, ax = plt.subplots(figsize = (15,5))
chart = sns.boxplot(x='country',y='points', data=country, order=meds.index, ax = ax)
plt.xticks(rotation = 90)

plt.show()


# замена стран на Коды 
import pandas as pd
from sklearn import preprocessing 

df4 = df2

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
d = defaultdict(LabelEncoder)


# Encoding the variable
df4['country'].astype(str)


#словарь с закодированными переменными
keys = le.classes_
values = le.transform(le.classes_)
dictionary = dict(zip(keys, values))
print(dictionary)



# Random Forest Classification

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
X = df4.iloc[:, [3, 4]].values
y = df4.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 500, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)


sc.get_params()

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))



-----****-----
# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
df5 = df4
dataset = pd.read_csv('winemag-data-130k-v2.csv')
X1 = df5.iloc[:, [3, 4]].values
y1 = df5.iloc[:, 1].values

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X1, y1)

# Predicting a new result
y_pred1 = regressor.predict(X1)
y_pred2=y_pred1.astype(np.int64)


from sklearn.metrics import classification_report
print(classification_report(y1,y_pred2))



