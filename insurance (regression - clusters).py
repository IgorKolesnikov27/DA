import numpy as np 
import pandas as pd 
import matplotlib.pyplot as pl
import seaborn as sns

df=pd.read_csv('insurance.csv')
df.head()
df.info()
df.describe()
df.isna().sum()
df.nunique()
df.count()

#encoding variables
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(df['sex'])
df['sex_enc'] = le.transform(df['sex'])
le.fit(df['smoker'])
df['smoker_enc'] = le.transform(df['smoker'])
le.fit(df['region'])
df['region_enc'] = le.transform(df['region'])

#вывод перекодированных значений
keys = le.classes_
values = le.transform(le.classes_)
dictionary = dict(zip(keys, values))
print(dictionary)


   
# finding missing values and corr()
sns.heatmap(df.isnull(),yticklabels=False,cbar=True,cmap='viridis')
sns.heatmap(df.corr(),yticklabels=True,cbar=True)
df.corr()

#graphics of smokers
pl.figure(figsize=(12,5))
pl.title("graphics of smokers")
sns.distplot(df[(df.smoker_enc == 1)]["charges"],color='g')
sns.distplot(df[(df.smoker_enc == 0)]['charges'],color='y')
df[df['smoker_enc'] == 1]["charges"].describe()
df[df['smoker_enc'] == 0]["charges"].describe()

sns.catplot(x="smoker", kind="count", hue="sex", data=df)
sns.catplot(x="smoker", kind="count", hue="region", data=df)
sns.catplot(x="smoker", kind="count", hue="children", data=df)
sns.catplot(x="smoker", y='bmi', kind="violin", hue="sex", data=df)
sns.catplot(x="smoker", y='charges', kind="violin", hue="sex", data=df)
sns.boxplot(x="smoker", y="charges", data=df)

   
#graphics of old persons   
pl.figure(figsize=(12,5))
pl.title("graphics of old persons")
sns.distplot(df[(df['age'] >= 30)]["charges"],color='g')
sns.distplot(df[(df['age'] <= 30)]["charges"],color='g')
sns.boxplot(y="smoker", x="charges", data = df[(df.age == 18)] , orient="h", palette = 'pink')
df[df['age'] >= 30]["charges"].describe()
df[df['age'] <= 30]["charges"].describe()

df[['age', 'charges']].groupby('age').sum().plot.bar()
df[['age', 'charges']].groupby('age').mean().plot.bar()

# encoding  2
from sklearn import preprocessing
le1 = preprocessing.LabelEncoder()
le1.fit(df['age'])
df['age_enc'] = le1.transform(df['age'])
#вывод перекодированных значений
keys = le1.classes_
values = le1.transform(le1.classes_)
dictionary = dict(zip(keys, values))
print(dictionary)
df[['age_enc', 'charges']].groupby('age_enc').median().plot()

# dist plots
sns.distplot(df["age"], color = 'g')
sns.distplot(df["charges"], color = 'g')
sns.distplot(df["bmi"], color = 'g')
sns.scatterplot(x='bmi',y='charges',data=df,hue='smoker')







# Importing the libraries
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as pl
import pandas as pd

# Importing the dataset
# x = df.drop(['charges'], axis = 1)
df1=df.drop(columns=['sex', 'smoker', 'region'])
X = df1.drop(columns=['charges']).values
y = df1.iloc[:, 3].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# results
print(regressor.score(X_test,y_test))
r2_score(y_test,y_pred)



# Polynomial Regression

# Importing the libraries
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as pl
import pandas as pd

# Importing the dataset
# x = df.drop(['charges'], axis = 1)
df1=df.drop(columns=['sex', 'smoker', 'region'])
X = df1.drop(columns=['charges']).values
y = df1.iloc[:, 3].values

quad = PolynomialFeatures (degree = 2)
x_quad = quad.fit_transform(X)

X_train,X_test,Y_train,Y_test = train_test_split(x_quad,y, random_state = 0)

plr = LinearRegression().fit(X_train,Y_train)

Y_train_pred = plr.predict(X_train)
Y_test_pred = plr.predict(X_test)

print(plr.score(X_test,Y_test))









# Random Forest Regression

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as pl
import pandas as pd

# Importing the dataset
# x = df.drop(['charges'], axis = 1)
df1=df.drop(columns=['sex', 'smoker', 'region'])
X = df1.drop(columns=['charges']).values
y = df1.iloc[:, 3].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor.fit(X, y)

# Predicting a new result
forest_train_pred = regressor.predict(X_train)
forest_test_pred = regressor.predict(X_test)


r2_score(y_train,forest_train_pred)
r2_score(y_test,forest_test_pred)






# K-Means Clustering

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as pl
import pandas as pd

# Importing the dataset
# x = df.drop(['charges'], axis = 1)
df1=df.drop(columns=['sex', 'smoker', 'region'])
X = df1.drop(columns=['charges']).values
y = df1.iloc[:, 3].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 5):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
pl.plot(range(1, 5), wcss)
pl.title('The Elbow Method')
pl.xlabel('Number of clusters')
pl.ylabel('WCSS')
pl.show()

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 2, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

# Visualising the clusters
pl.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
pl.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
pl.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
pl.title('Clusters of customers')
pl.xlabel('Annual Income (k$)')
pl.ylabel('Spending Score (1-100)')
pl.legend()
pl.show()
