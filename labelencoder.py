import pandas as pd
from sklearn import preprocessing 


df = pd.read_csv('50_Startups.csv')


from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
d = defaultdict(LabelEncoder)

# Encoding the variable
fit = df.apply(lambda x: d[x.name].fit_transform(x))


# Inverse the encoded
fit.apply(lambda x: d[x.name].inverse_transform(x))

# Using the dictionary to label future data
df1 = df.apply(lambda x: d[x.name].transform(x))
df1

df['State'] = df1['State']












#@# черновик




from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
d = defaultdict(LabelEncoder)

# Encoding the variable
fit = df2.apply(lambda x: d[x.name].fit_transform(x))

# Inverse the encoded
fit.apply(lambda x: d[x.name].inverse_transform(x))

# Using the dictionary to label future data
df2.apply(lambda x: d[x.name].transform(x))



'le = preprocessing.LabelEncoder()
'df4['country'] = le.fit_transform(df4['country'].astype(str))
'fit = df4.apply(lambda x: d[x.name].fit_transform(x))

# Inverse the encoded
'fit.apply(lambda x: d[x.name].inverse_transform(x))