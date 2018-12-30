import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import levene
from scipy.stats import mannwhitneyu

pd.set_option('display.max_columns', None)
df = pd.read_csv('planecrashinfo_20181121001952.csv')
df.head()
df.info()
df.describe()
df.shape

df1= df[['date', 'time', 'location', 'route', 'ac_type',  'aboard', 'fatalities']]
df1.head()

# работа с пропущенными значениями
# мы не можем использовать count т.к. пустые значения в csv == '?'
# df2=df.replace('?', 'NaN')'
# df2.count(axis=1)'
# мы не можем использовать heatmap т.к. пустые значения в csv == '?'
# df5 = pd.read_table('planecrashinfo_20181121001952.csv', sep=',')
# sns.heatmap(df1.isnull(),yticklabels=False,cbar=True,cmap='viridis')

# необходимо ячейки с '?' привести к NaN
# можно написать так для поиска: df2[df2.time.str.contains("?")==True]
df2=df1.replace('?', np.nan) 
sns.heatmap(df2.isnull(),yticklabels=False,cbar=True,cmap='viridis')

df3=df2.dropna(subset=['time', 'route'])
df3.head()
sns.heatmap(df3.isnull(),yticklabels=False,cbar=True,cmap='viridis')
df3.info()
df3.describe()

# отбор данных из смешанных столбцов (split)
df3["fatalities_num"] = df3["fatalities"].str.split("(", n = 1, expand=True)[0].str.strip()
df3["fatalities_num"] = pd.to_numeric(df3["fatalities_num"], errors="coerce")
df3["aboard_num"] = df3["aboard"].str.split("(", n = 1, expand=True)[0].str.strip()
df3["aboard_num"] = pd.to_numeric(df3["aboard_num"], errors="coerce")
df3['sum_k']=df3['fatalities_num']+df3['aboard_num']
df3['year']=df3['date'].str[-4:]
df3['year']=pd.to_numeric(df3['year'])


# графики (без to_frame('total_crashes') будет серия; total_killed - DF т.к. 2 столбца на входе)
total_killed= df3[["year",  "sum_k"]].groupby("year").sum()
# либо можно написать так: df3[df3['year']].groupby('year').sum_k.sum()
total_crash= df3["year"].value_counts().sort_index(ascending=True).rename_axis('year').to_frame('total_crashes')

ax = total_crash.plot(figsize=(16,4))
total_killed.plot(ax=ax, secondary_y=True)
total_crash.plot(figsize=(16,4))
total_killed.plot(figsize=(16,4))


# location
df3["location_num"]=df3["location_num"] = df3["location"].str.split(",", n = 3, expand=True)[2].str.strip() #города
df3["location_num1"]=df3["location_num"] = df3["location"].str.split(",", n = 3, expand=True)[1].str.strip()
df3["location_num2"]=df3["location_num"] = df3["location"].str.split(",", n = 3, expand=True)[0].str.strip()
sns.heatmap(df3.isnull(),yticklabels=False,cbar=True,cmap='viridis')

df3.nlargest(3, 'sum_k')[['location_num', 'sum_k']]


#вложенные where для графиков
TMCWC = df3.copy()
l=TMCWC[(TMCWC['ac_type'].str.contains("Zeppelin")==True) | (TMCWC['ac_type'].str.contains("Antonov")==True) | (TMCWC['ac_type'].str.contains("Wright")==True)]
l['company']=np.where(l['ac_type'].str.contains("Zeppelin"), "Zeppelin", np.where(l['ac_type'].str.contains("Antonov"), "Antonov", "Wright")) 
l.company.value_counts().plot(kind="bar")

# разделение столбца
Zeppelin=l.company.value_counts()[0]
Antonov=l.company.value_counts()[1]
Wright=l.company.value_counts()[2]


aboard=[
            l[(l['year'] > 1970) & (l['company']=="Zeppelin")].aboard_num.sum(),
            l[(l['year'] > 1970) & (l['company']=="Antonov")].aboard_num.sum(),
            l[(l['year'] > 1970) & (l['company']=="Wright")].aboard_num.sum()
       ]
fatality=[
             l[(l['year'] > 1970) & (l['company']=="Zeppelin")].fatalities_num.sum(),
             l[(l['year'] > 1970) & (l['company']=="Antonov")].fatalities_num.sum(),
             l[(l['year'] > 1970) & (l['company']=="Wright")].fatalities_num.sum()
         ]
print(aboard, fatality)

# графики по 3-м компаниям
import numpy as np
myarrayaboard = np.asarray(aboard)
myarrayfatality = np.asarray(fatality)
new_series_myarrayaboard = pd.Series(myarrayaboard)
new_series_myarrayfatality = pd.Series(myarrayfatality)

new_series_myarrayaboard.plot(legend = True)
new_series_myarrayfatality.plot(legend = True)

#AB testing
df3.info()
df4=df3.loc[:1600,:]
df4['fatalities_num'].hist()
df5=df4['fatalities_num'] = df4['fatalities_num'].values
df5


df6=df3.loc[1600:,:]
df6['fatalities_num'].hist()
df7=df6['fatalities_num'] = df6['fatalities_num'].values
df7

#AB Test
levene(df7, df5, center = 'median')
mannwhitneyu(df7, df5)
