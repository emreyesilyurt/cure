import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import ListedColormap
import seaborn as sns
import plotly as py
import plotly.graph_objs as go
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
from  sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

import warnings
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


df = pd.read_csv('marketing_campaign.csv', sep="\t")
df.dropna(inplace = True)

df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"])
dates = []
for i in df["Dt_Customer"]:
    i = i.date()
    dates.append(i)  

print("Registration date of the newest customer on record:",max(dates))
print("Registration date of the oldest customer on record:",min(dates))

days = []
d1 = max(dates)
for i in dates:
    delta = d1 - i
    days.append(delta)
df["Customer_For"] = days
df["Customer_For"] = pd.to_numeric(df["Customer_For"], errors="coerce")

print("Unique value in Martial Status\n", df["Marital_Status"].value_counts(), "\n")
print("Unique value in Education\n", df["Education"].value_counts())



df["Age"] = 2021-df["Year_Birth"]
df["Spent"] = df["MntWines"]+ df["MntFruits"]+ df["MntMeatProducts"]+ df["MntFishProducts"]+ df["MntSweetProducts"]+ df["MntGoldProds"]
df["Living_With"] = df["Marital_Status"].replace({"Married":"Partner", "Together":"Partner", "Absurd":"Alone", "Widow":"Alone", "YOLO":"Alone", "Divorced":"Alone", "Single":"Alone",})
df["Children"] = df["Kidhome"] + df["Teenhome"]
df["Family_Size"] = df["Living_With"].replace({"Alone": 1, "Partner":2}) + df["Children"]
df["Is_Parent"] = np.where(df.Children > 0, 1, 0)
df["Education"] = df["Education"].replace({"Basic":"Undergraduate","2n Cycle":"Undergraduate", "Graduation":"Graduate", "Master":"Graduate", "PhD":"Graduate"})
df = df.rename(columns={"MntWines": "Wines", "MntFruits":"Fruits", "MntMeatProducts":"Meat", "MntFishProducts":"Fish", "MntSweetProducts":"Sweets", "MntGoldProds":"Gold"})
to_drop = ["Marital_Status", "Dt_Customer", "Z_CostContact", "Z_Revenue", "Year_Birth", "ID"]
df = df.drop(to_drop, axis=1)
le = LabelEncoder()
df['Education'] = df[['Education']].apply(le.fit_transform)
df['Living_With'] = df[['Living_With']].apply(le.fit_transform)

personal = ["Income", "Recency", "Age", "Spent"]
 
df['Age'].quantile(0.25)
df['Age'].quantile(0.75)
Q1 = df['Age'].quantile(0.25)
Q3 = df['Age'].quantile(0.75)
IQR = Q3 - Q1
lower_lim = Q1 - 1.5 * IQR
upper_lim = Q3 + 1.5 * IQR

outliers_low = (df['Age'] < lower_lim)
outliers_up = (df['Age'] > upper_lim)
len(df['Age'] - (len(df['Age'][outliers_low] + len(df['Age'][outliers_up]))))
df['Age'][(outliers_low | outliers_up)]
df['Age'][~(outliers_low | outliers_up)]
df = df[~(outliers_low | outliers_up)]


Q1 = df['Income'].quantile(0.25)
Q3 = df['Income'].quantile(0.75)
IQR = Q3 - Q1

lower_lim = Q1 - 1.5 * IQR
upper_lim = Q3 + 1.5 * IQR

outliers_low = (df['Income'] < lower_lim)
outliers_up = (df['Income'] > upper_lim)
len(df['Income'] - (len(df['Income'][outliers_low] + len(df['Income'][outliers_up]))))
df['Income'][(outliers_low | outliers_up)]
df['Income'][~(outliers_low | outliers_up)]
df = df[~(outliers_low | outliers_up)]

new_df = df.copy()
cols_del = ['AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1','AcceptedCmp2', 'Complain', 'Response']
new_df = new_df.drop(cols_del, axis=1)

sc = StandardScaler()
sc.fit(new_df)
scaled_df = pd.DataFrame(sc.transform(new_df), columns = new_df.columns )

pca = PCA(n_components=3)
pca.fit(scaled_df)
pca_df = pd.DataFrame(pca.transform(scaled_df), columns=(["col1","col2", "col3"]))
pca_df.describe().T

pca_df.to_csv(r'/home/revo/Documents/cure/campaign2.csv', index = False)
