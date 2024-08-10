import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import pickle
plt.style.use("ggplot")
df = pd.read_csv("yield_df.csv")
df.drop('Unnamed: 0', axis=1, inplace=True)
df.info()
df.isnull().sum()
df.duplicated().sum()
df.drop_duplicates(inplace=True)
plt.figure(figsize=(15,20))
sns.countplot(y=df['Area'])
plt.show()
plt.figure(figsize=(15,20))
sns.countplot(y=df['Item'])
plt.show()
yield_per_country = df.groupby('Area')['hg/ha_yield'].sum().reset_index()
plt.figure(figsize=(15,20))
sns.barplot(y=yield_per_country['Area'], x=yield_per_country['hg/ha_yield'])
plt.show()
yield_per_crop = df.groupby('Item')['hg/ha_yield'].sum().reset_index()
plt.figure(figsize=(15,20))
sns.barplot(y=yield_per_crop['Item'], x=yield_per_crop['hg/ha_yield'])
plt.show()
col = ['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp', 'Area', 'Item', 'hg/ha_yield']
df = df[col]
X = df.drop('hg/ha_yield', axis=1)
y = df['hg/ha_yield']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True)
ohe = OneHotEncoder(drop='first')
scale = StandardScaler()
preprocesser = ColumnTransformer(
    transformers=[
        ('StandardScale', scale, [0, 1, 2, 3]),
        ('OneHotEncode', ohe, [4, 5])
    ], 
    remainder='passthrough'
) 
X_train_dummy = preprocesser.fit_transform(X_train)
X_test_dummy = preprocesser.transform(X_test)  
models = {
    'Linear Regression': LinearRegression(),
    'Lasso': Lasso(),
    'Ridge': Ridge(),
    'Decision Tree': DecisionTreeRegressor(),
    'KNN': KNeighborsRegressor(),
}
for name, model in models.items():
    model.fit(X_train_dummy, y_train)
    y_pred = model.predict(X_test_dummy)
    print(f"{name}: MAE: {mean_absolute_error(y_test, y_pred)}, R2 Score: {r2_score(y_test, y_pred)}")
dtr = DecisionTreeRegressor()
dtr.fit(X_train_dummy, y_train)
def prediction(Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item):
    features = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]], dtype=object)
    transform_features = preprocesser.transform(features)
    predicted_yield = dtr.predict(transform_features).reshape(-1, 1)
    return predicted_yield[0][0]
result = prediction(1990, 1485.0, 121.0, 16.37, 'Albania', 'Maize')
print(result)
pickle.dump(dtr, open("dtr.pkl", "wb"))
pickle.dump(preprocesser, open("preprocesser.pkl", "wb"))
