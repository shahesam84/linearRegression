# importing libraries
import seaborn as sns
import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
from sklearn.impute import SimpleImputer

# importing files
gdp_per_capita = pd.read_csv("CSV File", thousands=',', encoding='latin1', na_values="n/a")
oecd_bli = pd.read_csv("CSV File", thousands=',')

# dropping NaN rows
gdp_per_capita2 = gdp_per_capita[(gdp_per_capita['INEQUALITY']=='TOT')]
country_stats = pd.merge(gdp_per_capita2[["Country","Value"]], oecd_bli[["Country","2015"]], how = 'left', on='Country')
country_stats['2015'] = country_stats['2015'].apply (pd.to_numeric, errors='coerce')
country_stats = country_stats.dropna()

# defining input and outputs
X = np.c_[country_stats['2015']]
y = np.c_[country_stats['Value']]

# visualization
sns.scatterplot(data=country_stats, x='2015', y='Value')

lin_reg_model = linear_model.LinearRegression()

# training the model
lin_reg_model.fit(X, y)

X_new = [[22587]]

# making predictions
print(lin_reg_model.predict(X_new))
