import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("insurance.csv")
data.head()

data.describe()

# graficos

plt.scatter(data["age"], data["charges"])
plt.xlabel("Age")
plt.ylabel("Charges")
plt.title("Age vs Charges")
plt.show()

plt.scatter(data["bmi"], data["charges"])
plt.xlabel("bmi")
plt.ylabel("Charges")
plt.title("bmi vs Charges")
plt.show()

plt.scatter(data["children"], data["charges"])
plt.xlabel("children")
plt.ylabel("Charges")
plt.title("children vs Charges")
plt.show()

x = data.drop("charges", axis=1)
y = data["charges"]

age_bins = pd.cut(x["age"], bins=[0, 30,45,60,100], labels=False)

# definindo quais são as colunas não numéricas e numéricas
string_cols = ['sex', 'smoker', 'region']
num_columns = ['age', 'bmi', 'children']

preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(), string_cols),
    ('num', StandardScaler(), num_columns)
], remainder='passthrough')

# Apenas para visualizar como nosso dataset ficara
data_tranformed = preprocessor.fit_transform(x)

encoded_col_names = preprocessor.named_transformers_['cat'].get_feature_names_out(string_cols)

df_transformed = pd.DataFrame(data_tranformed, columns=list(encoded_col_names) + ['age', 'bmi', 'children'])
df_transformed["charges"] = y
df_transformed.head()

# Correlação
data.drop(columns=string_cols).corr()

# Calcula a matriz de correlação
corr_matrix = df_transformed.corr()

# Plota o heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Matriz de Correlação")
plt.show()

# TODO: Fazer separação por idade
bins = [0, 30,45,60,100]
labels = ['0-30', '30-45', '45-60', '60+']


X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=528, stratify=age_bins)


train_bins = pd.cut(X_train['age'], bins=bins, labels=labels)
test_bins = pd.cut(X_test['age'], bins=bins, labels=labels)

print(train_bins.value_counts())
print(test_bins.value_counts())

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# passando pelo pipeline, treinando o modelo
pipeline_linear_regression = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

pipeline_linear_regression.fit(X_train, y_train)

# Fazendo previsão
import numpy as np
y_pred = pipeline_linear_regression.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("R2 Score:", r2)
print("RMSE:", rmse)

# Plota resíduos vs. valores preditos
plt.figure(figsize=(8, 5))
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.title("Resíduos vs. Valores Preditos")
plt.xlabel("Valores Preditos")
plt.ylabel("Resíduos")
plt.show()

# usando random forest regressor
# RandomizedSearchCV

from sklearn.ensemble import RandomForestRegressor

pipeline_random_forest_regressor = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=25))
])

# passando pelo pipeline, treinando o modelo
pipeline_random_forest_regressor.fit(X_train, y_train)

y_pred = pipeline_random_forest_regressor.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("R2 Score:", r2)
print("RMSE:", rmse)

# estimar um novo plano
new_plan_to_estimate = pd.DataFrame({
    'age': [40],
    'sex': ['female'],
    'bmi': [20],
    'children': [0],
    'smoker': ['yes'],
    'region': ['southwest']
})

y_price = pipeline_random_forest_regressor.predict(new_plan_to_estimate)

print("O preço estimado do plano é:", y_price[0])

# estimar um novo plano mulheres
new_plan_to_estimate = pd.DataFrame({
    'age': [40],
    'sex': ['female'],
    'bmi': [20],
    'children': [0],
    'smoker': ['no'],
    'region': ['southwest']
})

y_price = pipeline_random_forest_regressor.predict(new_plan_to_estimate)

print("O preço estimado do plano é:", y_price[0])

