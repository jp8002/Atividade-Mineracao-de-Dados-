from pandas.core.frame import DataFrame
import pandas as pd
import numpy as np 
import matplotlib as plt 
import matplotlib.pyplot as plty
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error,root_mean_squared_error # Avaliar perfance do model
from sklearn.preprocessing import LabelEncoder # Encoda dados categóricos como valores numéricos

# Pergunta: Qual seria a porcentagem de desconto na próxima promoção?
# Escolhemos a Floresta Aleatória (Random Forest) por ser um conjunto de várias árvores de decisão,que captura relações não lineares e interações entre variáveis (como idade do jogo, preço base e avaliações) melhor que modelos lineares simples. Por agregar múltiplas árvores, ela é mais resistente a overfitting do que uma árvore isolada e lida bem com valores atípicos nas features, mantendo uma estimativa consistente da faixa de desconto sem depender do preço final promocional.

warnings.filterwarnings("ignore")

df = pd.read_csv("coleta_steam_final_preparada.csv")

encoders = {}
for col in df.select_dtypes(include=["object"]).columns:
    label_encoder = LabelEncoder()
    df[col] = label_encoder.fit_transform(df[col])
    encoders[col] = label_encoder
    
df.info()


X =  df.drop(columns=["app_id","desconto_pct","titulo","preco_final_brl","url_origem","data_coleta","windows","mac","linux","categoria_preco","data_lancamento_dt","data_lancamento"])
y = df["desconto_pct"]

y = y.fillna(y.median())

# random_state=42, garante que a semente randomica continue a mesma em todos os treinos garantindo que possa ser reproduzido
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# n_estimators=100: Número de árvores de decisão na floresta
regressor = RandomForestRegressor(n_estimators=100, random_state=42, oob_score=True)


regressor.fit(X_train, y_train)

print("Out-of-Bag Score:", regressor.oob_score_)

y_pred = regressor.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

rmse = root_mean_squared_error(y_test, y_pred)
print("Root Mean Squared Error:", rmse)

r2 = r2_score(y_test, y_pred)
print("R-squared:", r2)

importances = pd.Series(regressor.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature Importances:")
print(importances)




