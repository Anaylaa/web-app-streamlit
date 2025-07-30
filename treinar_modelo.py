# modelo_titanic.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from joblib import dump

# Carrega dados
df = pd.read_csv("titanic.csv")

# Seleciona colunas relevantes
df = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Survived"]]
df.dropna(subset=["Sex"], inplace=True)

# Codifica sexo
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

# Trata valores ausentes
imputer = SimpleImputer(strategy="mean")
X = df.drop("Survived", axis=1)
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
y = df["Survived"]

# Divide os dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treina o modelo
modelo = RandomForestClassifier()
modelo.fit(X_train, y_train)

# Salva o modelo e o imputer
dump(modelo, "modelo_titanic.joblib")
dump(imputer, "imputer_titanic.joblib")
