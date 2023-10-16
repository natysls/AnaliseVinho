import pandas as pd
import filter
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

data = pd.read_csv('wine.csv')
data.replace("?", pd.NA, inplace=True)

def qui_quadrado(df):
    df = filter.toda_filtragem(df)

    X = df.iloc[:, :-1]  # Atributos
    y = df.iloc[:, -1]   # Variável de classe

    # Pedi os 4 melhores atributos
    selector = SelectKBest(score_func = chi2, k = 4)
    X_new = selector.fit_transform(X, y)

    colunas_selecionadas = X.columns[selector.get_support()]
    print("Colunas selecionadas:", colunas_selecionadas)

qui_quadrado(data)