import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import load

# Configuração da página
st.set_page_config(page_title="Titanic - AED + Previsão", layout="wide")

# Estilo CSS
st.markdown("""
    <style>
    .stApp {
        background-color: #ecf2f9;
        font-family: 'Arial', sans-serif;
    }
    h1, h2, h3 {
        color: #1e3d59;
    }
    </style>
""", unsafe_allow_html=True)

# Carrega dados
@st.cache_data
def carregar_dados():
    df = pd.read_csv("titanic.csv")
    return df

df = carregar_dados()

# Mapeamentos
df.rename(columns={
    "Name": "Nome", "Sex": "Sexo", "Age": "Idade",
    "Survived": "Sobreviveu", "Pclass": "Classe",
    "SibSp": "Irmãos/Cônjuge", "Parch": "Pais/Filhos",
    "Fare": "Tarifa", "Embarked": "Embarque"
}, inplace=True)

df["Sexo"] = df["Sexo"].map({"male": "Masculino", "female": "Feminino"})
df["Sobreviveu"] = df["Sobreviveu"].map({0: "Não", 1: "Sim"})
df["Classe"] = df["Classe"].map({1: "Primeira", 2: "Segunda", 3: "Terceira"})

# Menu lateral
menu = st.sidebar.radio("Navegação", ["📊 Análise Exploratória", "🔍 Previsão com IA"])

# Página 1 - AED
if menu == "📊 Análise Exploratória":
    st.title("🚢 Análise Exploratória - Titanic")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total de passageiros", df.shape[0])
    with col2:
        st.metric("Sobreviventes", df["Sobreviveu"].value_counts()["Sim"])

    # Gráfico de sobrevivência
    st.subheader("📈 Distribuição de Sobreviventes")
    fig1 = px.pie(df, names="Sobreviveu", title="Proporção de Sobrevivência", color_discrete_sequence=px.colors.sequential.RdBu)
    st.plotly_chart(fig1, use_container_width=True)

    # Histograma da idade
    st.subheader("📊 Distribuição da Idade")
    fig2, ax = plt.subplots()
    sns.histplot(df["Idade"].dropna(), bins=30, kde=True, ax=ax, color="#2e86c1")
    st.pyplot(fig2)

    # Boxplot das tarifas por classe
    st.subheader("💵 Tarifas por Classe")
    fig3, ax3 = plt.subplots()
    sns.boxplot(x="Classe", y="Tarifa", data=df, ax=ax3, palette="Set2")
    ax3.set_ylim(0, 200)
    st.pyplot(fig3)

# Página 2 - Previsão
elif menu == "🔍 Previsão com IA":
    st.title("🔍 Prever Sobrevivência")

    st.write("Insira os dados abaixo para prever se uma pessoa sobreviveria:")

    sexo = st.selectbox("Sexo", ["Masculino", "Feminino"])
    idade = st.slider("Idade", 0, 100, 30)
    classe = st.selectbox("Classe", ["Primeira", "Segunda", "Terceira"])
    tarifa = st.number_input("Tarifa paga (em libras)", min_value=0.0, max_value=600.0, value=50.0)
    irmaos = st.number_input("Nº de irmãos/cônjuges a bordo", min_value=0, max_value=10, value=0)
    pais = st.number_input("Nº de pais/filhos a bordo", min_value=0, max_value=10, value=0)

    if st.button("🔮 Prever"):
        # Codifica entrada
        sexo_bin = 0 if sexo == "Masculino" else 1
        classe_map = {"Primeira": 1, "Segunda": 2, "Terceira": 3}
        classe_val = classe_map[classe]

        dados_input = pd.DataFrame([{
            "Pclass": classe_val,
            "Sex": sexo_bin,
            "Age": idade,
            "SibSp": irmaos,
            "Parch": pais,
            "Fare": tarifa
        }])

        modelo = load("modelo_titanic.joblib")
        imputer = load("imputer_titanic.joblib")
        dados_input = pd.DataFrame(imputer.transform(dados_input), columns=dados_input.columns)

        resultado = modelo.predict(dados_input)[0]
        prob = modelo.predict_proba(dados_input)[0][1]

        if resultado == 1:
            st.success(f"✅ Provavelmente Sobreviveu (confiança: {prob:.2%})")
        else:
            st.error(f"❌ Provavelmente Não Sobreviveu (confiança: {1 - prob:.2%})")
