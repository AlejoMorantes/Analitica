import streamlit as st
import pandas as pd
import joblib
from PIL import Image

# ======================================================
# 🔹 CONFIGURACIÓN GENERAL
# ======================================================
st.set_page_config(page_title="Predicción y Clientes", layout="centered")

# ======================================================
# 🔹 CARGA DE ARCHIVOS (Excel + Joblib)
# ======================================================

@st.cache_data
def load_excels():
    try:
        df_rfm = pd.read_excel("resultado_rfm.xlsx")
    except:
        df_rfm = pd.DataFrame()

    try:
        df_perfil = pd.read_excel("perfil_clusters_rfm.xlsx")
    except:
        df_perfil = pd.DataFrame()

    return df_rfm, df_perfil


@st.cache_resource
def load_joblib():
    try:
        data_joblib = joblib.load("MachineLearning.joblib")
        model = data_joblib.get("modelo")
        m = data_joblib.get("m")
        b = data_joblib.get("b")
        df_model = data_joblib.get("data")
        return model, m, b, df_model
    except:
        return None, None, None, None


df_rfm, df_perfil = load_excels()
model, m, b, df_model = load_joblib()

# ======================================================
# 🔹 FUNCIONES (equivalentes a los endpoints de la API)
# ======================================================

def predict_value(x):
    """Simula el endpoint /predict"""
    if m is None or b is None:
        return None
    return m * x + b


def listar_clientes():
    if df_rfm.empty:
        return []
    return df_rfm["Cliente"].dropna().unique().tolist()


def obtener_cliente(nombre):
    if df_rfm.empty:
        return {}
    data = df_rfm[df_rfm["Cliente"].astype(str).str.lower() == nombre.lower()]
    if data.empty:
        return {}
    return data.to_dict(orient="records")[0]


def listar_departamentos():
    if df_rfm.empty:
        return []
    return df_rfm["Departamento"].dropna().unique().tolist()


def clientes_por_departamento(dep):
    if df_rfm.empty:
        return []
    data = df_rfm[df_rfm["Departamento"].astype(str).str.lower() == dep.lower()]
    return data["Cliente"].dropna().unique().tolist()


def listar_clusters():
    if df_rfm.empty:
        return []
    return df_rfm["Cluster_RFM"].dropna().unique().tolist()


def clientes_por_cluster(cluster):
    if df_rfm.empty:
        return []
    data = df_rfm[df_rfm["Cluster_RFM"].astype(str) == str(cluster)]
    clientes = data["Cliente"].dropna().unique().tolist()
    return clientes, data.head(5).to_dict(orient="records")


def perfil_clusters_preview():
    if df_perfil.empty:
        return []
    return df_perfil.head(5).to_dict(orient="records")


def listar_meses_favoritos():
    if df_rfm.empty:
        return []
    return df_rfm["mes_favorito"].dropna().unique().tolist()


def clientes_por_mes(mes):
    if df_rfm.empty:
        return []
    mes_str = str(mes).strip().lower()
    df_rfm["mes_favorito"] = df_rfm["mes_favorito"].astype(str)
    data = df_rfm[df_rfm["mes_favorito"].str.lower() == mes_str]
    if data.empty:
        return []
    columnas = ["Cliente", "recency", "frequency"]
    columnas_existentes = [c for c in columnas if c in data.columns]
    return data[columnas_existentes]



# ======================================================
# 🔹 UI — INTERFAZ PRINCIPAL STREAMLIT
# ======================================================

st.title("📊 Panel de Predicción y Análisis de Clientes")

st.markdown("---")

# ======================================================
# 🔹 SECCIÓN 1 — Predicción Lineal
# ======================================================

st.header("📈 Predicción Lineal")

x_input = st.number_input("Meses a predecir desde Julio 2025", step=1.0)

if st.button("Predecir"):
    y = predict_value(x_input + 31)
    if y is not None:
        st.success(f"Resultado: y = {y:.4f}")
    else:
        st.error("No se pudo cargar el modelo.")

st.markdown("---")

# ======================================================
# 🔹 SECCIÓN 2 — Consulta de Clientes
# ======================================================

st.header("🧭 Consulta de Clientes")

clientes = listar_clientes()

if clientes:
    cliente = st.selectbox("Selecciona un cliente", clientes)
    if st.button("Ver información del cliente"):
        info = obtener_cliente(cliente)
        if info:
            for k, v in info.items():
                st.write(f"**{k}:** {v}")
        else:
            st.warning("Cliente no encontrado.")
else:
    st.warning("No hay clientes disponibles.")

st.markdown("---")

# ======================================================
# 🔹 FILTRO POR DEPARTAMENTO
# ======================================================

st.subheader("🏙️ Clientes por Departamento")

deps = listar_departamentos()

if deps:
    dep = st.selectbox("Departamento", deps)
    if st.button("Mostrar clientes por departamento"):
        st.write(clientes_por_departamento(dep))
else:
    st.warning("No se pudieron cargar los departamentos.")

st.markdown("---")

# ======================================================
# 🔹 PERFIL DE CLUSTERS
# ======================================================

st.subheader("🧩 Perfil de Clusters")

perfil = perfil_clusters_preview()

if perfil:
    st.dataframe(perfil)
else:
    st.info("No hay datos disponibles.")

clusters = listar_clusters()

if clusters:
    cluster_sel = st.selectbox("Cluster RFM", clusters)
    if st.button("Mostrar clientes del cluster"):
        clientes_c, preview = clientes_por_cluster(cluster_sel)
        st.write("Clientes:", clientes_c)
        st.write("Vista previa:")
        st.dataframe(preview)

st.markdown("---")

# ======================================================
# 🔹 FILTRO POR MES FAVORITO
# ======================================================

st.subheader("📅 Clientes por Mes Favorito")

meses = listar_meses_favoritos()

if meses:
    mes_sel = st.selectbox("Mes favorito", meses)
    if st.button("Mostrar clientes del mes"):
        st.dataframe(clientes_por_mes(mes_sel))
else:
    st.warning("No hay meses favoritos registrados.")

st.markdown("---")




#python -m streamlit run streamlit_app.py
