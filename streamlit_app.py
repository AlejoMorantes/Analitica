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
    data = df_rfm[df_rfm["mes_favorito"].astype(str).str.lower() == mes.lower()]
    if data.empty:
        return []
    columnas = ["Cliente", "recency", "frequency"]
    columnas = [c for c in columnas if c in data.columns]
    return data[columnas].to_dict(orient="records")


# ======================================================
# 🔹 UI — INTERFAZ PRINCIPAL STREAMLIT
# ======================================================

st.title(" Predicción y Análisis de Clientes")

st.markdown("---")

# ======================================================
# 🔹 SECCIÓN 1 — Predicción Lineal
# ======================================================

# st.header("📈 Predicción Lineal")

# x_input = st.number_input("Meses a predecir desde Julio 2025", step=1.0)

# if st.button("Predecir"):
#     y = predict_value(x_input + 31)
#     if y is not None:
#         st.success(f"Resultado: y = {y:.4f}")
#     else:
#         st.error("No se pudo cargar el modelo.")

# st.markdown("---")

# ======================================================
# 🔹 SECCIÓN — Gráfico de Regresión + Predicción
# ======================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

st.header("📊 Gráfico del Modelo + Predicción")

try:
    # === Cargar el archivo Excel original ===
    df = pd.read_excel("MachineLearning.xlsx")

    # Asegurar datetime
    df['Fecha'] = pd.to_datetime(df['Fecha'])

    # Indexar y resamplear mensual
    df_indexed = df.set_index('Fecha')
    df = df_indexed['Vlr Total'].resample('ME').sum().to_frame()
    df.index.name = 'Fecha'

    # === LIMPIEZA DE OUTLIERS POR IQR ===
    Q1 = df['Vlr Total'].quantile(0.25)
    Q3 = df['Vlr Total'].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df_replaced = df.copy()

    for i in range(len(df)):
        valor = df.iloc[i, 0]

        # Si es outlier ⇒ reemplazar por promedio de vecinos
        if valor < lower_bound or valor > upper_bound:
            vecinos_idx = [j for j in range(i-3, i+4) if j != i and 0 <= j < len(df)]
            promedio_vecinos = df.iloc[vecinos_idx, 0].mean()
            df_replaced.iloc[i, 0] = promedio_vecinos

    df = df_replaced.copy()

except Exception as e:
    st.error(f"No se pudo cargar MachineLearning.xlsx: {e}")
    df = None

# Si falló la carga, no continuar
if df is None or df.empty:
    st.stop()

# Asegurar modelo cargado
if m is None or b is None:
    st.error("El modelo dentro de MachineLearning.joblib no se pudo cargar.")
    st.stop()

# ======================================================
# 🔹 Construcción del gráfico
# ======================================================

# Crear variable tiempo
df["Tiempo"] = np.arange(len(df))

# Input de mes futuro
meses_futuros = st.number_input("Meses a predecir hacia adelante:", min_value=1, value=6)

# Punto X futuro
x_future = df["Tiempo"].max() + meses_futuros
y_future = m * x_future + b

# Fecha futura equivalente
ultima_fecha = df.index[-1]
fecha_future = ultima_fecha + pd.DateOffset(months=meses_futuros)

# Extender valores del modelo
tiempo_extendido = np.append(df["Tiempo"].values, x_future)
y_extendido = m * tiempo_extendido + b

# Fechas extendidas
fechas_ext = list(df.index) + [fecha_future]

# ============================
# GRAFICAR
# ============================

fig, ax = plt.subplots(figsize=(14, 5))

# Datos reales
ax.plot(df.index, df["Vlr Total"], marker="o", linewidth=2, label="Datos reales")

# Línea extendida
ax.plot(fechas_ext, y_extendido, linestyle="--", linewidth=2, color="orange",
        label="Regresión lineal extendida")

# Punto futuro
ax.scatter([fecha_future], [y_future], color="red", s=150,
           label=f"Predicción ({fecha_future.date()})")

ax.set_xlabel("Fecha")
ax.set_ylabel("Valor Total")
ax.set_title("Datos Originales + Modelo Extendido hasta la Predicción")

plt.xticks(rotation=45)
ax.legend()
plt.tight_layout()

st.pyplot(fig)

# Mostrar valor predicho
st.success(f"📌 Predicción para {fecha_future.date()}: **{y_future:,.0f}**")



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