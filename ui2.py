import streamlit as st
import requests
from PIL import Image

# ======================================================
# 🔹 CONFIGURACIÓN GENERAL
# ======================================================

st.set_page_config(page_title="Predicción y Clientes", layout="centered")

# 🔹 Puerto unificado para la API Flask
API_URL = "http://127.0.0.1:5001"

# ======================================================
# 🔹 SECCIÓN 1 — Predicción Lineal
# ======================================================

st.title("📈 Predicción Lineal:")
st.write("Introduce un valor de **x** y la app te devolverá el valor predicho de **y** usando la API Flask.")

x_input = st.number_input("Cuantos Meses Quieres Predecir DESDE Julio 2025", step=1.0, format="%.1f")

if st.button("Predecir"):
    try:
        response = requests.post(f"{API_URL}/predict", json={'x': x_input+31}) # DEBE DE HABER 31 MESES PEUSTO QUE ESE ES EL NÚMERO INICIAL JULIO 2025
        if response.status_code == 200:
            result = response.json()
            if 'y_pred' in result:
                st.success(f"✅ Resultado: y = {result['y_pred']:.4f}")
            else:
                st.error(f"Error en la respuesta: {result}")
        else:
            st.error(f"❌ Error al conectar con la API ({response.status_code})")
    except Exception as e:
        st.error(f"⚠️ No se pudo conectar con la API de predicción.\n\n{e}")

st.markdown("---")


# ======================================================
# 🔹 MOSTRAR IMAGEN SERVIDA POR LA API
# ======================================================


# st.subheader("Mapa de Clusters")

# try:
#     image_url = f"{API_URL}/static/descargar.png"  # o "/imagen_descargar"
#     st.image(image_url, caption="Imagen desde la API", width="stretch")
# except Exception as e:
#     st.error(f"No se pudo cargar la imagen: {e}")




# ======================================================
# 🔹 SECCIÓN 2 — Consulta de Clientes y Clusters
# ======================================================

st.title("🧭 Consulta de Clientes y Clusters")

# --- Obtener lista de clientes ---
try:
    response = requests.get(f"{API_URL}/clientes")
    clientes = response.json().get("clientes", []) if response.status_code == 200 else []
except Exception as e:
    clientes = []
    st.error(f"No se pudo conectar con la API: {e}")

if clientes:
    cliente = st.selectbox("Selecciona un cliente", clientes)
    if st.button("Ver información del cliente"):
        try:
            response = requests.get(f"{API_URL}/cliente/{cliente}")
            if response.status_code == 200:
                cliente_info = response.json()
                st.write("### Información del cliente")
                for k, v in cliente_info.items():
                    st.markdown(f"**{k}:** {v}")
            else:
                st.warning("Cliente no encontrado.")
        except Exception as e:
            st.error(f"Error de conexión: {e}")
else:
    st.warning("No hay clientes disponibles o la API no está activa.")

st.markdown("---")

# --- FILTRO POR DEPARTAMENTO ---
st.subheader("🏙️ Filtrar por Departamento")

try:
    response = requests.get(f"{API_URL}/departamentos")
    departamentos = response.json().get("departamentos", []) if response.status_code == 200 else []
except Exception as e:
    departamentos = []
    st.error(f"No se pudo conectar con la API: {e}")

if departamentos:
    departamento = st.selectbox("Selecciona un departamento", departamentos)
    if st.button("Mostrar clientes del departamento"):
        try:
            response = requests.get(f"{API_URL}/clientes_por_departamento/{departamento}")
            data = response.json().get("clientes", [])
            st.write(data if data else "No hay clientes en este departamento.")
        except Exception as e:
            st.error(f"Error de conexión: {e}")
else:
    st.warning("No se pudieron obtener los departamentos.")

st.markdown("---")

# --- PERFIL DE CLUSTERS ---
st.subheader("🧩 Vista previa de Clusters y Perfiles")

try:
    response = requests.get(f"{API_URL}/perfil_clusters")
    perfil_data = response.json().get("perfil", []) if response.status_code == 200 else []
    if perfil_data:
        st.dataframe(perfil_data)
    else:
        st.info("No se encontraron datos en perfil_clusters_rfm.xlsx.")
except Exception as e:
    st.error(f"Error de conexión con la API: {e}")

# --- CLUSTERS RFM ---
try:
    response = requests.get(f"{API_URL}/clusters")
    clusters = response.json().get("clusters", []) if response.status_code == 200 else []
except Exception as e:
    clusters = []
    st.error(f"No se pudo conectar con la API: {e}")

if clusters:
    cluster_sel = st.selectbox("Selecciona un Cluster RFM", clusters)
    if st.button("Mostrar clientes del cluster"):
        try:
            response = requests.get(f"{API_URL}/clientes_por_cluster/{cluster_sel}")
            data = response.json().get("clientes", [])
            st.dataframe(data if data else [])
        except Exception as e:
            st.error(f"Error de conexión: {e}")

st.markdown("---")

# --- FILTRO POR MES FAVORITO ---
st.subheader("📅 Filtrar por Mes Favorito")

try:
    response = requests.get(f"{API_URL}/meses_favoritos")
    meses = response.json().get("meses", []) if response.status_code == 200 else []
except Exception as e:
    meses = []
    st.error(f"No se pudo conectar con la API: {e}")

if meses:
    mes_sel = st.selectbox("Selecciona un mes favorito", meses)
    if st.button("Mostrar clientes del mes"):
        try:
            response = requests.get(f"{API_URL}/clientes_por_mes/{mes_sel}")
            data = response.json().get("datos", [])
            st.dataframe(data if data else [])
        except Exception as e:
            st.error(f"Error de conexión: {e}")
else:
    st.warning("No se pudieron obtener los meses favoritos.")

st.markdown("---")




# Para ejecutar:
# python -m streamlit run ui2.py
