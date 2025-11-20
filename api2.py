from flask import Flask, jsonify, request
import pandas as pd
import joblib

# ============================================================
# === CREAR LA APP ===========================================
# ============================================================
app = Flask(__name__) #, static_url_path='/static', static_folder='static'


# ============================================================
# === SECCIÓN 1: CARGAR ARCHIVOS EXCEL (CLIENTES & CLUSTERS) =
# ============================================================

DATA_PATH_RFM = 'resultado_rfm.xlsx'
DATA_PATH_PERFIL = 'perfil_clusters_rfm.xlsx'

try:
    df_rfm = pd.read_excel(DATA_PATH_RFM)
    print(f"✅ Archivo '{DATA_PATH_RFM}' cargado correctamente con {len(df_rfm)} registros.")
except Exception as e:
    print(f"❌ Error al cargar '{DATA_PATH_RFM}': {e}")
    df_rfm = pd.DataFrame()

try:
    df_perfil = pd.read_excel(DATA_PATH_PERFIL)
    print(f"✅ Archivo '{DATA_PATH_PERFIL}' cargado correctamente con {len(df_perfil)} registros.")
except Exception as e:
    print(f"❌ Error al cargar '{DATA_PATH_PERFIL}': {e}")
    df_perfil = pd.DataFrame()

# ============================================================
# === SECCIÓN 2: CARGAR MODELO JOBLIB ========================
# ============================================================

try:
    data_joblib = joblib.load('MachineLearning.joblib')
    model = data_joblib.get('modelo')
    m = data_joblib.get('m')
    b = data_joblib.get('b')
    df_model = data_joblib.get('data')
    print("✅ Archivo 'MachineLearning.joblib' cargado correctamente.")
except Exception as e:
    print("❌ Error al cargar el archivo MachineLearning.joblib:", e)
    data_joblib, model, m, b, df_model = None, None, None, None, None


# ============================================================
# === SECCIÓN 3: ENDPOINT PRINCIPAL ==========================
# ============================================================

@app.route('/')
def home():
    return jsonify({
        "status": "✅ API combinada funcionando correctamente",
        "endpoints_disponibles": [
            "/info",
            "/predict",
            "/clientes",
            "/cliente/<nombre_cliente>",
            "/departamentos",
            "/clientes_por_departamento/<departamento>",
            "/clusters",
            "/clientes_por_cluster/<cluster>",
            "/perfil_clusters",
            "/meses_favoritos",
            "/clientes_por_mes/<mes>"
        ]
    })


# ============================================================
# === SECCIÓN 4: ENDPOINTS MODELO LINEAL =====================
# ============================================================

@app.route('/info', methods=['GET'])
def info():
    if df_model is not None:
        return jsonify({
            'filas': len(df_model),
            'm': m,
            'b': b
        })
    else:
        return jsonify({'error': 'No se pudo cargar el archivo MachineLearning.joblib.'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        x = float(data['x'])
        y_pred = m * x + b
        return jsonify({'x': x, 'y_pred': y_pred})
    except Exception as e:
        return jsonify({'error': str(e)})


# ============================================================
# === SECCIÓN 5: ENDPOINTS DE CLIENTES Y CLUSTERS ============
# ============================================================

@app.route('/clientes', methods=['GET'])
def listar_clientes():
    try:
        clientes = df_rfm['Cliente'].dropna().unique().tolist()
        return jsonify({'clientes': clientes})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/cliente/<string:nombre_cliente>', methods=['GET'])
def obtener_cliente(nombre_cliente):
    try:
        cliente_data = df_rfm[df_rfm['Cliente'].astype(str).str.lower() == nombre_cliente.lower()]
        if cliente_data.empty:
            return jsonify({'error': f"No se encontró el cliente '{nombre_cliente}'"}), 404
        return jsonify(cliente_data.to_dict(orient='records')[0])
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/departamentos', methods=['GET'])
def listar_departamentos():
    try:
        departamentos = df_rfm['Departamento'].dropna().unique().tolist()
        return jsonify({'departamentos': departamentos})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/clientes_por_departamento/<string:departamento>', methods=['GET'])
def clientes_por_departamento(departamento):
    try:
        clientes = (
            df_rfm[df_rfm['Departamento'].astype(str).str.lower() == departamento.lower()]
            ['Cliente'].dropna().unique().tolist()
        )
        return jsonify({'clientes': clientes})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/clusters', methods=['GET'])
def listar_clusters():
    try:
        clusters = df_rfm['Cluster_RFM'].dropna().unique().tolist()
        return jsonify({'clusters': clusters})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/clientes_por_cluster/<cluster>', methods=['GET'])
def clientes_por_cluster(cluster):
    try:
        cluster_str = str(cluster).strip()
        filtrado = df_rfm[df_rfm['Cluster_RFM'].astype(str) == cluster_str]

        if filtrado.empty:
            return jsonify({'clientes': []})

        clientes_cluster = filtrado['Cliente'].dropna().unique().tolist()
        preview = filtrado.head(5).to_dict(orient='records')

        return jsonify({
            'cluster': cluster_str,
            'clientes': clientes_cluster,
            'preview': preview
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/perfil_clusters', methods=['GET'])
def mostrar_perfil_clusters():
    try:
        perfil_preview = df_perfil.head(5).to_dict(orient='records')
        return jsonify({'perfil': perfil_preview})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/clientes_por_mes/<string:mes>', methods=['GET'])
def clientes_por_mes(mes):
    try:
        mes_str = str(mes).strip().lower()
        filtrado = df_rfm[df_rfm['mes_favorito'].astype(str).str.lower() == mes_str]

        if filtrado.empty:
            return jsonify({'datos': []})

        columnas = ['Cliente', 'recency', 'frequency']                                   # filtrar por mes favorito variables
        columnas_existentes = [c for c in columnas if c in filtrado.columns]

        resultado = filtrado[columnas_existentes].to_dict(orient='records')
        return jsonify({'datos': resultado})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/meses_favoritos', methods=['GET'])
def listar_meses_favoritos():
    try:
        meses = df_rfm['mes_favorito'].dropna().unique().tolist()
        return jsonify({'meses': meses})
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/imagen_descargar')
def imagen_descargar():
    return app.send_static_file('descargar.png')


# ============================================================
# === EJECUCIÓN DE LA API ====================================
# ============================================================

if __name__ == '__main__':
    app.run(debug=True, port=5001)
