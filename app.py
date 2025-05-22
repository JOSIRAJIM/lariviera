import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

from procesamiento import cargar_y_preprocesar_datos
from actualizador_modelo import entrenar_o_cargar_modelo

st.set_page_config(page_title="Dashboard de Ventas", layout="wide")
st.title("📊 Dashboard de Predicción de Ventas")

archivo = st.file_uploader("📁 Sube tu archivo Excel", type=["xlsx"])
reentrenar = st.checkbox("¿Deseas reentrenar el modelo con los nuevos datos?")

if archivo:
    df, encoders = cargar_y_preprocesar_datos(archivo)
    X = df.drop(['CANTIDAD', 'TOTAL_LINEA', 'FACTURA', 'UNIDADES'], axis=1, errors='ignore')
    y = df['CANTIDAD']
    modelo = entrenar_o_cargar_modelo(X, y, modelo_path="modelo_venta.pkl", reentrenar=reentrenar)
    predicciones = modelo.predict(X)

    rmse = np.sqrt(mean_squared_error(y, predicciones))
    r2 = r2_score(y, predicciones)
    cv = cross_val_score(modelo, X, y, cv=5, scoring='r2')

    st.metric("📉 RMSE", f"{rmse:.2f}")
    st.metric("📈 R²", f"{r2:.2f}")
    st.write(f"**R² Cross-Validation:** {cv.mean():.2f} ± {cv.std():.2f}")

    df['PREDICCIONES'] = predicciones
    original = pd.read_excel(archivo)
    df['NOMBRETIENDA'] = original['NOMBRETIENDA']
    df['NOMBREARTICULO_VENTA'] = original['NOMBREARTICULO_VENTA']

    # -----------------------------
    # 🎛️ FILTROS EN LA BARRA LATERAL
    # -----------------------------
    st.sidebar.header("🔍 Filtros")

    tiendas = sorted(df['NOMBRETIENDA'].dropna().unique())
    articulos = sorted(df['NOMBREARTICULO_VENTA'].dropna().unique())

    tienda_sel = st.sidebar.selectbox("🏪 Selecciona Tienda", ['Todas'] + tiendas)
    articulo_sel = st.sidebar.selectbox("🧾 Selecciona Artículo", ['Todos'] + articulos)

    df_filtrado = df.copy()
    if tienda_sel != 'Todas':
        df_filtrado = df_filtrado[df_filtrado['NOMBRETIENDA'] == tienda_sel]
    if articulo_sel != 'Todos':
        df_filtrado = df_filtrado[df_filtrado['NOMBREARTICULO_VENTA'] == articulo_sel]

    # -----------------------------
    # 📦 DISTRIBUCIÓN ÓPTIMA
    # -----------------------------
    distribucion = df_filtrado.groupby(['NOMBRETIENDA', 'NOMBREARTICULO_VENTA'])['PREDICCIONES'].sum().reset_index()
    distribucion.sort_values(by='PREDICCIONES', ascending=False, inplace=True)

    st.subheader("📦 Distribución óptima sugerida")
    st.dataframe(distribucion.head(10))

    # -----------------------------
    # 📈 COMPARACIÓN DE VALORES
    # -----------------------------
    st.subheader("📈 Comparación entre valores reales y predicciones")
    fig, ax = plt.subplots()
    ax.scatter(range(len(df_filtrado)), df_filtrado['CANTIDAD'], alpha=0.5, label="Reales", color="blue")
    ax.scatter(range(len(df_filtrado)), df_filtrado['PREDICCIONES'], alpha=0.5, label="Predichos", color="red")
    ax.set_title("Predicción vs Real")
    ax.legend()
    st.pyplot(fig)

    # -----------------------------
    # 🧠 IMPORTANCIA DE VARIABLES
    # -----------------------------
    import xgboost as xgb
    fig2, ax2 = plt.subplots()
    xgb.plot_importance(modelo, ax=ax2, max_num_features=10)
    st.pyplot(fig2)

    # -----------------------------
    # ⬇️ DESCARGA DE RESULTADO
    # -----------------------------
    distribucion.to_csv("distribucion_optima.csv", index=False)
    with open("distribucion_optima.csv", "rb") as f:
        st.download_button("⬇️ Descargar CSV de Distribución", f, file_name="distribucion_optima.csv")
