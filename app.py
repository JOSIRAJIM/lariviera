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
    
    if 'FECHA' in original.columns:
        df['FECHA'] = pd.to_datetime(original['FECHA'])

    if 'UNIDADES' in original.columns:
        df['UNIDADES'] = original['UNIDADES']

    # 📍 FILTROS LATERALES
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

    # 📦 DISTRIBUCIÓN
    distribucion = df_filtrado.groupby(['NOMBRETIENDA', 'NOMBREARTICULO_VENTA'])['PREDICCIONES'].sum().reset_index()
    distribucion.sort_values(by='PREDICCIONES', ascending=False, inplace=True)

    st.subheader("📦 Distribución óptima sugerida")
    st.dataframe(distribucion.head(10))

    # 📈 COMPARACIÓN PREDICCIONES VS REALES
    st.subheader("📈 Comparación entre valores reales y predicciones")
    fig, ax = plt.subplots()
    ax.scatter(range(len(df_filtrado)), df_filtrado['CANTIDAD'], alpha=0.5, label="Reales", color="blue")
    ax.scatter(range(len(df_filtrado)), df_filtrado['PREDICCIONES'], alpha=0.5, label="Predichos", color="red")
    ax.set_title("Predicción vs Real")
    ax.legend()
    st.pyplot(fig)

    # 🔥 MAPA DE CALOR
    st.subheader("🔥 Mapa de calor de correlaciones")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    corr = df_filtrado.select_dtypes(include=[np.number]).corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax3)
    st.pyplot(fig3)

    # 📊 GRÁFICA DE BARRAS
    st.subheader("📊 Gráfica de barras por tienda")
    barras = df_filtrado.groupby('NOMBRETIENDA')['PREDICCIONES'].sum().sort_values(ascending=False)
    st.bar_chart(barras)

    # 📊 Comparación entre Distribución Manual y Predicción del Modelo
    st.subheader("📊 Comparación entre Distribución Manual (UNIDADES) y Predicción del Modelo")

    if 'UNIDADES' in df_filtrado.columns:
        comparacion = df_filtrado.groupby(['NOMBRETIENDA', 'NOMBREARTICULO_VENTA'])[['UNIDADES', 'PREDICCIONES']].sum().reset_index()
        comparacion['DIFERENCIA'] = comparacion['PREDICCIONES'] - comparacion['UNIDADES']
        comparacion.sort_values(by='DIFERENCIA', ascending=False, inplace=True)

        st.write("Top 10 diferencias (Predicción - Manual):")
        st.dataframe(comparacion.head(10))

        # Gráfico comparativo
        fig_comp, ax_comp = plt.subplots(figsize=(12, 6))
        top_diff = comparacion.head(10).set_index('NOMBREARTICULO_VENTA')
        top_diff[['UNIDADES', 'PREDICCIONES']].plot(kind='bar', ax=ax_comp)
        ax_comp.set_title("Top 10 Artículos: Unidades Manuales vs Predichas")
        ax_comp.set_ylabel("Cantidad")
        ax_comp.set_xlabel("Artículo")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig_comp)
    else:
        st.warning("La columna 'UNIDADES' no está disponible en los datos.")

    # 📉 GRÁFICA DE LÍNEAS (si hay fechas)
    if 'FECHA' in df_filtrado.columns:
        st.subheader("📆 Evolución temporal de predicción vs real")
        df_linea = df_filtrado.groupby('FECHA')[['CANTIDAD', 'PREDICCIONES']].sum().reset_index()
        df_linea.set_index('FECHA', inplace=True)
        st.line_chart(df_linea)

    # 🧠 IMPORTANCIA DE VARIABLES
    import xgboost as xgb
    st.subheader("🧠 Variables más importantes para el modelo")
    fig4, ax4 = plt.subplots()
    xgb.plot_importance(modelo, ax=ax4, max_num_features=10)
    st.pyplot(fig4)

    # ⬇️ DESCARGA
    distribucion.to_csv("distribucion_optima.csv", index=False)
    with open("distribucion_optima.csv", "rb") as f:
        st.download_button("⬇️ Descargar CSV de Distribución", f, file_name="distribucion_optima.csv")
