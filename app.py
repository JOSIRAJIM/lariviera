import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb  # Importación añadida
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from procesamiento import cargar_y_preprocesar_datos
from actualizador_modelo import entrenar_o_cargar_modelo

# Configuración inicial
st.set_page_config(page_title="Dashboard de Ventas", layout="wide")
st.title("📊 Dashboard de Predicción de Ventas")

# Carga de datos
archivo = st.file_uploader("📁 Sube tu archivo Excel", type=["xlsx"])
reentrenar = st.checkbox("¿Deseas reentrenar el modelo con los nuevos datos?")

if archivo:
    # Procesamiento de datos
    df, encoders = cargar_y_preprocesar_datos(archivo)
    X = df.drop(['CANTIDAD', 'TOTAL_LINEA', 'FACTURA', 'UNIDADES'], axis=1, errors='ignore')
    y = df['CANTIDAD']
    
    # Modelo
    modelo = entrenar_o_cargar_modelo(X, y, modelo_path="modelo_venta.pkl", reentrenar=reentrenar)
    
    # Predicciones
    predicciones = modelo.predict(X)
    predicciones = np.round(predicciones).astype(int)
    
    # Métricas
    rmse = np.sqrt(mean_squared_error(y, predicciones))
    r2 = r2_score(y, predicciones)
    cv = cross_val_score(modelo, X, y, cv=5, scoring='r2')
    
    # Mostrar métricas
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📉 RMSE", f"{rmse:.2f}")
    with col2:
        st.metric("📈 R²", f"{r2:.2f}")
    with col3:
        st.write(f"**R² Cross-Validation:** {cv.mean():.2f} ± {cv.std():.2f}")

    # Preparar datos para visualización
    df['PREDICCIONES'] = predicciones
    original = pd.read_excel(archivo)
    df['NOMBRETIENDA'] = original['NOMBRETIENDA']
    df['NOMBREARTICULO_VENTA'] = original['NOMBREARTICULO_VENTA']
    
    if 'FECHA' in original.columns:
        df['FECHA'] = pd.to_datetime(original['FECHA'])

    if 'UNIDADES' in original.columns:
        df['UNIDADES'] = original['UNIDADES']

    # Filtros laterales
    st.sidebar.header("🔍 Filtros")
    tiendas = sorted(df['NOMBRETIENDA'].dropna().unique())
    articulos = sorted(df['NOMBREARTICULO_VENTA'].dropna().unique())

    tienda_sel = st.sidebar.selectbox("🏪 Selecciona Tienda", ['Todas'] + tiendas)
    articulo_sel = st.sidebar.selectbox("🧾 Selecciona Artículo", ['Todos'] + articulos)

    # Aplicar filtros
    df_filtrado = df.copy()
    if tienda_sel != 'Todas':
        df_filtrado = df_filtrado[df_filtrado['NOMBRETIENDA'] == tienda_sel]
    if articulo_sel != 'Todos':
        df_filtrado = df_filtrado[df_filtrado['NOMBREARTICULO_VENTA'] == articulo_sel]

    # Sección de visualizaciones
    st.subheader("📦 Distribución óptima sugerida")
    distribucion = df_filtrado.groupby(['NOMBRETIENDA', 'NOMBREARTICULO_VENTA'])['PREDICCIONES'].sum().round().astype(int).reset_index()
    st.dataframe(distribucion.sort_values(by='PREDICCIONES', ascending=False).head(10))

    # Gráfico comparativo
    st.subheader("📈 Comparación entre valores reales y predicciones")
    fig, ax = plt.subplots()
    ax.scatter(y, predicciones, alpha=0.5, color='royalblue')
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    ax.set_xlabel('Valores Reales')
    ax.set_ylabel('Predicciones')
    ax.set_title('Dispersión: Reales vs Predicciones')
    st.pyplot(fig)

    # Mapa de calor de correlaciones (versión corregida)
    st.subheader("🔥 Mapa de calor de correlaciones 🔥")
    numerical_cols = df_filtrado.select_dtypes(include=[np.number]).columns
    
    try:
        if 'modelo' in locals() and isinstance(modelo, xgb.XGBRegressor):
            feature_importances = pd.Series(modelo.feature_importances_, index=X.columns)
            sorted_features = feature_importances.sort_values(ascending=False).index
            corr = df_filtrado[numerical_cols].corr().loc[sorted_features, sorted_features]
        else:
            corr = df_filtrado[numerical_cols].corr()
        
        fig3, ax3 = plt.subplots(figsize=(12, 8))
        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            ax=ax3
        )
        plt.title("Correlaciones entre Variables", pad=20)
        st.pyplot(fig3)
    except Exception as e:
        st.error(f"Error al generar mapa de calor: {str(e)}")

    # Importancia de variables
    if isinstance(modelo, xgb.XGBRegressor):
        st.subheader("🧠 Importancia de Variables (XGBoost)")
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        xgb.plot_importance(modelo, ax=ax4, max_num_features=10)
        plt.title("Importancia de Características")
        st.pyplot(fig4)

    # Descarga de resultados
    st.subheader("⬇️ Exportar Resultados")
    distribucion.to_csv("distribucion_optima.csv", index=False)
    with open("distribucion_optima.csv", "rb") as f:
        st.download_button(
            label="Descargar predicciones como CSV",
            data=f,
            file_name="predicciones_ventas.csv",
            mime="text/csv"
        )

# Mensaje si no hay archivo
else:
    st.info("ℹ️ Por favor, sube un archivo Excel para comenzar el análisis.")
