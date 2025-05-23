import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
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
    # Cargar y preprocesar datos
    df, encoders = cargar_y_preprocesar_datos(archivo)

    # Validar columnas esenciales
    required_columns = ['CANTIDAD', 'NOMBRETIENDA', 'NOMBREARTICULO_VENTA']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        st.error(f"❌ Faltan columnas requeridas después del preprocesamiento: {', '.join(missing_cols)}")
        st.stop()

    # Separar variables para entrenamiento
    X = df.drop(['CANTIDAD', 'TOTAL_LINEA', 'FACTURA', 'UNIDADES'], axis=1, errors='ignore')
    y = df['CANTIDAD']

    # Entrenar o cargar modelo
    modelo = entrenar_o_cargar_modelo(X, y, modelo_path="modelo_venta.pkl", reentrenar=reentrenar)

    # Predicciones
    predicciones = modelo.predict(X)
    predicciones = np.round(predicciones).astype(int)  # Eliminar decimales

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

    # Recuperar columnas originales desde archivo subido
    original = pd.read_excel(archivo)
    df['PREDICCIONES'] = predicciones
    df['NOMBRETIENDA'] = original['NOMBRETIENDA'].fillna('Sin Tienda')
    df['NOMBREARTICULO_VENTA'] = original['NOMBREARTICULO_VENTA'].fillna('Sin Artículo')

    if 'FECHA' in original.columns:
        df['FECHA'] = pd.to_datetime(original['FECHA'])

    if 'UNIDADES' in original.columns:
        df['UNIDADES'] = original['UNIDADES']

    # 📍 FILTROS LATERALES
    st.sidebar.header("🔍 Filtros")

    # Validar y limpiar tiendas/artículos antes de usar
    tiendas = sorted(df['NOMBRETIENDA'].dropna().astype(str).unique())
    articulos = sorted(df['NOMBREARTICULO_VENTA'].dropna().astype(str).unique())

    tienda_sel = st.sidebar.selectbox("🏪 Selecciona Tienda", ['Todas'] + tiendas)
    articulo_sel = st.sidebar.selectbox("🧾 Selecciona Artículo", ['Todos'] + articulos)

    df_filtrado = df.copy()
    if tienda_sel != 'Todas':
        df_filtrado = df_filtrado[df_filtrado['NOMBRETIENDA'] == tienda_sel]
    if articulo_sel != 'Todos':
        df_filtrado = df_filtrado[df_filtrado['NOMBREARTICULO_VENTA'] == articulo_sel]

    # 📦 DISTRIBUCIÓN
    distribucion = df_filtrado.groupby(['NOMBRETIENDA', 'NOMBREARTICULO_VENTA'])['PREDICCIONES'].sum().round().astype(int).reset_index()
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

    # 📊 RANKING DE VARIABLES POR CORRELACIÓN CON 'CANTIDAD'
    st.subheader("📊 Ranking de variables por correlación con 'CANTIDAD'")
    df_numeric = df_filtrado.select_dtypes(include=[np.number])

    if 'CANTIDAD' in df_numeric.columns:
        corr_with_target = df_numeric.corr()['CANTIDAD'].abs().sort_values(ascending=False)
        corr_df = pd.DataFrame(corr_with_target).rename(columns={'CANTIDAD': 'Correlación Absoluta'})
        st.dataframe(corr_df.style.background_gradient(cmap='Blues'))
    else:
        st.warning("La columna 'CANTIDAD' no está disponible.")

    # 🔥 MAPA DE CALOR MEJORADO - Solo variables clave
    st.subheader("🔥 Mapa de calor: Variables clave")
    columnas_relevantes = [
        'CANTIDAD',
        'UNIDADES',
        'NOMBRETIENDA_cod',
        'NOMBREARTICULO_VENTA_cod',
        'MARCAARTICULO_cod'
    ]
    cols_existentes = [col for col in columnas_relevantes if col in df_filtrado.columns]

    if len(cols_existentes) >= 2:
        df_relevant = df_filtrado[cols_existentes]
        corr = df_relevant.corr()

        fig3, ax3 = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            linewidths=0.5,
            cbar_kws={"shrink": 0.75},
            annot_kws={"size": 10},
            square=True,
            ax=ax3
        )
        ax3.set_title("Correlación entre variables clave", fontsize=14)
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig3)
    else:
        st.warning("No hay suficientes variables clave para generar el mapa de calor.")

    # 📊 Comparación entre Distribución Manual y Predicción del Modelo
    st.subheader("📊 Comparación entre Distribución Manual (UNIDADES) y Predicción del Modelo")

    if 'UNIDADES' in df_filtrado.columns:
        comparacion = df_filtrado.groupby(['NOMBRETIENDA', 'NOMBREARTICULO_VENTA'])[['UNIDADES', 'PREDICCIONES']].sum().round().astype(int).reset_index()
        comparacion['DIFERENCIA'] = comparacion['PREDICCIONES'] - comparacion['UNIDADES']
        comparacion.sort_values(by='DIFERENCIA', ascending=False, inplace=True)

        # Filtros interactivos adicionales
        st.markdown("### 🔍 Filtro adicional para análisis detallado")
        tiendas_comp = ['Todas'] + sorted(df_filtrado['NOMBRETIENDA'].astype(str).unique())
        articulos_comp = ['Todos'] + sorted(df_filtrado['NOMBREARTICULO_VENTA'].astype(str).unique())

        tienda_graf = st.selectbox("Selecciona una tienda para análisis gráfico", tiendas_comp)
        articulo_graf = st.selectbox("Selecciona un artículo para análisis gráfico", articulos_comp)

        df_graf = comparacion.copy()
        if tienda_graf != 'Todas':
            df_graf = df_graf[df_graf['NOMBRETIENDA'] == tienda_graf]
        if articulo_graf != 'Todos':
            df_graf = df_graf[df_graf['NOMBREARTICULO_VENTA'] == articulo_graf]

        if not df_graf.empty:
            # Cálculo del porcentaje de diferencia
            df_graf['DIFERENCIA_PCT'] = (df_graf['DIFERENCIA'] / df_graf['PREDICCIONES']).abs() * 100
            umbral_pct = 20

            colors_unidades = ['red' if row['DIFERENCIA_PCT'] > umbral_pct else 'skyblue' for _, row in df_graf.iterrows()]
            colors_prediccion = ['red' if row['DIFERENCIA_PCT'] > umbral_pct else 'lightgreen' for _, row in df_graf.iterrows()]

            # Gráfico de barras comparativo con alertas
            st.markdown("### 📊 Comparación: UNIDADES vs PREDICCIONES (con alertas visuales)")
            fig_comp, ax_comp = plt.subplots(figsize=(12, 6))

            bar_width = 0.35
            indices = np.arange(len(df_graf))

            ax_comp.bar(indices, df_graf['UNIDADES'], width=bar_width, label='UNIDADES', color=colors_unidades)
            ax_comp.bar(indices + bar_width, df_graf['PREDICCIONES'], width=bar_width, label='PREDICCIONES', color=colors_prediccion)

            ax_comp.set_title(f"Comparativa Unidades vs Predicciones {' - ' + tienda_graf if tienda_graf != 'Todas' else ''} {' - ' + articulo_graf if articulo_graf != 'Todos' else ''}", fontsize=14)
            ax_comp.set_ylabel("Cantidad")
            ax_comp.set_xlabel("Artículo")
            ax_comp.set_xticks(indices + bar_width / 2)
            ax_comp.set_xticklabels(df_graf['NOMBREARTICULO_VENTA'], rotation=45)
            ax_comp.legend()

            ax_comp.axhline(y=0, color='black', linewidth=0.8, linestyle='--')
            ax_comp.text(0.95, 0.95, f"Diferencia > {umbral_pct}% resaltada", transform=ax_comp.transAxes,
                         fontsize=10, verticalalignment='top', horizontalalignment='right',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            plt.tight_layout()
            st.pyplot(fig_comp)

            # Tabla y descarga
            st.markdown("### 📌 Diferencia entre predicción y unidades:")
            st.dataframe(df_graf[['NOMBRETIENDA', 'NOMBREARTICULO_VENTA', 'UNIDADES', 'PREDICCIONES', 'DIFERENCIA']])
            csv = df_graf.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Descargar tabla como CSV",
                data=csv,
                file_name="comparativa_unidades_predicciones.csv",
                mime="text/csv"
            )
        else:
            st.info("🚫 No hay datos que coincidan con los filtros seleccionados.")
    else:
        st.warning("La columna 'UNIDADES' no está disponible en los datos.")

    # 📉 GRÁFICA DE LÍNEAS (si hay fechas)
    if 'FECHA' in df_filtrado.columns:
        st.subheader("📆 Evolución temporal de predicción vs real")
        df_linea = df_filtrado.groupby('FECHA')[['CANTIDAD', 'PREDICCIONES']].sum().round().astype(int).reset_index()
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
