import pandas as pd
from sklearn.preprocessing import LabelEncoder

def cargar_y_preprocesar_datos(archivo):
    df = pd.read_excel(archivo)
    columns_to_drop = ['TIENDA.1', 'CATEGORIA', 'CENTRO', 'PERIODO', 'CLASE', 'FECHA_TRASLADO']
    unnamed_cols = [col for col in df.columns if 'Unnamed' in str(col)]
    columns_to_drop += unnamed_cols
    df.drop(columns=columns_to_drop, axis=1, inplace=True, errors='ignore')

    df['FECHA_HH'] = pd.to_datetime(df['FECHA_HH'], errors='coerce')
    df['DIA_FECHA'] = df['FECHA_HH'].dt.day
    df['MES_FECHA'] = df['FECHA_HH'].dt.month
    df['ANO_FECHA'] = df['FECHA_HH'].dt.year
    df.drop('FECHA_HH', axis=1, inplace=True)

    df.fillna(0, inplace=True)

    label_encoders = {}
    categorical_cols_remaining = df.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols_remaining:
        df[col] = df[col].astype(str)
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    return df, label_encoders