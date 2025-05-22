import os
import joblib
import xgboost as xgb

def entrenar_o_cargar_modelo(X, y, modelo_path="modelo_venta.pkl", reentrenar=False):
    if os.path.exists(modelo_path) and not reentrenar:
        modelo = joblib.load(modelo_path)
        print("✅ Modelo cargado desde archivo existente.")
    else:
        modelo = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5
        )
        modelo.fit(X, y)
        joblib.dump(modelo, modelo_path)
        print("✅ Modelo entrenado y guardado.")
    return modelo