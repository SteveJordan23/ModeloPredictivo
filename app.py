import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

# Acceder a las credenciales desde Streamlit Secrets Manager
credentials_dict = st.secrets["google_service_account"]

# Crear credenciales a partir del diccionario
credentials = Credentials.from_service_account_info(credentials_dict)

# Autenticación con Google Sheets
gc = gspread.authorize(credentials)

# ID de tu Google Sheet
SPREADSHEET_ID = "190SePJxAYEyfjbMa-EEx-eYY0Hihi8GmKUZHzk5SOX8" 

# Cargar el modelo entrenado
import pickle
with open("modelo_entrenado.pkl", "rb") as model_file:
    saved_data = pickle.load(model_file)
    model = saved_data["model"]
    X_columns = saved_data["columns"]

# Título de la aplicación
st.title("Evaluación de Clientes - Modelo de Churn")

# Subir el archivo con clientes a evaluar
st.header("Subir datos de clientes a evaluar")
uploaded_file = st.file_uploader("Sube un archivo CSV", type="csv")

if uploaded_file is not None:
    # Cargar los datos subidos
    new_data = pd.read_csv(uploaded_file)
    st.write("Datos cargados:")
    st.write(new_data.head())

    # Validar que los datos contienen el identificador 'Customer ID'
    if "Customer ID" not in new_data.columns:
        st.error("La columna 'Customer ID' es obligatoria en los datos. Por favor, agrégala.")
    else:
        # Guardar identificadores
        customer_ids = new_data["Customer ID"]

        # Eliminar columnas irrelevantes excepto 'Customer ID'
        id_columns = ["Customer ID", "City", "Zip Code", "Latitude", "Longitude", "Churn Reason", "Churn Category"]
        new_data = new_data.drop(columns=id_columns, errors="ignore")

        # Manejar valores nulos
        new_data = new_data.fillna(value={"Customer Satisfaction": 3})  # Ejemplo de relleno

        # Codificar variables categóricas con One-Hot Encoding
        new_data = pd.get_dummies(new_data, drop_first=True)

        # Alinear columnas con las del modelo entrenado
        new_data = new_data.reindex(columns=X_columns, fill_value=0)

        # Realizar predicciones
        predictions = model.predict(new_data)

        # Crear un DataFrame con los resultados
        results = pd.DataFrame({
            "Customer ID": customer_ids,
            "Prediction": predictions
        })
        st.write("Resultados de las Predicciones")
        st.write(results)

        # Subir resultados a Google Sheets
        st.header("Subiendo los resultados a Google Sheets...")
        sh = gc.open_by_key(SPREADSHEET_ID)
        worksheet = sh.sheet1  # Usa la primera hoja
        worksheet.update([results.columns.values.tolist()] + results.values.tolist())
        st.success("Resultados subidos exitosamente a Google Sheets.")

        # Enlace para acceder a la hoja de cálculo
        st.write(f"[Abrir Google Sheet](https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID})")
