import streamlit as st
import pandas as pd
import joblib
import gspread
import streamlit as st
from PIL import Image

from gspread_dataframe import set_with_dataframe
from google.oauth2.service_account import Credentials


# Configurar autenticación con Google Sheets
SCOPES = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
#CREDENTIALS_FILE = "credentials.json"  # Archivo de credenciales de Google Cloud
SPREADSHEET_TITLE = "prediccion"  # Título del Google Sheet
#from google.oauth2.service_account import Credentials
#import gspread

# Configuración de credenciales desde Streamlit Secrets Manager o archivo JSON
#SCOPES = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]

try:
    # Opción 1: Usando Streamlit Secrets
    if "google_service_account" in st.secrets:
        credentials_dict = st.secrets["google_service_account"]
        credentials = Credentials.from_service_account_info(credentials_dict, scopes=SCOPES)
    # Opción 2: Usando un archivo JSON (si no usas Streamlit Secrets Manager)
    else:
        CREDENTIALS_FILE = "credentials.json"  # Archivo de credenciales
        credentials = Credentials.from_service_account_file(CREDENTIALS_FILE, scopes=SCOPES)

    # Autorizar gspread con las credenciales
    gc = gspread.authorize(credentials)

except Exception as e:
    st.error(f"Error en la autenticación de Google Sheets: {e}")
    gc = None  # Asegúrate de que gc sea None si hay un problema
# Cargar credenciales para Google Sheets
#credentials = Credentials.from_service_account_file(CREDENTIALS_FILE, scopes=SCOPES)
#gc = gspread.authorize(credentials)

# Cargar credenciales desde Streamlit Secrets
#credentials_dict = st.secrets["google_service_account"]
#credentials = Credentials.from_service_account_info(credentials_dict)
# Configurar título de la aplicación

# Cargar y mostrar el logo de Unitec
logo_path = "unitec.png"  # Asegúrate de usar el nombre exacto del archivo
try:
    logo = Image.open(logo_path)
   # st.image(logo, use_column_width=True)
    st.image(logo, use_container_width=True)
except FileNotFoundError:
    st.warning("No se encontró el logo de Unitec. Asegúrate de que el archivo esté en el directorio correcto.")

st.title("Evaluación de Clientes Modelo de Predicción - Grupo 5")

# Cargar el modelo y el preprocesador
@st.cache_resource
def load_model_and_preprocessor():
    try:
        model = joblib.load("randomForestModel.pkl")
        preprocessor = joblib.load("preprocessor.pkl")
        return model, preprocessor
    except FileNotFoundError as e:
        st.error(f"No se pudo cargar el archivo: {e}")
        return None, None

model, preprocessor = load_model_and_preprocessor()

# Subida de datos
st.header("Subir Datos para Predicción")
uploaded_file = st.file_uploader("Sube un archivo CSV", type="csv")

if uploaded_file is not None:
    # Leer los datos subidos
    new_data = pd.read_csv(uploaded_file)
    st.write("Datos cargados:")
    st.write(new_data.head())

    # Validar si el modelo y el preprocesador están disponibles
    if model is None or preprocessor is None:
        st.error("Error: El modelo o el preprocesador no se cargaron correctamente.")
    else:
        # Definir las columnas que serán restauradas
        columns_to_restore = ["Customer ID", "City", "Zip Code", "Latitude", "Longitude", "Phone Service", "Internet Type", "Gender", "Offer"]

        # Validar que las columnas necesarias estén presentes
        if not all(col in new_data.columns for col in columns_to_restore):
            st.error("Error: Faltan columnas necesarias en los datos subidos.")
        else:
            # Restaurar las columnas eliminadas antes del preprocesamiento
            removed_columns = new_data[columns_to_restore]

            # Preprocesar los datos
            new_data = new_data.drop(columns=columns_to_restore, axis=1, errors="ignore")
            X_new_processed = preprocessor.transform(new_data)

            # Hacer predicciones
            predictions = model.predict(X_new_processed)

            # Agregar predicciones al DataFrame original
            new_data["Predictions"] = predictions
            final_data = pd.concat([removed_columns, new_data], axis=1)

            # Mostrar las predicciones en la aplicación
            st.header("Resultados de las Predicciones")
            st.write(final_data)

            # Guardar los resultados en Google Sheets
            try:
                st.header("Guardando en Google Sheets...")
                try:
                    # Si el Google Sheet ya existe, lo abrimos
                    spreadsheet = gc.open(SPREADSHEET_TITLE)
                    worksheet = spreadsheet.sheet1
                except gspread.exceptions.SpreadsheetNotFound:
                    # Si no existe, lo creamos
                    spreadsheet = gc.create(SPREADSHEET_TITLE)
                    worksheet = spreadsheet.get_worksheet(0)

                # Cargar los datos a la hoja de cálculo
                set_with_dataframe(worksheet, final_data)
                st.success(f"Predicciones guardadas en Google Sheets: {SPREADSHEET_TITLE}")
                st.write(f"[Abrir Google Sheet](https://docs.google.com/spreadsheets/d/{spreadsheet.id})")
            except Exception as e:
                st.error(f"Error al guardar en Google Sheets: {e}")

            # Descargar resultados como archivo CSV
            csv_data = final_data.to_csv(index=False)
            st.download_button(
                label="Descargar Predicciones como CSV",
                data=csv_data,
                file_name="predicciones.csv",
                mime="text/csv"
            )

            # Incrustar el reporte de Looker Studio
            st.header("Reporte en Looker Studio")
            st.markdown(
                """
                <iframe 
                    width="100%" 
                    height="600" 
                    src="https://lookerstudio.google.com/reporting/21028ef0-7f4c-4de2-bd41-97238c0e7951/page/MjRZE/edit" 
                    frameborder="0" 
                    allowfullscreen>
                </iframe>
                """,
                unsafe_allow_html=True
            )
            
            # Enlace directo al reporte de Looker Studio
            st.header("Reporte en Looker Studio")
            st.markdown(
                """
                Haz clic en el enlace para acceder al reporte de Looker Studio:
                [Ver Reporte](https://lookerstudio.google.com/reporting/21028ef0-7f4c-4de2-bd41-97238c0e7951/page/MjRZE/edit)
                """
            )
