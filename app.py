import io
import traceback
from flask import Flask, render_template, request
from model import VendedoresModel 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, roc_auc_score, f1_score
import matplotlib.pyplot as plt
import os
import logging
import base64

app = Flask(__name__)

# Configuración inicial (asegúrate de tener esto)
app.config['UPLOAD_FOLDER'] = 'static/images'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

modelo = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/datos')
def datos():
    try:
        # Verificar carpeta 'data'
        if not os.path.exists('data'):
            return render_template('datos.html', 
                               error="No existe la carpeta 'data'",
                               datos=[])

        # Buscar archivos CSV
        csv_files = [f for f in os.listdir('data') if f.endswith('.csv')]
        if not csv_files:
            return render_template('datos.html',
                               error="No hay archivos CSV en la carpeta 'data'",
                               datos=[])

        # Leer el archivo CSV
        csv_path = os.path.join('data', csv_files[0])
        
        # Intentar diferentes combinaciones de encoding/separador
        encodings = ['utf-8', 'latin1', 'iso-8859-1']
        separators = [',', ';', '\t']
        
        df = None
        for encoding in encodings:
            for sep in separators:
                try:
                    df = pd.read_csv(csv_path, encoding=encoding, sep=sep)
                    break
                except:
                    continue
            if df is not None:
                break

        if df is None:
            return render_template('datos.html',
                               error="No se pudo leer el archivo CSV con ningún encoding/separador conocido",
                               datos=[])

        # Limpieza básica de datos
        df = df.fillna('')
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        
        # Convertir a lista de diccionarios (asegurando formato UTF-8)
        datos = []
        for _, row in df.iterrows():
            clean_row = {}
            for col, val in row.items():
                if isinstance(val, str):
                    clean_row[col] = val.encode('utf-8', 'ignore').decode('utf-8')
                else:
                    clean_row[col] = val
            datos.append(clean_row)

        # Debug: Verificar estructura de datos
        print(f"Total registros: {len(datos)}")
        print(f"Primer registro: {datos[0]}")
        print(f"Columnas: {list(datos[0].keys())}")

        return render_template('datos.html',
                           datos=datos
                           )

    except Exception as e:
        traceback.print_exc()  # Esto imprimirá el error completo en consola
        return render_template('datos.html',
                           error=f"Error inesperado: {str(e)}",
                           datos=[]), 500

@app.route('/prediccion', methods=['GET', 'POST'])
def prediccion():
    global modelo  # Usar la variable global 'modelo'

    if request.method == 'POST':
        try:
            # Recoger los datos del formulario
            form_data = request.form.to_dict()

            # Convertir los datos a un DataFrame
            pred_df = pd.DataFrame([form_data])

            # Preprocesar los datos (igual que en entrenamiento)
            for col in pred_df.columns:
                if col in modelo.label_encoders:
                    pred_df[col] = pred_df[col].astype(str)  # Asegurar que es string antes de transformar
                    pred_df[col] = modelo.label_encoders[col].transform(pred_df[col])
                pred_df[col] = pd.to_numeric(pred_df[col], errors='coerce')  # Convertir a numérico

            # Escalar los datos
            pred_scaled = modelo.scaler.transform(pred_df)

            # Realizar la predicción
            prediccion = modelo.model.predict(pred_scaled)
            prediccion_texto = modelo.label_encoders['NIVEL_VULNERABILIDAD'].inverse_transform(prediccion)

            return render_template('prediccion.html', prediccion=prediccion_texto[0])

        except Exception as e:
            error_msg = f"Error al realizar la predicción: {str(e)}"
            app.logger.error(error_msg)
            return render_template('error.html', error_message="Error en la predicción", error_details=error_msg if app.debug else None), 500

    return render_template('prediccion.html', prediccion=None)


@app.route('/graficas')
def mostrar_graficas():
    try:
        modelo = VendedoresModel()
        modelo.cargar_datos()
        modelo.preprocesar_datos()
        modelo.entrenar_evaluar_modelo([...])  # Tus features aquí
        modelo.generar_graficas_analiticas()
        return render_template('graficas.html', graficas=modelo.graficas_disponibles())
    except Exception as e:
        return render_template('graficas.html', error=str(e), graficas={})


@app.route('/referencias')
def referencias():
    return render_template('referencias.html')


if __name__ == '__main__':
    app.run(debug=True)