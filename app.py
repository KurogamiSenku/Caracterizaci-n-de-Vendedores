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
    if request.method == 'POST':
        # Procesar formulario de predicción
        datos = {
            'GENERO': request.form.get('genero'),
            'EDAD': int(request.form.get('edad')),
            'ADULTO MAYOR': request.form.get('adulto-mayor'),
            'PERSONA EN DISCAPACIDAD': request.form.get('discapacidad'),
            'CERTIFICADO REGISTRO ÚNICO DE VÍCTIMAS CONFLICTO ARMADO': request.form.get('victima-conflicto'),
            'CERTIFICADO RESGUARDO INDÍGENA': request.form.get('resguardo-indigena'),
            'IDENTIFICADO LGBTIQ+': request.form.get('identificado-lgbtiq'),
            'PERTENECE A ALGUNA ASOCIACIÓN': request.form.get('pertenece-asociacion'),
            'PRODUCTO QUE VENDE': request.form.get('producto-venta')
        }
        
        if modelo:
            try:
                resultado = modelo.predecir(datos, list(modelo.X_train.columns))
                return render_template('prediccion.html', 
                                    prediccion=resultado['prediccion'],
                                    probabilidades=resultado['probabilidades'],
                                    show_results=True)
            except Exception as e:
                return render_template('prediccion.html', 
                                    error=f"Error en predicción: {str(e)}",
                                    show_results=True)
        else:
            return render_template('prediccion.html', 
                                error="Modelo no disponible",
                                show_results=True)
    
    return render_template('prediccion.html', show_results=False)


@app.route('/graficas')
def mostrar_graficas():
    try:
        # Inicializar modelo si no está cargado
        global modelo
        if modelo is None:
            modelo = VendedoresModel()
            if not modelo.cargar_datos():
                raise Exception("No se pudieron cargar los datos")
            modelo.preprocesar_datos()
            
            # Entrenar modelo si no está entrenado
            if not hasattr(modelo, 'model'):
                features = [
                    'GENERO', 'EDAD', 'ADULTO MAYOR', 'PERSONA EN DISCAPACIDAD',
                    'CERTIFICADO REGISTRO ÚNICO DE VÍCTIMAS CONFLICTO ARMADO',
                    'CERTIFICADO RESGUARDO INDÍGENA', 'IDENTIFICADO LGBTIQ+'
                ]
                modelo.entrenar_evaluar_modelo(features)
        
        # Generar gráficas si no existen
        if not any(modelo.graficas_disponibles().values()):
            modelo.generar_matriz_confusion(modelo.y_test, modelo.model.predict(modelo.X_test))
            modelo.generar_graficas_analiticas()
        
        return render_template('graficas.html',
                            graficas=modelo.graficas_disponibles(),
                            titulo="Análisis Gráfico Completo")
    
    except Exception as e:
        error_msg = f"Error al generar gráficas: {str(e)}"
        app.logger.error(error_msg)
        return render_template('error.html',
                            error_message="No se pudieron mostrar las gráficas",
                            error_details=error_msg if app.debug else None), 500



@app.route('/referencias')
def referencias():
    return render_template('referencias.html')


if __name__ == '__main__':
    app.run(debug=True)