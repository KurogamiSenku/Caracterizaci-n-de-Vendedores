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

modelo = VendedoresModel()
modelo.cargar_datos()
modelo.preprocesar_datos()
features = ['GENERO', 'EDAD', 'ADULTO MAYOR', 'PERSONA EN DISCAPACIDAD',
            'CERTIFICADO REGISTRO ÚNICO DE VÍCTIMAS CONFLICTO ARMADO',
            'CERTIFICADO RESGUARDO INDÍGENA', 'IDENTIFICADO LGBTIQ+',
            'PERTENECE A ALGUNA ASOCIACIÓN', 'PRODUCTO QUE VENDE', 'MIGRANTE_VENEZOLANO']

modelo.entrenar_evaluar_modelo(features)



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
        try:
            datos_usuario = {
                'GENERO': request.form['GENERO'],
                'EDAD': int(request.form['EDAD']),
                'ADULTO MAYOR': request.form['ADULTO_MAYOR'],
                'NACIONALIDAD': request.form['NACIONALIDAD'],
                'PRODUCTO QUE VENDE': request.form['PRODUCTO QUE VENDE'],
                'DETALLE DEL PRODUCTO': request.form['DETALLE DEL PRODUCTO'],
                'PERSONA EN DISCAPACIDAD': request.form['PERSONA EN DISCAPACIDAD'],
                'CERTIFICADO REGISTRO ÚNICO DE VÍCTIMAS CONFLICTO ARMADO': request.form['CERTIFICADO REGISTRO ÚNICO DE VÍCTIMAS CONFLICTO ARMADO'],
                'CERTIFICADO RESGUARDO INDÍGENA': request.form['CERTIFICADO RESGUARDO INDÍGENA'],
                'IDENTIFICADO LGBTIQ+': request.form['IDENTIFICADO LGBTIQ+'],
                'PERTENECE A ALGUNA ASOCIACIÓN': request.form['PERTENECE A ALGUNA ASOCIACIÓN'],
                'MIGRANTE_VENEZOLANO': 1 if 'VENEZOLANO' in request.form['NACIONALIDAD'].strip().upper() else 0
            }

            features = modelo.X_train.columns.tolist() if modelo.X_train is not None else [
                'GENERO', 'EDAD', 'ADULTO MAYOR', 'PERSONA EN DISCAPACIDAD',
                'CERTIFICADO REGISTRO ÚNICO DE VÍCTIMAS CONFLICTO ARMADO',
                'CERTIFICADO RESGUARDO INDÍGENA', 'IDENTIFICADO LGBTIQ+',
                'PERTENECE A ALGUNA ASOCIACIÓN', 'PRODUCTO QUE VENDE', 'MIGRANTE_VENEZOLANO'
            ]
            print("Datos recibidos del formulario:")
            print(datos_usuario)

            resultado = modelo.predecir(datos_usuario, features=features)
            if resultado is None:
                raise ValueError("La predicción falló. Verifica los valores ingresados.")

            prediccion, probabilidades = resultado


            datos_usuario['NIVEL_VULNERABILIDAD'] = prediccion
            modelo.df = pd.concat([modelo.df, pd.DataFrame([datos_usuario])], ignore_index=True)
            modelo.df.to_csv('./data/CARACTERIZACION_DE_VENDEDORES_INFORMALES_DEL_MUNICIPIO_DE_CHIA.csv', index=False)

            return render_template('prediccion.html',
                                   show_results=True,
                                   prediccion=prediccion,
                                   probabilidades=probabilidades,
                                   error=None,
                                   opciones_producto=modelo.obtener_opciones_validas('PRODUCTO QUE VENDE'),
                                   opciones_genero=modelo.obtener_opciones_validas('GENERO'))

        except Exception as e:
            print("Error al realizar la predicción:", str(e))
            return render_template('prediccion.html',
                                   show_results=True,
                                   prediccion=None,
                                   probabilidades=None,
                                   error=str(e),
                                   opciones_producto=modelo.obtener_opciones_validas('PRODUCTO QUE VENDE'),
                                   opciones_genero=modelo.obtener_opciones_validas('GENERO'))

    else:
        opciones_genero = modelo.obtener_opciones_validas('GENERO')
        opciones_producto = modelo.obtener_opciones_validas('PRODUCTO QUE VENDE')
        opciones_nacionalidad = modelo.obtener_opciones_validas('NACIONALIDAD')

        return render_template('prediccion.html',
                               show_results=False,
                               opciones_genero=opciones_genero,
                               opciones_producto=opciones_producto,
                               opciones_nacionalidad=opciones_nacionalidad)

@app.route('/graficas')
def mostrar_graficas():
    try:
        modelo = VendedoresModel()
        modelo.cargar_datos()
        modelo.preprocesar_datos()
        modelo.entrenar_evaluar_modelo([...])  
        modelo.generar_graficas_analiticas()
        return render_template('graficas.html', graficas=modelo.graficas_disponibles())
    except Exception as e:
        return render_template('graficas.html', error=str(e), graficas={})


@app.route('/referencias')
def referencias():
    return render_template('referencias.html')


if __name__ == '__main__':
    app.run(debug=True)