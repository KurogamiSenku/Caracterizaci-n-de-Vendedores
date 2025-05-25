import os
import traceback
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
from model import VendedoresModel # Asegúrate de que 'model.py' esté en la misma carpeta o accesible
import logging

# Configurar logging para ver mensajes en la consola de Flask
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# Configuración inicial para la carpeta de imágenes estáticas
# Las gráficas se guardarán aquí por model.py
app.config['STATIC_REPORTS_FOLDER'] = 'static/reports'
os.makedirs(app.config['STATIC_REPORTS_FOLDER'], exist_ok=True)

# Instanciar el modelo al inicio de la aplicación
# Esto se hace una vez al iniciar Flask para cargar el modelo ya entrenado.
modelo = VendedoresModel()

# --- Bloque de Carga del Modelo al Inicio de la Aplicación ---
# Se intenta cargar el modelo guardado por 'entrenar.py'.
# Si el modelo no se carga, la aplicación no podrá hacer predicciones.
MODEL_PATH = 'modelo_vendedores.pkl' # Asegúrate de que esta sea la única definición de MODEL_PATH aquí

try:
    print(f"DEBUG APP: MODEL_PATH antes de llamar a cargar_modelo: '{MODEL_PATH}'") # <-- AÑADIR ESTA LÍNEA

    if modelo.cargar_modelo(MODEL_PATH):
        print("✅ Modelo cargado correctamente y listo para hacer predicciones.")
        if hasattr(modelo, 'features_trained_on') and modelo.features_trained_on is not None:
            print(f"Features esperadas por el modelo (desde el modelo cargado): {modelo.features_trained_on}")
        else:
            print("Advertencia: Las características de entrenamiento no se cargaron con el modelo.")
    else:
        print(f"❌ ERROR: El modelo entrenado no pudo ser cargado. Asegúrate de haber ejecutado entrenar.py.")
        modelo = None

except Exception as e:
    print(f"❌ ERROR: Falló la carga del modelo al iniciar la aplicación: {e}")
    traceback.print_exc()
    modelo = None
# --- Rutas de la Aplicación Flask ---

@app.route('/')
def home():
    """Ruta de la página de inicio."""
    return render_template('index.html')

@app.route('/datos')
def datos():
    """
    Muestra los datos del archivo CSV.
    Intenta leer el CSV con diferentes codificaciones y separadores.
    """
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

        # Leer el archivo CSV (tomamos el primero que encontremos)
        csv_path = os.path.join('data', csv_files[0])

        # Intentar diferentes combinaciones de encoding/separador
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'utf-8-sig']
        separators = [',', ';', '\t']

        df = None
        for encoding in encodings:
            for sep in separators:
                try:
                    df = pd.read_csv(csv_path, encoding=encoding, sep=sep)
                    print(f"CSV leído exitosamente con encoding '{encoding}' y separador '{sep}'.")
                    break # Salir de los bucles si la lectura es exitosa
                except Exception:
                    continue # Intentar con la siguiente combinación
            if df is not None:
                break # Salir del bucle de encodings si la lectura es exitosa

        if df is None:
            return render_template('datos.html',
                                   error="No se pudo leer el archivo CSV con ningún encoding/separador conocido",
                                   datos=[])

        # Limpieza básica de datos para visualización (no para el modelo)
        df = df.fillna('')
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

        # Convertir a lista de diccionarios para pasar a la plantilla
        datos = df.to_dict(orient='records')

        # Debug: Verificar estructura de datos
        print(f"Total registros para visualización: {len(datos)}")
        if datos:
            print(f"Primer registro para visualización: {datos[0]}")
            print(f"Columnas para visualización: {list(datos[0].keys())}")

        return render_template('datos.html', datos=datos)

    except Exception as e:
        traceback.print_exc()
        return render_template('datos.html',
                               error=f"Error inesperado al cargar datos para visualización: {str(e)}",
                               datos=[]), 500


@app.route('/prediccion', methods=['GET', 'POST'])
def prediccion():
    """
    Maneja el formulario de predicción y muestra los resultados.
    """
    prediccion_display = None # Variable para mostrar la etiqueta de la predicción (Vulnerable/No Vulnerable)
    prob_vulnerable = None    # Probabilidad numérica para el frontend
    prob_no_vulnerable = None # Probabilidad numérica para el frontend
    mensaje_error = None
    show_results = False # Variable para controlar la visibilidad de los resultados en HTML

    # Opciones por defecto para los desplegables del formulario
    opciones_genero = []
    opciones_producto = []
    opciones_nacionalidad = []

    # Si el modelo se cargó correctamente, obtener las opciones reales de los LabelEncoders
    if modelo and modelo.label_encoders:
        opciones_genero = modelo.obtener_opciones_validas('GENERO')
        opciones_producto = modelo.obtener_opciones_validas('PRODUCTO_QUE_VENDE')
        opciones_nacionalidad = modelo.obtener_opciones_validas('NACIONALIDAD')
    else:
        mensaje_error = "El modelo no está cargado. Por favor, asegúrate de haber ejecutado entrenar.py."
        # Si el modelo no está cargado, podemos usar opciones predeterminadas para que el formulario se muestre
        opciones_genero = ['FEMENINO', 'MASCULINO']
        opciones_producto = ['Alimentos', 'Ropa', 'Artesanías', 'Accesorios', 'Otros']
        opciones_nacionalidad = ['COLOMBIANA', 'OTRA', 'VENEZOLANA']


    print("\n=== DEBUG: OPCIONES DEL FORMULARIO ===")
    print(f"Opciones género: {opciones_genero}")
    print(f"Opciones producto: {opciones_producto}")
    print(f"Opciones nacionalidad: {opciones_nacionalidad}")
    print("===============================\n")

    if request.method == 'POST': # Quitamos 'and modelo' aquí, ya se maneja en el bloque de carga inicial.
        if not modelo.model: # Verifica si el modelo está cargado antes de intentar predecir
            mensaje_error = "El modelo no está cargado. No se puede realizar la predicción."
        else:
            try:
                # Recopilar datos del formulario
                # ¡CRÍTICO! Los nombres de las claves en este diccionario
                # DEBEN coincidir EXACTAMENTE con los nombres de las características
                # que el modelo fue entrenado (definidas en 'features' en entrenar.py
                # y los nombres finales después del preprocesamiento en model.py).
                datos_entrada = {
                    'GENERO': request.form['genero'],
                    'EDAD': int(request.form['edad']),
                    'NACIONALIDAD': request.form['nacionalidad'],
                    'PRODUCTO_QUE_VENDE': request.form['producto_que_vende'],
                    # --- Manejo de Checkboxes: 'on' a 1, ausente a 0 ---
                    'ADULTO_MAYOR': 1 if 'adulto_mayor' in request.form else 0,
                    'PERSONA_EN_DISCAPACIDAD': 1 if 'discapacidad' in request.form else 0,
                    'CERTIFICADO_REGISTRO_UNICO_DE_VICTIMAS_CONFLICTO_ARMADO': 1 if 'victima_conflicto' in request.form else 0,
                    'CERTIFICADO_RESGUARDO_INDIGENA': 1 if 'resguardo_indigena' in request.form else 0,
                    'IDENTIFICADO_LGBTIQ+': 1 if 'lgbtiq' in request.form else 0, # Corregido: 'lgbtiq' en vez de 'lgb_tiq'
                    'PERTENECE_A_ALGUNA_ASOCIACION': 1 if 'pertenece_asociacion' in request.form else 0, # Corregido el nombre del campo del formulario
                    'MIGRANTE_VENEZOLANO': 1 if 'migrante_venezolano' in request.form else 0 # Asegúrate que este checkbox exista en el HTML
                }

                print(f"Datos recibidos para predicción: {datos_entrada}")

                # Realizar la predicción
                prediccion_decodificada, probabilidades_dict = modelo.predecir(datos_entrada)

                if prediccion_decodificada is not None and probabilidades_dict is not None:
                    # Mapear la predicción decodificada a las etiquetas del frontend
                    if prediccion_decodificada == 'Alto': # Si 'Alto' de tu modelo significa 'Vulnerable'
                        prediccion_display = 'Vulnerable'
                    elif prediccion_decodificada == 'Bajo': # Si 'Bajo' de tu modelo significa 'No Vulnerable'
                        prediccion_display = 'No Vulnerable'
                    else: # En caso de que el modelo devuelva algo inesperado
                        prediccion_display = prediccion_decodificada

                    # Asignar probabilidades con los nombres correctos para el HTML
                    # y como NÚMEROS (no cadenas formateadas)
                    prob_vulnerable = probabilidades_dict.get('Alto', 0) # Obtén la probabilidad de la clase 'Alto'
                    prob_no_vulnerable = probabilidades_dict.get('Bajo', 0) # Obtén la probabilidad de la clase 'Bajo'

                    show_results = True # ¡Activamos la visibilidad de los resultados!

                else:
                    mensaje_error = "Error al realizar la predicción. Revisa la consola para más detalles."

            except ValueError as ve:
                mensaje_error = f"Error en los datos de entrada: {ve}. Asegúrate de que todos los campos numéricos sean válidos."
                app.logger.error(f"ValueError en prediccion: {ve}")
                traceback.print_exc()
            except Exception as e:
                mensaje_error = f"Ocurrió un error inesperado durante la predicción: {e}"
                app.logger.error(f"Error general en prediccion: {e}")
                traceback.print_exc()

    # Renderiza la plantilla con los resultados o el formulario
    return render_template('prediccion.html',
                            prediccion=prediccion_display, # Pasa la etiqueta de la predicción
                            error=mensaje_error, # Renombrado a 'error' para coincidir con el HTML
                            prob_vulnerable=prob_vulnerable, # Probabilidad numérica
                            prob_no_vulnerable=prob_no_vulnerable, # Probabilidad numérica
                            opciones_genero=opciones_genero,
                            opciones_producto=opciones_producto,
                            opciones_nacionalidad=opciones_nacionalidad,
                            show_results=show_results # Pasa la variable de control de visibilidad
                           )

@app.route('/graficas')
def mostrar_graficas():
    """
    Muestra las gráficas de evaluación del modelo previamente generadas y guardadas.
    No re-genera las gráficas aquí, solo las muestra desde la carpeta estática.
    """
    try:
        # Simplemente renderiza la plantilla que contiene las etiquetas <img>
        # que apuntan a las imágenes guardadas en 'static/reports/'.
        # Asegúrate de que 'entrenar.py' haya guardado las imágenes en esta ruta.
        return render_template('graficas.html')
    except Exception as e:
        traceback.print_exc()
        return render_template('graficas.html', error=f"Error al mostrar gráficas: {str(e)}")

@app.route('/referencias')
def referencias():
    """Ruta para la página de referencias."""
    return render_template('referencias.html')

if __name__ == '__main__':
    app.run(debug=True)