import os
from model import VendedoresModel

# Define la ruta al archivo CSV
file_path = './data/CARACTERIZACION_DE_VENDEDORES_INFORMALES_DEL_MUNICIPIO_DE_CHIA.csv'

# Instanciar el modelo
modelo = VendedoresModel()

# Asegúrate de que las carpetas 'models' y 'reports' existan
model_dir = './models'
reports_dir = './reports'

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    print(f"Directorio creado: {model_dir}")
if not os.path.exists(reports_dir):
    os.makedirs(reports_dir)
    print(f"Directorio creado: {reports_dir}")

# Cargar y preprocesar los datos
if not modelo.cargar_datos(file_path):
    print("No se pudieron cargar los datos. Terminando el entrenamiento.")
    exit()

if not modelo.preprocesar_datos():
    print("No se pudieron preprocesar los datos. Terminando el entrenamiento.")
    exit()

# Definir las características (columnas de entrada para el modelo)
# ESTOS NOMBRES DEBEN COINCIDIR EXACTAMENTE CON LOS NOMBRES DE LAS COLUMNAS
# RESULTANTES DESPUÉS DEL PREPROCESAMIENTO EN model.py
features = [
    'GENERO',
    'EDAD',
    'ADULTO_MAYOR',
    'NACIONALIDAD',
    'PRODUCTO_QUE_VENDE',
    'PERSONA_EN_DISCAPACIDAD',
    # Nombres de características que el modelo espera según tu output de error
    'CERTIFICADO_REGISTRO_UNICO_DE_VICTIMAS_CONFLICTO_ARMADO',
    'CERTIFICADO_RESGUARDO_INDIGENA',
    'IDENTIFICADO_LGBTIQ+', # CON el '+'
    'PERTENECE_A_ALGUNA_ASOCIACION', # CON 'C' sin tilde
    'MIGRANTE_VENEZOLANO'
]

# Entrenar y evaluar el modelo
print("\nIniciando el entrenamiento y evaluación del modelo...")
# Pasamos las features directamente aquí
if modelo.entrenar_evaluar_modelo(features):
    print("Entrenamiento y evaluación completados exitosamente.")
    # NO necesitas llamar a generar_graficas_analiticas() aquí,
    # ya que se llama internamente dentro de entrenar_evaluar_modelo().
    # modelo.generar_graficas_analiticas() # <--- ¡ELIMINA O COMENTA ESTA LÍNEA!
    
    # Guardar el modelo
    # Asegúrate de que el método guardar_modelo en model.py acepta un nombre de archivo
    # o de lo contrario, modifícalo para que no requiera argumentos si no los necesita.
    # Por ejemplo: modelo.guardar_modelo('modelo_vendedores.pkl')
    # Basado en tu anterior output, 'modelo_vendedores.pkl' era el nombre.
    if modelo.guardar_modelo('modelo_vendedores.pkl'): # Asumiendo que espera un nombre de archivo
        print("El modelo ha sido guardado y está listo para ser utilizado en la aplicación web.")
    else:
        print("Error al guardar el modelo.")
else:
    print("Falló el entrenamiento del modelo.")