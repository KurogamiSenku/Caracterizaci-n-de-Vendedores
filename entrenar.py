# entrenar.py
from model import VendedoresModel

# Instanciar el modelo
modelo = VendedoresModel()

# Cargar y preparar datos
if modelo.cargar_datos():
    modelo.preprocesar_datos()

    # Seleccionar las características relevantes (asegúrate que coincidan con tu dataset)
    features = [
        'GENERO', 'EDAD', 'ADULTO MAYOR', 'PERSONA EN DISCAPACIDAD',
        'CERTIFICADO REGISTRO ÚNICO DE VÍCTIMAS CONFLICTO ARMADO',
        'CERTIFICADO RESGUARDO INDÍGENA', 'IDENTIFICADO LGBTIQ+',
        'PERTENECE A ALGUNA ASOCIACIÓN'
    ]

    # Entrenar y evaluar el modelo
    if modelo.entrenar_evaluar_modelo(features):
        modelo.generar_graficas_analiticas()
        modelo.guardar_modelo()
        print("Modelo entrenado y guardado exitosamente.")
    else:
        print("Falló el entrenamiento del modelo.")
else:
    print("No se pudieron cargar los datos.")
