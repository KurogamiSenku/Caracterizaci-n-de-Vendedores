import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')  # <-- Soluciona el error de GUI
import matplotlib.pyplot as plt

import io
import base64
from sklearn.linear_model import LinearRegression

data = {
    "Study Hours": [10, 15, 12, 8, 14, 5, 16, 7, 11, 13, 9, 4, 18, 3, 17, 6, 14, 2, 20, 1],
    "Final Grade": [3.8, 4.2, 3.6, 3, 4.5, 2.5, 4.8, 2.8, 3.7, 4, 3.2, 2.2, 5, 1.8, 4.9, 2.7, 4.4, 1.5, 5, 1]
}

df = pd.DataFrame(data)
x= df[["Study Hours"]]
y = df[["Final Grade"]]

model = LinearRegression()
model.fit(x,y)

#generar grafico con matplop e imagen 
def generate_plot(hours=None):
    fig, ax = plt.subplots()

    # Dibujar puntos reales y la línea de regresión
    ax.scatter(df["Study Hours"], df["Final Grade"], color="blue", label="Datos Reales")
    ax.plot(df["Study Hours"], model.predict(x), color="red", label="Línea de Regresión")

    # Si el usuario ingresó horas, agregar el punto de predicción
    if hours is not None:
        predicted_grade = calculateGrade(hours)
        ax.scatter(hours, predicted_grade, color="green", marker="o", s=100, label="Predicción Usuario")

    ax.set_xlabel("Horas de Estudio")
    ax.set_ylabel("Calificación Final")
    ax.set_title("Regresión Lineal: Horas de Estudio vs Calificación")
    ax.legend()

    # Guardar imagen en memoria
    img = io.BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    plt.close(fig)

    # Convertir imagen a base64
    encoded_img = base64.b64encode(img.getvalue()).decode("utf-8")
    return encoded_img

def calculateGrade(hours):
    result = model.predict([[hours]])
    grade = float(result[0][0])  # asegurarse de tener un float
    grade = min(max(grade, 0.0), 5.0)  # limitar entre 0 y 5
    return round(grade, 2)  # devolver número redondeado
    
