#import re
from flask import Flask, render_template
#from datetime import datetime

#import model

app = Flask (__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/datos')
def datos():
    return render_template('datos.html')

@app.route('/prediccion')
def prediccion():
    return render_template('prediccion.html')

@app.route('/graficas')
def graficas():
    return render_template('graficas.html')

@app.route('/referencias')
def referencias():
    return render_template('referencias.html')

if __name__ == '__main__':
    app.run(debug=True)

#@app.route("/linearregressionpage", methods=["GET", "POST"])
#def calculateGrade():
    #calculateResult = None
    #hours = None

    #if request.method == "POST":
        #try:
           # hours = float(request.form["hours"])
           # calculateResult = linearRegression601N.calculateGrade(hours)
        #except ValueError:
           # calculateResult = "Invalid input. Please enter a number."

    # Generar gráfico con el valor ingresado
    #plot_url = linearRegression601N.generate_plot(hours)

    #return render_template("linearRegressionGrades.html", result=calculateResult, plot_url=plot_url, hours=hours)
