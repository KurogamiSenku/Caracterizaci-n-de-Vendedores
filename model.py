import traceback
import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,  # Para validación cruzada
    GridSearchCV     # Para ajuste de hiperparámetros
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import pickle
import os

class VendedoresModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.X_train = None  # Guardamos X_train
        self.X_test = None
        self.y_test = None

    def cargar_datos(self, file_path='./data/CARACTERIZACION_DE_VENDEDORES_INFORMALES_DEL_MUNICIPIO_DE_CHIA.csv'):
        # (Código de carga y limpieza de datos - SIN CAMBIOS)
        try:
            self.df = pd.read_csv(file_path, sep=None, engine='python', encoding='utf-8')
            # Limpiar nombres de columnas: mayúsculas y sin espacios
            self.df.columns = [col.upper().strip() for col in self.df.columns]
            # Imprimir para verificar
            print("Columnas procesadas:", self.df.columns.tolist())

            binary_cols = ['ADULTO MAYOR', 'PERSONA EN DISCAPACIDAD',
                           'CERTIFICADO REGISTRO ÚNICO DE VÍCTIMAS CONFLICTO ARMADO',
                           'CERTIFICADO RESGUARDO INDÍGENA', 'IDENTIFICADO LGBTIQ+',
                           'PERTENECE A ALGUNA ASOCIACIÓN']

            for col in binary_cols:
                self.df[col] = self.df[col].astype(str).str.upper().str.strip().replace(
                    {'SI': 1, 'SÍ': 1, 'YES': 1, 'TRUE': 1, '1': 1}).replace(
                    {'NO': 0, 'FALSE': 0, '0': 0}).fillna(0).astype(int)

            self.df['MIGRANTE_VENEZOLANO'] = self.df['NACIONALIDAD'].apply(
                lambda x: 1 if x == 'VENEZOLANO' else 0)
            self.df['PUNTUACION_VULNERABILIDAD'] = (
                self.df[binary_cols].sum(axis=1) + self.df['MIGRANTE_VENEZOLANO'])
            self.df['NIVEL_VULNERABILIDAD'] = pd.cut(self.df['PUNTUACION_VULNERABILIDAD'],
                                                     bins=[-1, 1, 3, 6],
                                                     labels=['Bajo', 'Medio', 'Alto'])
            return True

        except Exception as e:
            print(f"Error en cargar_datos(): {str(e)}")
            traceback.print_exc()
            return False

    def preprocesar_datos(self, target='NIVEL_VULNERABILIDAD'):
        # (Código de preprocesamiento - SIN CAMBIOS SIGNIFICATIVOS)
        for col in ['GENERO', 'PRODUCTO QUE VENDE', 'PERTENECE A ALGUNA ASOCIACIÓN']:
            self.label_encoders[col] = LabelEncoder()
            self.df[col] = self.label_encoders[col].fit_transform(self.df[col].astype(str))

        if target in self.df.columns:
            self.label_encoders[target] = LabelEncoder()
            self.df[target] = self.label_encoders[target].fit_transform(self.df[target])

        if 'EDAD' in self.df.columns:
            self.df['EDAD'] = self.scaler.fit_transform(self.df[['EDAD']])

    def entrenar_evaluar_modelo(self, features, target='NIVEL_VULNERABILIDAD'):
        try:
            X = self.df[features]
            y = self.df[target]

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y)
            
            # Validación Cruzada (k-fold)
            self.validar_modelo_cruzada(X, y)

            # Ajuste de Hiperparámetros
            self.ajustar_hiperparametros(X, y)

            # Re-entrenamiento con los mejores parámetros
            self.reentrenar_modelo(X, y)

            # Evaluación final
            self.evaluar_modelo()

            return True

        except Exception as e:
            print("Error en entrenamiento, validación o evaluación:", str(e))
            traceback.print_exc()
            return False

    def validar_modelo_cruzada(self, X, y, cv=5):  # cv: número de folds
        """Realiza la validación cruzada k-fold y muestra los resultados."""

        modelo = LogisticRegression(class_weight='balanced', solver='liblinear',
                                    max_iter=1000, random_state=42)
        scores = cross_val_score(modelo, X, y, cv=cv, scoring='f1_weighted')  # Usamos F1-weighted
        print(f"\n=== Validación Cruzada ({cv}-fold) ===")
        print(f"F1-Score Promedio: {np.mean(scores):.4f}")
        print(f"Desviación Estándar: {np.std(scores):.4f}")
        # Puedes guardar los scores si necesitas un análisis más profundo

    def ajustar_hiperparametros(self, X, y):
        """Ajusta los hiperparámetros del modelo usando GridSearchCV."""

        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Fuerza de la regularización
            'penalty': ['l1', 'l2'],  # Tipo de regularización
            'solver': ['liblinear']  # Aseguramos 'liblinear' para l1
        }
        grid_search = GridSearchCV(
            LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
            param_grid, cv=5, scoring='f1_weighted'  # Usamos F1-weighted
        )
        grid_search.fit(X, y)

        print("\n=== Mejor Hiperparámetros ===")
        print("Mejores parámetros:", grid_search.best_params_)
        self.best_params_ = grid_search.best_params_  # Guardamos los mejores parámetros
        self.model = grid_search.best_estimator_  # Guardamos el mejor modelo

    def reentrenar_modelo(self, X, y):
        """Re-entrena el modelo con los mejores hiperparámetros."""

        self.model.fit(X, y)  # Re-entrenar con TODO el conjunto

    def evaluar_modelo(self):
        """Evalúa el modelo en el conjunto de prueba."""

        y_pred = self.model.predict(self.X_test)

        print("\n=== Evaluación Final (Conjunto de Prueba) ===")
        print(f"Accuracy: {accuracy_score(self.y_test, y_pred):.4f}")
        print(f"Precision: {precision_score(self.y_test, y_pred, average='weighted'):.4f}")
        print(f"Recall: {recall_score(self.y_test, y_pred, average='weighted'):.4f}")

        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))

        self.generar_matriz_confusion(self.y_test, y_pred)

        if len(np.unique(self.y_test)) == 2:
            y_prob = self.model.predict_proba(self.X_test)[:, 1]
            print(f"\nAUC-ROC: {roc_auc_score(self.y_test, y_prob):.4f}")

    def generar_matriz_confusion(self, y_true, y_pred, output_dir='static/images'):
        # (Código para generar matriz de confusión - SIN CAMBIOS)
        os.makedirs(output_dir, exist_ok=True)
        cm = confusion_matrix(y_true, y_pred)
        classes = self.label_encoders['NIVEL_VULNERABILIDAD'].classes_ if 'NIVEL_VULNERABILIDAD' in self.label_encoders else np.unique(y_true)

        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Matriz de Confusión')
        plt.colorbar()

        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('Etiqueta Real')
        plt.xlabel('Etiqueta Predicha')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/matriz_confusion.png')
        plt.close()
        print("\nMatriz de confusión generada en:", f'{output_dir}/matriz_confusion.png')

    def predecir(self, datos, features):
        # (Código para predecir - SIN CAMBIOS)
        try:
            datos_procesados = pd.DataFrame([datos])

            for col in features:
                if col in self.label_encoders:
                    datos_procesados[col] = self.label_encoders[col].transform([str(datos[col])])
                elif col in ['ADULTO MAYOR', 'PERSONA EN DISCAPACIDAD',
                           'CERTIFICADO REGISTRO ÚNICO DE VÍCTIMAS CONFLICTO ARMADO',
                           'CERTIFICADO RESGUARDO INDÍGENA', 'IDENTIFICADO LGBTIQ+']:
                    datos_procesados[col] = 1 if datos[col] == 'Sí' else 0
                elif col == 'EDAD':
                    datos_procesados[col] = self.scaler.transform([[datos['EDAD']]])[0][0]

            prediccion = self.model.predict(datos_procesados[features])[0]
            probabilidad = self.model.predict_proba(datos_procesados[features])[0]

            if 'NIVEL_VULNERABILIDAD' in self.label_encoders:
                prediccion = self.label_encoders['NIVEL_VULNERABILIDAD'].inverse_transform([prediccion])[0]

            return {'prediccion': prediccion,
                    'probabilidades': {cls: round(prob * 100, 2) for cls, prob in
                                     zip(self.model.classes_, probabilidad)}}
        except Exception as e:
            print("Error en predicción:", str(e))
            return None

    def get_recommendations(self, nivel_vulnerabilidad):
        # (Código para recomendaciones - SIN CAMBIOS)
        recomendaciones = {
            'Alto': ["Prioridad alta para programas de apoyo social",
                     "Recomendado para subsidios especiales",
                     "Necesita acompañamiento continuo",
                     "Acceso prioritario a servicios de salud"],
            'Medio': ["Beneficiario de programas de capacitación",
                      "Elegible para microcréditos",
                      "Seguimiento semestral recomendado",
                      "Acceso a talleres de emprendimiento"],
            'Bajo': ["Potencial para programas de formalización",
                     "Elegible para capacitaciones avanzadas",
                     "Seguimiento anual suficiente",
                     "Acceso a programas de expansión comercial"]}
        return recomendaciones.get(nivel_vulnerabilidad, [])

    def guardar_modelo(self, ruta='modelos/'):
        # (Código para guardar modelo - SIN CAMBIOS)
        os.makedirs(ruta, exist_ok=True)
        with open(f'{ruta}modelo_vendedores.pkl', 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'label_encoders': self.label_encoders,
                'features': list(self.X_train.columns) if self.X_train is not None else []
            }, f)

    def generar_graficas_analiticas(self, output_dir='static/images'):
        # (Código para generar gráficas - SIN CAMBIOS)
        try:
            os.makedirs(output_dir, exist_ok=True)

            plt.figure(figsize=(10, 6))
            self.df['NIVEL_VULNERABILIDAD'].value_counts().sort_index().plot(
                kind='bar', color=['#4CAF50', '#FFC107', '#F44336'])
            plt.title('Distribución de Niveles de Vulnerabilidad')
            plt.xlabel('Nivel de Vulnerabilidad')
            plt.ylabel('Cantidad de Vendedores')
            plt.xticks(rotation=0)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/distribucion_vulnerabilidad.png')
            plt.close()

            if self.model is not None:
                plt.figure(figsize=(10, 6))
                coefficients = pd.DataFrame({
                    'Feature': self.X_train.columns,
                    'Importance': self.model.coef_[0]  # Acceder a los coeficientes
                }).sort_values('Importance', key=abs, ascending=False)

                coefficients.plot(kind='bar', x='Feature', y='Importance', legend=False)
                plt.title('Importancia de Características en el Modelo')
                plt.xlabel('Características')
                plt.ylabel('Importancia (coeficientes)')
                plt.tight_layout()
                plt.savefig(f'{output_dir}/importancia_caracteristicas.png')
                plt.close()

            plt.figure(figsize=(8, 6))
            self.df['GENERO'].value_counts().plot(kind='pie', autopct='%1.1f%%')
            plt.title('Distribución por Género')
            plt.ylabel('')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/distribucion_genero.png')
            plt.close()

            return True
        except Exception as e:
            print(f"Error generando gráficas analíticas: {str(e)}")
            return False

    def graficas_disponibles(self):
        base_path = 'static/images'
        print(f"Base path en graficas_disponibles: {base_path}") 
        return {
            'matriz_confusion': os.path.exists(f'{base_path}/matriz_confusion.png'),
            'distribucion_vulnerabilidad': os.path.exists(f'{base_path}/distribucion_vulnerabilidad.png'),
            'importancia_caracteristicas': os.path.exists(f'{base_path}/importancia_caracteristicas.png'),
            'distribucion_genero': os.path.exists(f'{base_path}/distribucion_genero.png')
        }

    @staticmethod
    def cargar_modelo(ruta='modelos/modelo_vendedores.pkl'):
        # (Código para cargar modelo - SIN CAMBIOS)
        try:
            with open(ruta, 'rb') as f:
                datos = pickle.load(f)

            modelo = VendedoresModel()
            modelo.model = datos['model']
            modelo.scaler = datos['scaler']
            modelo.label_encoders = datos['label_encoders']

            return modelo
        except Exception as e:
            print("Error cargando modelo:", str(e))
            return None