import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            confusion_matrix, classification_report, roc_auc_score)
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt  
import pickle
import os

class VendedoresModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.X_test = None
        self.y_test = None
        
    def cargar_datos(self, file_path='./data/CARACTERIZACION_DE_VENDEDORES_INFORMALES_DEL_MUNICIPIO_DE_CHIA.csv'):
        """Carga y prepara los datos"""
        try:
            self.df = pd.read_csv('./data/CARACTERIZACION_DE_VENDEDORES_INFORMALES_DEL_MUNICIPIO_DE_CHIA.csv', sep=';')
            print("Datos cargados correctamente. Filas:", len(self.df))
            
            # Calcular nivel de vulnerabilidad
            condiciones = [
                self.df['ADULTO MAYOR'] == 'Sí',
                self.df['PERSONA EN DISCAPACIDAD'] == 'Sí',
                self.df['CERTIFICADO REGISTRO ÚNICO DE VÍCTIMAS CONFLICTO ARMADO'] == 'Sí',
                self.df['CERTIFICADO RESGUARDO INDÍGENA'] == 'Sí',
                self.df['IDENTIFICADO LGBTIQ+'] == 'Sí'
            ]
            self.df['PUNTUACION_VULNERABILIDAD'] = sum(cond.astype(int) for cond in condiciones)
            self.df['NIVEL_VULNERABILIDAD'] = pd.cut(
                self.df['PUNTUACION_VULNERABILIDAD'],
                bins=[-1, 1, 3, 5],
                labels=['Bajo', 'Medio', 'Alto']
            )
            
            return True
        except Exception as e:
            print("Error al cargar datos:", str(e))
            return False
    
    def preprocesar_datos(self, target='NIVEL_VULNERABILIDAD'):
        """Preprocesa los datos para el modelo"""
        # Codificación de variables categóricas
        for col in ['GENERO', 'PRODUCTO QUE VENDE', 'PERTENECE A ALGUNA ASOCIACIÓN']:
            self.label_encoders[col] = LabelEncoder()
            self.df[col] = self.label_encoders[col].fit_transform(self.df[col].astype(str))
        
        # Convertir variables binarias
        binary_cols = ['ADULTO MAYOR', 'PERSONA EN DISCAPACIDAD',
                      'CERTIFICADO REGISTRO ÚNICO DE VÍCTIMAS CONFLICTO ARMADO',
                      'CERTIFICADO RESGUARDO INDÍGENA', 'IDENTIFICADO LGBTIQ+']
        for col in binary_cols:
            self.df[col] = self.df[col].map({'Sí': 1, 'No': 0})
        
        # Codificar target
        if target in self.df.columns:
            self.label_encoders[target] = LabelEncoder()
            self.df[target] = self.label_encoders[target].fit_transform(self.df[target])
        
        # Escalar edad
        if 'EDAD' in self.df.columns:
            self.df['EDAD'] = self.scaler.fit_transform(self.df[['EDAD']])
    
    def entrenar_evaluar_modelo(self, features, target='NIVEL_VULNERABILIDAD'):
        """Entrena y evalúa el modelo de regresión logística"""
        try:
            # 1. División del dataset
            X = self.df[features]
            y = self.df[target]
            
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y)
            
            # 2. Ajuste del modelo
            self.model = LogisticRegression(
                class_weight='balanced',
                solver='liblinear',
                max_iter=1000,
                random_state=42
            )
            self.model.fit(self.X_train, self.y_train)
            
            # 3. Predicción y evaluación
            y_pred = self.model.predict(self.X_test)
            
            # 4. Métricas clave
            print("\n=== Métricas de Evaluación ===")
            print(f"Accuracy: {accuracy_score(self.y_test, y_pred):.4f}")
            print(f"Precision: {precision_score(self.y_test, y_pred, average='weighted'):.4f}")
            print(f"Recall: {recall_score(self.y_test, y_pred, average='weighted'):.4f}")
            
            # Reporte completo de clasificación
            print("\nClassification Report:")
            print(classification_report(self.y_test, y_pred))
            
            # Matriz de confusión
            self.generar_matriz_confusion(self.y_test, y_pred)
            
            # AUC-ROC para clasificación binaria
            if len(np.unique(y)) == 2:
                y_prob = self.model.predict_proba(self.X_test)[:, 1]
                print(f"\nAUC-ROC: {roc_auc_score(self.y_test, y_prob):.4f}")
            
            return True
        except Exception as e:
            print("Error en entrenamiento y evaluación:", str(e))
            return False
    
    def generar_matriz_confusion(self, y_true, y_pred, output_dir='static/images'):
        """Genera y guarda matriz de confusión usando matplotlib"""
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
        
        # Etiquetas con los valores
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
        """Realiza predicciones con el modelo entrenado"""
        try:
            # Preparar datos de entrada
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
            
            # Realizar predicción
            prediccion = self.model.predict(datos_procesados[features])[0]
            probabilidad = self.model.predict_proba(datos_procesados[features])[0]
            
            # Mapear a etiquetas originales si hay LabelEncoder
            if 'NIVEL_VULNERABILIDAD' in self.label_encoders:
                prediccion = self.label_encoders['NIVEL_VULNERABILIDAD'].inverse_transform([prediccion])[0]
            
            return {
                'prediccion': prediccion,
                'probabilidades': {cls: round(prob*100, 2) for cls, prob in 
                                 zip(self.model.classes_, probabilidad)}
            }
        except Exception as e:
            print("Error en predicción:", str(e))
            return None
    def get_recommendations(self, nivel_vulnerabilidad): 
        recomendaciones = {
            'Alto': [
                "Prioridad alta para programas de apoyo social",
                "Recomendado para subsidios especiales",
                "Necesita acompañamiento continuo",
                "Acceso prioritario a servicios de salud"
            ],
            'Medio': [
                "Beneficiario de programas de capacitación",
                "Elegible para microcréditos",
                "Seguimiento semestral recomendado",
                "Acceso a talleres de emprendimiento"
            ],
            'Bajo': [
                "Potencial para programas de formalización",
                "Elegible para capacitaciones avanzadas",
                "Seguimiento anual suficiente",
                "Acceso a programas de expansión comercial"
            ]
        }
        return recomendaciones.get(nivel_vulnerabilidad, [])
    def guardar_modelo(self, ruta='modelos/'):
        """Guarda el modelo entrenado"""
        os.makedirs(ruta, exist_ok=True)
        with open(f'{ruta}modelo_vendedores.pkl', 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'label_encoders': self.label_encoders,
                'features': list(self.X_train.columns) if self.X_train is not None else []
            }, f)

    def generar_graficas_analiticas(self, output_dir='static/images'):  
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # 1. Gráfica de distribución de vulnerabilidad
            plt.figure(figsize=(10, 6))
            self.df['NIVEL_VULNERABILIDAD'].value_counts().sort_index().plot(
                kind='bar', 
                color=['#4CAF50', '#FFC107', '#F44336']
            )
            plt.title('Distribución de Niveles de Vulnerabilidad')
            plt.xlabel('Nivel de Vulnerabilidad')
            plt.ylabel('Cantidad de Vendedores')
            plt.xticks(rotation=0)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/distribucion_vulnerabilidad.png')
            plt.close()
            
            # 2. Gráfica de importancia de características
            if hasattr(self, 'model'):
                plt.figure(figsize=(10, 6))
                coefficients = pd.DataFrame({
                    'Feature': self.X_train.columns,
                    'Importance': self.model.coef_[0]
                }).sort_values('Importance', key=abs, ascending=False)
                
                coefficients.plot(kind='bar', x='Feature', y='Importance', legend=False)
                plt.title('Importancia de Características en el Modelo')
                plt.xlabel('Características')
                plt.ylabel('Importancia (coeficientes)')
                plt.tight_layout()
                plt.savefig(f'{output_dir}/importancia_caracteristicas.png')
                plt.close()
            
            # 3. Gráfica de distribución por género
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
        return {
            'matriz_confusion': os.path.exists(f'{base_path}/matriz_confusion.png'),
            'distribucion_vulnerabilidad': os.path.exists(f'{base_path}/distribucion_vulnerabilidad.png'),
            'importancia_caracteristicas': os.path.exists(f'{base_path}/importancia_caracteristicas.png'),
            'distribucion_genero': os.path.exists(f'{base_path}/distribucion_genero.png')
        }

    @staticmethod
    def cargar_modelo(ruta='modelos/modelo_vendedores.pkl'):
        """Carga un modelo previamente entrenado"""
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
    
    


