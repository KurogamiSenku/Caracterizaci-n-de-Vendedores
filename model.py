import traceback
import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    f1_score,
    roc_curve 
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from imblearn.over_sampling import SMOTE

class VendedoresModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.df = None
        self.features_trained_on = None # Añadido para almacenar las características usadas en el entrenamiento

    def cargar_datos(self, file_path='./data/CARACTERIZACION_DE_VENDEDORES_INFORMALES_DEL_MUNICIPIO_DE_CHIA.csv'):
        print(f"Cargando datos desde: {file_path}")
        try:
            for sep in [',', ';', '\t']:
                for encoding in ['utf-8', 'latin1', 'iso-8859-1', 'utf-8-sig']:
                    try:
                        self.df = pd.read_csv(file_path, sep=sep, encoding=encoding)
                        print(f"Datos cargados exitosamente con separador '{sep}' y codificación '{encoding}'.")

                        # --- INICIO DE LA LÓGICA DE LIMPIEZA Y RENOMBRE ESPECÍFICO ---
                        # Primero, estandarizamos todos los nombres de columnas a mayúsculas,
                        # quitando espacios y reemplazándolos por guiones bajos.
                        self.df.columns = self.df.columns.str.upper().str.strip().str.replace(' ', '_')

                        # Mapeo de nombres originales
                        column_rename_map = {
                            'Nº': 'N',
                            'MES_DE_RADICACIÓN': 'MES_DE_RADICACION',
                            'GÉNERO': 'GENERO',
                            'PRODUCTO_QUE_VENDE': 'PRODUCTO_QUE_VENDE',
                            'DETALLE_DEL_PRODUCTO': 'DETALLE_DEL_PRODUCTO',
                            'PERSONA_EN_DISCAPACIDAD': 'PERSONA_EN_DISCAPACIDAD',
                            'CERTIFICADO_REGISTRO_ÚNICO_DE_VÍCTIMAS_CONFLICTO_ARMADO': 'CERTIFICADO_REGISTRO_UNICO_DE_VICTIMAS_CONFLICTO_ARMADO',
                            'CERTIFICADO_RESGUARDO_INDÍGENA': 'CERTIFICADO_RESGUARDO_INDIGENA',
                            'IDENTIFICADO_LGBTIQ+': 'IDENTIFICADO_LGBTIQ+',
                            'PERTENECE_A_ALGUNA_ASOCIACIÓN': 'PERTENECE_A_ALGUNA_ASOCIACION',
                            'MIGRANTE_VENEZOLANO': 'MIGRANTE_VENEZOLANO'
                        }
                        # Aplicar mapeo de nombres. Las columnas no en el mapeo se mantienen como están.
                        self.df = self.df.rename(columns=column_rename_map)

                        # --- FIN DE LA LÓGICA DE LIMPIEZA Y RENOMBRE ESPECÍFICO ---

                        if 'GENERO' in self.df.columns:
                            self.df['GENERO'] = self.df['GENERO'].astype(str).str.upper().str.strip()
                            gender_mapping = {
                                'F': 'FEMENINO', 'FEMENIN': 'FEMENINO', 'MUJER': 'FEMENINO',
                                'M': 'MASCULINO', 'MASCULIN': 'MASCULINO', 'HOMBRE': 'MASCULINO'
                            }
                            self.df['GENERO'] = self.df['GENERO'].replace(gender_mapping)
                            self.df = self.df[self.df['GENERO'].isin(['FEMENINO', 'MASCULINO'])]
                            print("Conteo de GENERO después de limpieza:")
                            print(self.df['GENERO'].value_counts())

                        if 'PRODUCTO_QUE_VENDE' in self.df.columns:
                            self.df['PRODUCTO_QUE_VENDE'] = self.df['PRODUCTO_QUE_VENDE'].astype(str).str.upper().str.strip()
                            product_mapping = {
                                'PRENDAS_DE_VESTIR': 'ROPA',
                                'PRENDAS DE VESTIR': 'ROPA',
                            }
                            self.df['PRODUCTO_QUE_VENDE'] = self.df['PRODUCTO_QUE_VENDE'].replace(product_mapping)
                            top_products = self.df['PRODUCTO_QUE_VENDE'].value_counts().nlargest(5).index
                            self.df.loc[~self.df['PRODUCTO_QUE_VENDE'].isin(top_products), 'PRODUCTO_QUE_VENDE'] = 'OTROS'
                            print("Conteo de PRODUCTO_QUE_VENDE después de limpieza:")
                            print(self.df['PRODUCTO_QUE_VENDE'].value_counts())

                        if 'NACIONALIDAD' in self.df.columns:
                            self.df['NACIONALIDAD'] = self.df['NACIONALIDAD'].astype(str).str.upper().str.strip()
                            nationality_mapping = {
                                'COLOMBIANO': 'COLOMBIANA',
                                'VENEZOLANO': 'VENEZOLANA',
                            }
                            self.df['NACIONALIDAD'] = self.df['NACIONALIDAD'].replace(nationality_mapping)
                            nationality_counts = self.df['NACIONALIDAD'].value_counts()
                            rare_nationalities = nationality_counts[nationality_counts < 3].index
                            self.df.loc[self.df['NACIONALIDAD'].isin(rare_nationalities), 'NACIONALIDAD'] = 'OTRA'
                            print("Conteo de NACIONALIDAD después de limpieza:")
                            print(self.df['NACIONALIDAD'].value_counts())

                        if 'EDAD' in self.df.columns:
                            self.df['EDAD'] = pd.to_numeric(self.df['EDAD'], errors='coerce')
                            self.df['EDAD'] = self.df['EDAD'].fillna(self.df['EDAD'].median())
                            print("Estadísticas de EDAD después de limpieza:")
                            print(self.df['EDAD'].describe())
                        else:
                            print("Advertencia: La columna 'EDAD' no se encontró. No se puede procesar.")

                        # --- CONVERSIÓN DE COLUMNAS BINARIAS A 0/1 ---
                        binary_cols_to_convert = [
                            'PERSONA_EN_DISCAPACIDAD',
                            'CERTIFICADO_REGISTRO_UNICO_DE_VICTIMAS_CONFLICTO_ARMADO',
                            'CERTIFICADO_RESGUARDO_INDIGENA',
                            'IDENTIFICADO_LGBTIQ+',
                            'PERTENECE_A_ALGUNA_ASOCIACION'
                        ]

                        for col in binary_cols_to_convert:
                            if col in self.df.columns:
                                self.df[col] = self.df[col].astype(str).str.upper().str.strip()
                                mapping = {
                                    'SI': 1, 'SÍ': 1, 'VERDADERO': 1, 'TRUE': 1, '1': 1,
                                    'NO': 0, 'FALSO': 0, 'FALSE': 0, '0': 0
                                }
                                self.df[col] = self.df[col].map(mapping).fillna(0).astype(int)
                                print(f"Columna binaria '{col}' convertida a 0/1.")
                            else:
                                print(f"Advertencia: La columna binaria '{col}' no se encontró para conversión.")

                        # --- CREACIÓN DE PUNTUACION_VULNERABILIDAD ---
                        self.df['PUNTUACION_VULNERABILIDAD'] = 0

                        # Regla 1: Adulto Mayor (> 60 años)
                        if 'EDAD' in self.df.columns:
                            self.df['ADULTO_MAYOR'] = (self.df['EDAD'] > 60).astype(int)
                            self.df['PUNTUACION_VULNERABILIDAD'] += self.df['ADULTO_MAYOR']
                            print("Columna 'ADULTO_MAYOR' creada/actualizada y sumada a la puntuación.")
                        else:
                            print("Advertencia: No se puede calcular 'ADULTO_MAYOR' ni sumarlo a la puntuación sin la columna 'EDAD'.")

                        # Regla 2, 3, 4 y otras binarias: Suman 1 punto si la columna binaria es 1
                        vulnerability_contributing_cols = [
                            'PERSONA_EN_DISCAPACIDAD',
                            'CERTIFICADO_REGISTRO_UNICO_DE_VICTIMAS_CONFLICTO_ARMADO',
                            'CERTIFICADO_RESGUARDO_INDIGENA',
                        ]
                        for col in vulnerability_contributing_cols:
                            if col in self.df.columns:
                                self.df['PUNTUACION_VULNERABILIDAD'] += self.df[col]
                                print(f"Columna '{col}' sumada a la puntuación de vulnerabilidad.")
                            else:
                                print(f"Advertencia: La columna '{col}' no se encontró para sumar a la puntuación de vulnerabilidad.")

                        # Columna MIGRANTE_VENEZOLANO basada en NACIONALIDAD (asegurarse de que se cree)
                        if 'NACIONALIDAD' in self.df.columns:
                            self.df['MIGRANTE_VENEZOLANO'] = (self.df['NACIONALIDAD'] == 'VENEZOLANA').astype(int)
                            print("Columna 'MIGRANTE_VENEZOLANO' creada.")

                        # Ahora sí, discretizar PUNTUACION_VULNERABILIDAD en NIVEL_VULNERABILIDAD
                        if 'PUNTUACION_VULNERABILIDAD' in self.df.columns:
                            try:
                                self.df['NIVEL_VULNERABILIDAD'] = pd.qcut(self.df['PUNTUACION_VULNERABILIDAD'], q=2, labels=['Bajo', 'Alto'], duplicates='drop').astype(str)
                                print("NIVEL_VULNERABILIDAD creado/discretizado a partir de PUNTUACION_VULNERABILIDAD.")
                            except ValueError as e:
                                print(f"Advertencia al discretizar con pd.qcut: {e}. Intentando con pd.cut y mediana.")
                                umbral = self.df['PUNTUACION_VULNERABILIDAD'].median()
                                self.df['NIVEL_VULNERABILIDAD'] = pd.cut(self.df['PUNTUACION_VULNERABILIDAD'], bins=[-np.inf, umbral, np.inf], labels=['Bajo', 'Alto'], right=True).astype(str)
                                print(f"NIVEL_VULNERABILIDAD discretizado usando pd.cut (umbral: {umbral}).")

                            print("Conteo final de NIVEL_VULNERABILIDAD después de la discretización:")
                            print(self.df['NIVEL_VULNERABILIDAD'].value_counts())
                        else:
                            print("ERROR CRÍTICO: 'PUNTUACION_VULNERABILIDAD' no fue creada. El modelo no tendrá una variable objetivo.")
                            return False

                        return True

                    except Exception as e:
                        print(f"Intento fallido con separador '{sep}' y codificación '{encoding}': {e}")
            raise Exception("No se pudo cargar el archivo con ninguna de las opciones probadas.")
        except Exception as e:
            print(f"Error al cargar los datos: {e}")
            traceback.print_exc()
            return False

    def preprocesar_datos(self):
        if self.df is None:
            print("Error: No hay datos para preprocesar. Carga los datos primero.")
            return False

        # Codificación de variables categóricas
        for column in ['GENERO', 'PRODUCTO_QUE_VENDE', 'NACIONALIDAD']:
            if column in self.df.columns:
                le = LabelEncoder()
                self.df[column] = le.fit_transform(self.df[column])
                self.label_encoders[column] = le
                print(f"Columna '{column}' codificada.")
            else:
                print(f"Advertencia: La columna '{column}' no se encontró para codificación.")

        # Codificar la variable objetivo
        if 'NIVEL_VULNERABILIDAD' in self.df.columns:
            le_vulnerability = LabelEncoder()
            self.df['NIVEL_VULNERABILIDAD'] = le_vulnerability.fit_transform(self.df['NIVEL_VULNERABILIDAD'])
            self.label_encoders['NIVEL_VULNERABILIDAD'] = le_vulnerability
            print(f"Clases objetivo codificadas: {le_vulnerability.classes_}")
            print(f"Mapeo de clases objetivo: {{'{le_vulnerability.classes_[0]}': {le_vulnerability.transform([le_vulnerability.classes_[0]])[0]}, '{le_vulnerability.classes_[1]}': {le_vulnerability.transform([le_vulnerability.classes_[1]])[0]}}}")
            print("Conteo de clases objetivo después de codificación:")
            print(self.df['NIVEL_VULNERABILIDAD'].value_counts())
        else:
            print("Error: La columna 'NIVEL_VULNERABILIDAD' no se encontró en los datos para la codificación.")
            return False

        # Escalado de la edad
        if 'EDAD' in self.df.columns:
            self.df['EDAD'] = self.scaler.fit_transform(self.df[['EDAD']])
            print("EDAD escalada.")

        print("Datos preprocesados exitosamente.")
        return True

    def entrenar_evaluar_modelo(self, features):
        if self.df is None:
            print("Error: No hay datos para entrenar el modelo. Carga y preprocesa los datos primero.")
            return False

        if not all(f in self.df.columns for f in features):
            missing_features = [f for f in features if f not in self.df.columns]
            print(f"Error: Faltan las siguientes características en los datos: {missing_features}")
            print(f"Columnas disponibles: {self.df.columns.tolist()}")
            return False
        if 'NIVEL_VULNERABILIDAD' not in self.df.columns:
            print("Error: La columna 'NIVEL_VULNERABILIDAD' no se encontró en los datos para el entrenamiento.")
            return False

        X = self.df[features]
        y = self.df['NIVEL_VULNERABILIDAD']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        # Guarda las características usadas para el entrenamiento
        self.features_trained_on = self.X_train.columns.tolist() 

        print("\n--- Conteo de clases en entrenamiento (original) ---")
        print(self.y_train.value_counts())
        print("\n--- Conteo de clases en prueba ---")
        print(self.y_test.value_counts())

        print("Aplicando SMOTE al conjunto de entrenamiento...")
        try:
            # Calcular k_neighbors de forma más robusta
            minority_class_count = self.y_train.value_counts().min()
            # k_neighbors debe ser al menos 1 y menor que el número de muestras de la clase minoritaria
            k_neighbors_val = max(1, min(5, minority_class_count - 1))
            
            if minority_class_count <= 1: # Si solo hay 0 o 1 muestra en la clase minoritaria
                print(f"Advertencia: La clase minoritaria tiene muy pocas muestras ({minority_class_count}). SMOTE puede no ser efectivo o fallar. Saltando SMOTE.")
            else:
                print(f"Usando SMOTE con k_neighbors={k_neighbors_val}.")
                smote = SMOTE(random_state=42, k_neighbors=k_neighbors_val)
                self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
                print("SMOTE aplicado exitosamente.")
                print("\n--- Conteo de clases después de SMOTE ---")
                print(self.y_train.value_counts())
        except ValueError as e:
            print(f"Advertencia al aplicar SMOTE: {e}. No se aplicará SMOTE.")
            traceback.print_exc() # Imprime la traza para depuración


        param_grid = {
            'C': [0.1, 1, 10, 100],
            'penalty': ['l1', 'l2']
        }
        grid_search = GridSearchCV(LogisticRegression(solver='liblinear', random_state=42), param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)

        self.model = grid_search.best_estimator_
        print(f"Mejores hiperparámetros encontrados: {grid_search.best_params_}")
        print(f"Mejor F1-weighted score en validación cruzada: {grid_search.best_score_:.2f}")

        return self.evaluar_modelo()

    def evaluar_modelo(self):
        if self.model is None or self.X_test is None or self.y_test is None:
            print("Error: El modelo no ha sido entrenado o los datos de prueba no están disponibles.")
            return False

        try:
            y_pred = self.model.predict(self.X_test)
            y_pred_proba = self.model.predict_proba(self.X_test)

            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)

            print("\n--- Evaluación del Modelo en el Conjunto de Prueba ---")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision (weighted): {precision:.4f}")
            print(f"Recall (weighted): {recall:.4f}")
            print(f"F1-Score (weighted): {f1:.4f}")

            print("\nClassification Report:")
            target_names = [self.label_encoders['NIVEL_VULNERABILIDAD'].inverse_transform([0])[0],
                            self.label_encoders['NIVEL_VULNERABILIDAD'].inverse_transform([1])[0]]

            print(classification_report(self.y_test, y_pred, target_names=target_names, zero_division=0))

            # Asegúrate de que la clase positiva sea la que se mapea a 1 en el LabelEncoder
            # Si 'Alto' es 0 y 'Bajo' es 1, entonces y_pred_proba[:, 1] es la prob de 'Bajo'
            # y roc_auc_score se calcula para la clase positiva (la que normalmente se asocia con 1)
            auc_roc = roc_auc_score(self.y_test, y_pred_proba[:, 1]) 

            print(f"AUC-ROC: {auc_roc:.4f}")

            cm = confusion_matrix(self.y_test, y_pred)
            print("\nMatriz de Confusión:\n", cm)

            self.generar_matriz_confusion(cm, target_names) 

            self.generar_graficas_analiticas(self.y_test, y_pred_proba[:, 1], target_names) # Pasar y_test también

            return True

        except Exception as e:
            print(f"Error durante la evaluación del modelo: {e}")
            traceback.print_exc()
            return False

    def generar_matriz_confusion(self, cm, class_names):
        """
        Genera y guarda una gráfica de la matriz de confusión.
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Matriz de Confusión')
        plt.ylabel('Etiqueta Verdadera')
        plt.xlabel('Etiqueta Predicha')
        output_dir = './static/reports'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(os.path.join(output_dir, 'matriz_confusion.png'))
        plt.close()

    def generar_graficas_analiticas(self, y_true, y_pred_proba_positive_class, target_names):
        """
        Genera y guarda gráficas analíticas como la curva ROC y la distribución de probabilidades.
        """
        try:
            # Curva ROC
            fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba_positive_class)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='blue', lw=2, label='Curva ROC (área = %0.2f)' % roc_auc_score(y_true, y_pred_proba_positive_class))
            plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Clasificador aleatorio')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Tasa de Falsos Positivos')
            plt.ylabel('Tasa de Verdaderos Positivos')
            plt.title('Curva ROC')
            plt.legend(loc="lower right")
            output_dir = './static/reports'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            plt.savefig(os.path.join(output_dir, 'curva_roc.png'))
            plt.close()
            print("Gráfica de Curva ROC guardada en ./static/reports/curva_roc.png")

            # Histograma de probabilidades (opcional, para visualización)
            # Asegúrate de que las etiquetas de las clases sean correctas para el histograma
            class_0_label = target_names[self.label_encoders['NIVEL_VULNERABILIDAD'].transform([0])[0]] if 0 in self.label_encoders['NIVEL_VULNERABILIDAD'].classes_ else 'Clase 0'
            class_1_label = target_names[self.label_encoders['NIVEL_VULNERABILIDAD'].transform([1])[0]] if 1 in self.label_encoders['NIVEL_VULNERABILIDAD'].classes_ else 'Clase 1'

            plt.figure(figsize=(8, 6))
            sns.histplot(y_pred_proba_positive_class[y_true == 0], color='red', label=f'Clase {class_0_label}', kde=True, stat="density", linewidth=0)
            sns.histplot(y_pred_proba_positive_class[y_true == 1], color='blue', label=f'Clase {class_1_label}', kde=True, stat="density", linewidth=0)
            plt.title('Distribución de Probabilidades Predichas')
            # Asegúrate que el título del eje X refleje qué clase es la "positiva" para el modelo (1)
            plt.xlabel(f'Probabilidad de Clase Positiva ({target_names[1]})') 
            plt.ylabel('Densidad')
            plt.legend()
            plt.savefig(os.path.join(output_dir, 'prob_distribution.png'))
            plt.close()
            print("Gráfica de Distribución de Probabilidades guardada en ./static/reports/prob_distribution.png")

        except Exception as e:
            print(f"Error al generar gráficas analíticas: {e}")
            traceback.print_exc()

    def guardar_modelo(self, filename='modelo_vendedores.pkl'):
        """
        Guarda el modelo entrenado en un archivo.
        filename: Nombre del archivo donde se guardará el modelo.
        """
        if self.model is None or self.scaler is None or not self.label_encoders or self.features_trained_on is None:
            print("Error: El modelo no está completamente entrenado o no se ha preprocesado. No se puede guardar.")
            return False

        try:
            output_dir = './modelos'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            ruta_completa = os.path.join(output_dir, filename)

            with open(ruta_completa, 'wb') as file:
                pickle.dump({
                    'model': self.model,
                    'scaler': self.scaler,
                    'label_encoders': self.label_encoders,
                    'features': self.features_trained_on # Guarda las características usadas
                }, file)
            print(f"Modelo, scaler y encoders guardados exitosamente en {ruta_completa}")
            return True
        except Exception as e:
            print(f"Error al guardar el modelo: {e}")
            traceback.print_exc()
            return False

    def cargar_modelo(self, filename='modelo_vendedores.pkl'):
        """Carga un modelo entrenado desde un archivo.
        filename: Nombre del archivo desde el que se cargará el modelo.
        """
        print(f"DEBUG MODEL: 'cargar_modelo' recibió filename: '{filename}'") # <-- AÑADIR ESTA LÍNEA

        try:
            ruta_completa = os.path.join('./modelos', filename) # Asegúrate de que sea './modelos' aquí
            print(f"DEBUG MODEL: Intentando abrir archivo en ruta_completa: '{ruta_completa}'") # <-- AÑADIR ESTA LÍNEA

            with open(ruta_completa, 'rb') as file:
                data = pickle.load(file)
                self.model = data['model']
                self.scaler = data['scaler']
                self.label_encoders = data['label_encoders']
                self.features_trained_on = data.get('features', None)
                if self.features_trained_on is None:
                    print("Advertencia: Las características de entrenamiento no se cargaron con el modelo.")
            print(f"Modelo cargado exitosamente desde {ruta_completa}")
            return True
        except FileNotFoundError:
            print(f"Error: El archivo del modelo '{ruta_completa}' no se encontró.") # Esta es la línea que genera tu error
            return False
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
            traceback.print_exc()
            return False

    def predecir(self, datos_nuevos):
        if self.model is None or self.scaler is None or not self.label_encoders or self.features_trained_on is None:
            print("Error: El modelo no está entrenado o los preprocesadores no están cargados. Asegúrate de que el modelo haya sido guardado y cargado correctamente con todas sus dependencias.")
            return None, None

        df_pred = pd.DataFrame([datos_nuevos])

        # Asegurar que las columnas de df_pred coincidan con las de self.features_trained_on
        for col_name in self.features_trained_on:
            if col_name not in df_pred.columns:
                # Añadir columna si falta, con valor por defecto 0
                df_pred[col_name] = 0
            # Convertir a numérico si es necesario y asegurar que sea int para columnas binarias
            if col_name in ['ADULTO_MAYOR', 'PERSONA_EN_DISCAPACIDAD', 'CERTIFICADO_REGISTRO_UNICO_DE_VICTIMAS_CONFLICTO_ARMADO',
                            'CERTIFICADO_RESGUARDO_INDIGENA', 'IDENTIFICADO_LGBTIQ+', 'PERTENECE_A_ALGUNA_ASOCIACION',
                            'MIGRANTE_VENEZOLANO']:
                # Asegúrate de que el valor exista antes de intentar convertirlo
                if col_name in df_pred.columns and pd.notna(df_pred[col_name].iloc[0]):
                    df_pred[col_name] = pd.to_numeric(df_pred[col_name], errors='coerce').fillna(0).astype(int)
                else: # Si el dato no se proporcionó o es NaN, asigna 0
                    df_pred[col_name] = 0


        # Codificar variables categóricas (GENERO, PRODUCTO_QUE_VENDE, NACIONALIDAD)
        for column in ['GENERO', 'PRODUCTO_QUE_VENDE', 'NACIONALIDAD']:
            if column in self.features_trained_on: # Solo procesa si la columna era una feature de entrenamiento
                if column in df_pred.columns and column in self.label_encoders:
                    le = self.label_encoders[column]
                    # Estandarizar el valor de entrada para la codificación
                    input_value = str(datos_nuevos.get(column, '')).upper().strip()
                    
                    if input_value in le.classes_:
                        df_pred[column] = le.transform([input_value])[0]
                    else:
                        print(f"Advertencia: Valor desconocido '{datos_nuevos.get(column)}' para '{column}'. Usando valor por defecto (0).")
                        df_pred[column] = 0 # Valor por defecto para categorías no vistas
                else: # Si la columna era una feature pero no está en el input o no tiene encoder, asigna 0
                    df_pred[column] = 0


        # Escalado de la edad
        if 'EDAD' in self.features_trained_on:
            if 'EDAD' in df_pred.columns:
                try:
                    # Asegurarse de que EDAD sea numérico antes de escalar
                    df_pred['EDAD'] = pd.to_numeric(df_pred['EDAD'], errors='coerce')
                    df_pred['EDAD'] = self.scaler.transform(df_pred[['EDAD']].fillna(self.scaler.mean_[0])) # Usa la media del scaler para NA
                except Exception as e:
                    print(f"Error al escalar EDAD: {e}. Asegúrate que EDAD sea un número. Asignando 0.")
                    df_pred['EDAD'] = 0
            else: # Si EDAD era feature pero no está en el input, asigna 0 (o la media escalada)
                df_pred['EDAD'] = 0


        # Reordenar las columnas para que coincidan con el orden de entrenamiento
        df_pred = df_pred[self.features_trained_on]

        try:
            prediccion_codificada = self.model.predict(df_pred)[0]
            probabilidades = self.model.predict_proba(df_pred)[0]

            prediccion_decodificada = "Desconocido"
            probabilidades_dict = {}

            if 'NIVEL_VULNERABILIDAD' in self.label_encoders:
                # Asegurarse de que la predicción codificada esté dentro de los límites del encoder
                if prediccion_codificada in range(len(self.label_encoders['NIVEL_VULNERABILIDAD'].classes_)):
                    prediccion_decodificada = self.label_encoders['NIVEL_VULNERABILIDAD'].inverse_transform([prediccion_codificada])[0]
                else:
                    print(f"Advertencia: Predicción codificada fuera de rango del LabelEncoder para NIVEL_VULNERABILIDAD: {prediccion_codificada}")
                    # Podrías asignar una clase por defecto o la clase mayoritaria
                    prediccion_decodificada = "Clase Indefinida" # O una clase por defecto adecuada

                for i, class_label in enumerate(self.label_encoders['NIVEL_VULNERABILIDAD'].classes_):
                    probabilidades_dict[class_label] = probabilidades[i]
            else:
                for i, prob in enumerate(probabilidades):
                    probabilidades_dict[f'Clase {i}'] = prob

            return prediccion_decodificada, probabilidades_dict

        except Exception as e:
            print(f"Error durante la predicción final del modelo: {e}")
            traceback.print_exc()
            return None, None

    def obtener_opciones_validas(self, columna):
        if columna in self.label_encoders:
            return self.label_encoders[columna].classes_.tolist()
        elif columna == 'EDAD':
            return "Número entre 18 y 90" # Esto es solo una guía, el modelo acepta cualquier número preprocesado
        # Asegúrate de que estos nombres coincidan con los nombres estandarizados en cargar_datos
        elif columna in ['ADULTO_MAYOR', 'PERSONA_EN_DISCAPACIDAD',
                         'CERTIFICADO_REGISTRO_UNICO_DE_VICTIMAS_CONFLICTO_ARMADO',
                         'CERTIFICADO_RESGUARDO_INDIGENA', 'IDENTIFICADO_LGBTIQ+',
                         'PERTENECE_A_ALGUNA_ASOCIACION', 'MIGRANTE_VENEZOLANO']:
            return ['0', '1']
        else:
            return [] # Retorna una lista vacía si no se encuentran opciones