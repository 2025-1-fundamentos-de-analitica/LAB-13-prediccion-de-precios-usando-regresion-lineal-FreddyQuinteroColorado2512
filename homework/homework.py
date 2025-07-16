#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#
import pandas as pd
import gzip
import zipfile
import json
import pickle
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import median_absolute_error, mean_squared_error, r2_score

def read_csv_from_zip(zip_path):
    with zipfile.ZipFile(zip_path, "r") as z:
        file_name = z.namelist()[0]
        with z.open(file_name) as f:
            return pd.read_csv(f)
        
def clean_data(data):
    """Limpieza de los datasets."""
    data["Age"] = 2021 - data["Year"]
    data.drop(columns=["Year", "Car_Name"], inplace=True)
    return data

def regression_model(x_train):
    """Creación un pipeline para el modelo de clasificación."""
    
    categorical_features = ["Fuel_Type", "Selling_type", "Transmission"]
    numerical_features = [ "Selling_Price", "Driven_kms", "Owner", "Age"]
  

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('scaler', MinMaxScaler(), numerical_features),
        ])
    
    classifier = LinearRegression()
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('feature_selection', SelectKBest(score_func=f_regression)), 
        ('classifier', classifier )
    ])

    return model


def optimize_hyperparameters(model, x_train, y_train):
    """Optimización los hiperparametros del pipeline usando validación cruzada."""

    param_grid = {
        "feature_selection__k": range(1,12), 

              
    }
    search = GridSearchCV(model, param_grid, n_jobs=-1, cv=10, scoring="neg_mean_absolute_error", error_score="raise", refit=True )
    search.fit(x_train, y_train)

    return search


def save_model(model):
    """Guardar el modelo (comprimido con gzip)."""

    with gzip.open("files/models/model.pkl.gz", "wb") as f:
        pickle.dump(model, f)

def metrics(model, x_train, y_train, x_test, y_test):
    """Calcula métricas de regresión para evaluar el modelo."""
    
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    metrics_train = {
        "type": "metrics",
        "dataset": "train",
        "r2": float(r2_score(y_train, y_train_pred)),
        "mse": float(mean_squared_error(y_train, y_train_pred)),
        "mad": float(median_absolute_error(y_train, y_train)),
    }

    metrics_test = {
        "type": "metrics",
        "dataset": "test",
        "r2": float(r2_score(y_test, y_test_pred)),
        "mse": float(mean_squared_error(y_test, y_test_pred)),
        "mad": float(median_absolute_error(y_test, y_test_pred))
    }

    return metrics_train, metrics_test



def save_metrics(metrics_train, metrics_test,  file_path="files/output/metrics.json"):
    """Guarda las métricas en un archivo JSON"""
    metrics_data = [metrics_train, metrics_test]

    with open(file_path, "w") as f:
        for item in metrics_data:
            f.write(json.dumps(item) + "\n")

def ensure_directories():
    """Asegura que los directorios de salida existen."""
    import os
    os.makedirs("files/models", exist_ok=True)
    os.makedirs("files/output", exist_ok=True)

ensure_directories()

train_zip_path = "files/input/train_data.csv.zip"
test_zip_path = "files/input/test_data.csv.zip"
       
train_data = clean_data(read_csv_from_zip(train_zip_path))
test_data = clean_data(read_csv_from_zip(test_zip_path))

x_train = train_data.drop(columns=["Present_Price"])  
y_train = train_data["Present_Price"]

x_test = test_data.drop(columns=["Present_Price"])
y_test = test_data["Present_Price"]



model = regression_model(x_train)
best_model = optimize_hyperparameters(model, x_train, y_train)
save_model(best_model)

metrics_train, metrics_test = metrics(best_model, x_train, y_train, x_test, y_test)
save_metrics(metrics_train, metrics_test)
