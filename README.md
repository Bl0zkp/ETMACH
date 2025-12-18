# ETMACH
repositorio y versiones de examen Machine learning

#Descripción del proyecto

Este proyecto implementa un **pipeline completo de Machine Learning** orientado a la **evaluación de riesgo crediticio**, cubriendo desde la preparación de los datos hasta el despliegue de un modelo mediante una **API REST**.

El flujo del proyecto permite:

* Organizar y preparar los datos desde archivos `.parquet`.
* Construir un dataset unificado y generar archivos auxiliares (como JSON).
* Entrenar un modelo de Machine Learning y evaluar sus métricas.
* Analizar la distribución de decisiones del modelo.
* Exponer el modelo entrenado a través de una API utilizando **FastAPI** y **Uvicorn**.

El repositorio está estructurado para facilitar la reproducibilidad del proceso completo, desde los datos crudos hasta la inferencia en producción.

---

#  Estructura del proyecto

```
home-credit-risk/
│
├── data/
│   └── raw/                # Datos originales en formato parquet
│
├── 02_data_preparation/
│   └── build_dataset.py    # Construcción del dataset y generación de JSON
│
├── 03_modeling/
│   └── train.py            # Entrenamiento del modelo y métricas de validación
│
├── 04_evaluation/
│   └── evaluate.py         # Evaluación del modelo y análisis de decisiones
│
├── 05_deployment/
│   └── app.py              # API para exponer el modelo
│
└── README.md
```

---

# Ejecución del proyecto

# Configuración inicial

Se establece la raíz del proyecto y se ajusta el `PYTHONPATH` para asegurar que los módulos puedan ser encontrados correctamente.

Además, se crea la carpeta `data/raw` (solo si no existe) y se copian los archivos `.parquet` necesarios para el procesamiento.

---

### Construcción del dataset

Se ejecuta el script de preparación de datos, el cual se encarga de:

* Unir las tablas necesarias.
* Limpiar y transformar los datos.
* Generar archivos auxiliares como JSON para el modelado.

```bash
python 02_data_preparation/build_dataset.py
```

---

###  Entrenamiento del modelo

Se entrena el modelo de Machine Learning y se calculan las métricas de validación.

```bash
python 03_modeling/train.py
```

---

###Evaluación del modelo

Se analizan las métricas obtenidas y la distribución de las decisiones del modelo.

```bash
python 04_evaluation/evaluate.py
```

---

###  Despliegue de la API

El modelo entrenado se expone a través de una API REST usando **Uvicorn**.

```bash
uvicorn 05_deployment.app:app --reload
```

También es posible ejecutar el servidor en segundo plano para realizar pruebas sin bloquear la sesión.

---

## Pruebas de la API

### Health Check

Verifica que el servicio esté activo:

```bash
curl http://127.0.0.1:8000/health
```

---

### Evaluación de riesgo

Ejemplo de petición para evaluar el riesgo crediticio:

```python
import requests

url = "http://127.0.0.1:8000/evaluate_risk"

payload = {
    "data": {
        "AMT_INCOME_TOTAL": 150000,
        "DAYS_BIRTH": -12000,
        "DAYS_EMPLOYED": -2000
    }
}

response = requests.post(url, json=payload)
print(response.status_code)
print(response.json())
```

---

## Tecnologías utilizadas

* Python
* Pandas / NumPy
* Scikit-learn (u otra librería de ML)
* FastAPI
* Uvicorn

---

## Objetivo final

Proveer una solución completa y reproducible para la **evaluación automatizada de riesgo crediticio**, integrando buenas prácticas de preparación de datos, modelado, evaluación y despliegue en un entorno productivo.
