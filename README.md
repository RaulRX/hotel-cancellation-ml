# Proyecto de evaluación de modelos de clasificación binaria

Repositorio: <https://github.com/RaulRX/hotel-cancellation-ml.git>

---

## Autores del proyecto

| Autor | Perfil | Rol principal |
|---|---|---|
| **Angel Pérez Izquierdo** | [LinkedIn](https://www.linkedin.com/in/anpeiz) | MLOps Engineer / Ingeniero IA (Random Forest, Red Neuronal) |
| **Raúl Sánchez Serrano** | [LinkedIn](https://www.linkedin.com/in/raulsanchezserrano) | MLOps Engineer / Ingeniero IA (Regresión Logística, Árbol de Decisión, Gradient Boosting) |

> Detalle completo de roles y contribuciones en [FINAL_REPORT.md](./FINAL_REPORT.md).

---

## Descripción del problema y datos

El objetivo del proyecto es **predecir si un cliente va a cancelar su reserva hotelera** (clasificación binaria). Anticipar las cancelaciones permite a los hoteles optimizar la asignación de recursos, gestionar el overbooking y mejorar los ingresos.

### Dataset

- **Nombre**: Hotel Booking Demand Dataset
- **Fuente**: [Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)
- **Ubicación**: `data/raw/dataset.csv`
- **Dimensiones**: ~119.390 reservas × 32 variables
- **Variable objetivo**: `is_canceled` (binaria: `1` = cancelada, `0` = no cancelada)
- **Hoteles incluidos**: Resort Hotel y City Hotel (misma cadena hotelera, período de dos años)

### Tipología de variables

| Tipo | Ejemplos |
|---|---|
| Numéricas | `lead_time`, `adr`, `stays_in_week_nights`, `previous_cancellations` |
| Categóricas | `hotel`, `meal`, `country`, `market_segment`, `deposit_type`, `customer_type` |
| Temporales | `arrival_date_year`, `arrival_date_month`, `arrival_date_day_of_month` |
| Binarias | `is_repeated_guest`, `is_canceled` |

> Análisis exploratorio completo (distribuciones, correlaciones, tratamiento de nulos) en [FINAL_REPORT.md](./FINAL_REPORT.md).

---

## Estructura del proyecto

```
hotel-cancellation-ml/
├── .gitignore                  # Archivos excluidos del repositorio
├── requirements.txt            # Dependencias del proyecto
├── README.md                   # Documentación principal
├── FINAL_REPORT.md             # Informe final (EDA, diseño, resultados, reflexión)
├── Dockerfile                  # Imagen Docker de la aplicación
├── docker-compose.yaml         # Orquestación del contenedor con volúmenes
│
├── data/
│   ├── raw/                    # Dataset original sin modificar (dataset.csv)
│   └── processed/              # Datos tras limpieza y transformación
│
├── models/
│   ├── best_model.pkl          # Mejor modelo serializado (generado por /evaluate)
│   └── tests/                  # Modelos candidatos serializados tras /train
│
├── notebooks/
│   ├── exploration/            # Notebooks de análisis exploratorio inicial
│   └── final/                  # Notebooks finales con resultados y comparativa
│
├── outputs/                    # Gráficos, métricas y predicciones generados
│
└── src/                        # Código fuente del pipeline
    ├── __init__.py
    ├── config.py               # Parámetros y configuración global
    ├── data_loader.py          # Carga y validación del dataset
    ├── preprocess_data.py      # Limpieza, transformación y constructores de pipelines
    ├── trainer.py              # Entrenamiento de los modelos candidatos
    ├── evaluator.py            # Métricas, visualizaciones y selección del mejor modelo
    ├── predict.py              # Inferencia batch con el modelo seleccionado
    └── api/                    # API REST (FastAPI)
        ├── main.py             # Entrada de la aplicación y registro de routers
        └── routers/
            ├── best_model.py   # Router GET /best-model
            ├── train.py        # Router POST /train
            ├── evaluate.py     # Router POST /evaluate
            └── predict.py      # Router POST /predict
```

---

## Requisitos del entorno

- **Python**: 3.11
- **Gestor de dependencias**: pip + virtualenv

### Librerías principales

| Librería | Uso |
|---|---|
| `pandas`, `numpy` | Manipulación y transformación de datos |
| `scikit-learn` | Preprocesamiento, regresión logística, árbol de decisión, random forest, métricas |
| `lightgbm` | Gradient Boosting (LightGBM) |
| `tensorflow` / `keras` | Red neuronal multicapa |
| `matplotlib`, `seaborn` | Visualizaciones (curvas ROC, matrices de confusión) |
| `fastapi`, `uvicorn` | API REST para exposición de endpoints |
| `joblib` | Serialización de modelos |

El detalle completo de versiones se encuentra en `requirements.txt`.

---

## Instrucciones para ejecutar el proyecto

### Opción 1 — Ejecución vía CLI (pipeline completo)

> Asegúrate de tener Python 3.11 instalado.

Un único comando ejecuta el pipeline de extremo a extremo: reprocesa el dato crudo, entrena todos los modelos, los evalúa, selecciona el ganador por F1-score y escribe `models/best_model.pkl`.

```bash
# 1. Clonar el repositorio
git clone https://github.com/RaulRX/hotel-cancellation-ml.git
cd hotel-cancellation-ml

# 2. Crear y activar el entorno virtual
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar el pipeline completo
python -m src.trainer
```

Al terminar, la consola imprime una tabla comparativa de métricas con el modelo ganador marcado con `(*)`. Los artefactos quedan en `models/` y `outputs/`.

---

### Opción 2 — Ejecución vía API local (FastAPI)

Mismos pasos de instalación que la Opción 1. En lugar de lanzar el pipeline directamente, arranca el servidor y controla cada fase mediante llamadas HTTP.

```bash
# Arrancar el servidor
uvicorn src.api.main:app --reload --port 8000
```

Documentación interactiva disponible en `http://localhost:8000/docs` (Swagger UI).

El flujo de llamadas es: **`POST /train` → `POST /predict` → `POST /evaluate`** (ver sección [Endpoints de la API](#endpoints-de-la-api)).

---

### Opción 3 — Ejecución con Docker Compose

> Requiere Docker y Docker Compose instalados.

```bash
# 1. Clonar el repositorio
git clone https://github.com/RaulRX/hotel-cancellation-ml.git
cd hotel-cancellation-ml

# 2. Construir la imagen y levantar el contenedor
docker compose up --build
```

El servidor queda disponible en `http://localhost:8000`. Los directorios `data/`, `models/` y `outputs/` se montan como volúmenes, por lo que los artefactos persisten en el host tras apagar el contenedor.

```bash
# Parar y eliminar el contenedor
docker compose down
```

---

### Opción 4 — Ejecución vía GitHub Actions

El flujo completo está automatizado mediante un workflow de GitHub Actions. Los pasos que ejecuta:

1. Descarga de datos
2. Compilación de la imagen Docker y ejecución del contenedor

Para lanzarlo, accede a la pestaña **Actions** del repositorio y ejecuta el workflow correspondiente.

---

### Endpoints de la API

| Endpoint | Método | Body | Descripción |
|---|---|---|---|
| `GET /best-model` | `GET` | — | Ejecuta el pipeline completo (train + evaluate) y devuelve el mejor modelo |
| `POST /train` | `POST` | JSON opcional con `hyperparams` | Entrena todos los modelos candidatos y los serializa en `models/tests/` |
| `POST /evaluate` | `POST` | — | Evalúa los modelos entrenados, genera métricas y selecciona el mejor en `models/best_model.pkl` |
| `POST /predict` | `POST` | — | Ejecuta inferencia batch con el mejor modelo sobre el dataset y guarda resultados en `outputs/predictions.json` |

**Flujo paso a paso**: **`POST /train` → `POST /predict` → `POST /evaluate`**

**Flujo completo en un solo paso**: **`GET /best-model`** (equivale a ejecutar train + predict + evaluate de forma encadenada)

---

**`GET /best-model`**

Ejecuta el pipeline completo de forma encadenada (train → evaluate) en una sola llamada. Equivale a llamar a `POST /train` seguido de `POST /evaluate`. Útil para obtener el mejor modelo sin gestionar el flujo manualmente.

```bash
curl http://localhost:8000/best-model
```

No requiere cuerpo. Llama internamente a `src.trainer.train_all`.

Respuesta exitosa (`200`) — ejemplo:

```json
{
  "status": "success",
  "best_model": "lightgbm",
  "metrics": {
    "logistic_regression": {"accuracy": 0.80, "f1": 0.78, "roc_auc": 0.86},
    "decision_tree":        {"accuracy": 0.83, "f1": 0.81, "roc_auc": 0.88},
    "lightgbm":             {"accuracy": 0.89, "f1": 0.87, "roc_auc": 0.95}
  }
}
```

Errores: `404` si no se encuentra el CSV, `500` para cualquier otro error.

---

**`POST /train`**

Carga el dataset, limpia los datos (o reutiliza los datos preprocesados si ya existen) y entrena todos los modelos candidatos. Los pipelines serializados (preprocesado + modelo) se guardan en `models/tests/`. Permite pasar hiperparámetros opcionales por modelo.

```bash
# Sin hiperparámetros (usa los valores por defecto)
curl -X POST http://localhost:8000/train

# Con hiperparámetros personalizados
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{
    "hyperparams": {
      "lightgbm": {"num_leaves": 80, "n_estimators": 500},
      "decision_tree": {"max_depth": 6}
    }
  }'
```

Body (todos los campos son opcionales):

```json
{
  "hyperparams": {
    "logistic_regression": {},
    "decision_tree": {"max_depth": 6},
    "lightgbm": {"num_leaves": 80, "n_estimators": 500}
  }
}
```

Respuesta exitosa (`200`):

```json
{
  "status": "success",
  "models_trained": ["logistic_regression", "decision_tree", "lightgbm"],
  "models_dir": "models/tests"
}
```

Errores: `404` si no se encuentra el CSV, `500` para cualquier otro error.

---

**`POST /evaluate`**

Evalúa todos los modelos guardados en `models/tests/`, compara sus métricas (F1-score como métrica principal), genera visualizaciones en `outputs/` y copia el mejor modelo a `models/best_model.pkl`.

```bash
curl -X POST http://localhost:8000/evaluate
```

No requiere cuerpo. Llama internamente a `src.evaluator.evaluate_all`.

Respuesta exitosa (`200`) — ejemplo:

```json
{
  "best_model": "lightgbm",
  "metrics": {
    "logistic_regression": {"accuracy": 0.80, "f1": 0.78, "roc_auc": 0.86},
    "decision_tree":        {"accuracy": 0.83, "f1": 0.81, "roc_auc": 0.88},
    "lightgbm":             {"accuracy": 0.89, "f1": 0.87, "roc_auc": 0.95}
  }
}
```

Errores: `404` si no hay modelos entrenados, `500` para cualquier otro error.

---

**`POST /predict`**

Carga el mejor modelo serializado (`models/best_model.pkl`) y ejecuta inferencia batch sobre el dataset. Las predicciones se persisten en `outputs/predictions.json`.

```bash
curl -X POST http://localhost:8000/predict
```

No requiere cuerpo. La ejecución es asíncrona internamente para no bloquear el event loop.

Respuesta exitosa (`200`):

```json
{
  "status": "ok",
  "message": "Predictions saved to outputs/predictions.json"
}
```

Errores: `404` si no existe `best_model.pkl` (ejecuta `/evaluate` primero), `500` para errores de inferencia.

---

## Modelos evaluados

El pipeline entrena y compara los siguientes algoritmos de clasificación binaria:

| # | Modelo | Librería |
|---|---|---|
| 1 | Regresión Logística | scikit-learn |
| 2 | Árbol de Decisión | scikit-learn |
| 3 | Random Forest | scikit-learn |
| 4 | Gradient Boosting (LightGBM) | lightgbm |
| 5 | Red Neuronal Multicapa | tensorflow / keras |

---

## Métrica principal y evaluación

**Métrica principal**: Accuracy

**Justificación**: La accuracy mide la proporción de predicciones correctas sobre el total de casos. Es la métrica más directa e interpretable en clasificación: un modelo con accuracy del 85 % acierta 85 de cada 100 reservas, sin necesidad de conocer conceptos adicionales para leer el resultado.

Su elección como métrica de referencia se apoya en tres razones:

1. **Interpretabilidad universal**: cualquier stakeholder no técnico entiende inmediatamente qué significa "el modelo acierta el X % de los casos". Esto facilita la comunicación de resultados y la toma de decisiones.
2. **Visión global del rendimiento**: agrega en un único número el comportamiento del modelo sobre todas las clases, lo que permite comparar modelos distintos de forma rápida y objetiva durante la fase de selección.
3. **Punto de partida sólido**: en proyectos donde se exploran varios algoritmos en paralelo, la accuracy actúa como criba inicial. Los modelos que no superan un umbral mínimo de accuracy quedan descartados antes de analizar métricas más finas.

Además de la accuracy, se reportan métricas complementarias que cubren aspectos que esta no captura por sí sola: Precision, Recall, F1-score y AUC-ROC.

Para cada modelo se reportan:

- **Accuracy**, **Precision**, **Recall**, **F1-score**, **AUC-ROC**
- Matriz de confusión
- Curva ROC comparativa entre modelos
- Importancia de variables (feature importance / SHAP)

---

## Resultados y conclusiones

### Comparativa de modelos

| Modelo | Accuracy | F1-score | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 76.99% | 66.14% | 86.12% |
| Decision Tree | 72.06% | 62.74% | 85.53% |
| Random Forest | 80.91% | 57.43% | 87.16% |
| LightGBM | **84.08%** | **69.69%** | **90.36%** |
| Deep Neural Network (Keras) | 79.85% | 62.36% | 73.89% |

### Modelo seleccionado

**LightGBM** — seleccionado por obtener la mejor accuracy con un valor de **84.08%**, liderando también en F1-score (69.69%) y AUC-ROC (90.36%).

### Conclusiones

`[TODO: resumen de los hallazgos más relevantes: qué modelo funciona mejor y por qué, qué variables resultan más importantes, qué limitaciones se han encontrado]`

> Detalle completo de análisis, diseño del sistema y reflexión crítica en [FINAL_REPORT.md](./FINAL_REPORT.md).

---

## Documentación adicional

El informe final del proyecto se encuentra en [FINAL_REPORT.md](./FINAL_REPORT.md) e incluye:

- Definición de roles de la pareja
- Justificación del problema
- Análisis exploratorio de datos (EDA)
- Diseño del sistema
- Resultados y elección final del modelo
- Reflexión crítica sobre limitaciones y mejoras
