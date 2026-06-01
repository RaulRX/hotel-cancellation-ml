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
│
├── data/
│   ├── raw/                    # Dataset original sin modificar (dataset.csv)
│   └── processed/              # Datos tras limpieza y transformación
│
├── models/
│   └── tests/                  # Modelos entrenados serializados (.pkl)
│
├── notebooks/
│   ├── exploration/            # Notebooks de análisis exploratorio inicial
│   └── final/                  # Notebooks finales con resultados y comparativa
│
├── outputs/                    # Gráficos y evidencias generados (PNG, HTML)
│
└── src/                        # Código fuente del pipeline
    ├── __init__.py
    ├── config.py               # Parámetros y configuración global
    ├── data_loader.py          # Carga y validación del dataset
    ├── preprocess_data.py      # Limpieza y transformación de datos
    ├── trainer.py              # Entrenamiento y comparación de modelos
    ├── evaluator.py            # Métricas, visualizaciones y selección del mejor modelo
    └── predict.py              # Inferencia con el modelo seleccionado
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

### Opción 1 — Ejecución local (entorno virtual)

> **Requerido por el entorno de evaluación.** Asegúrate de tener Python 3.11 instalado.

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
python -m src.data_loader        # Carga y validación del dataset
python -m src.preprocess_data    # Limpieza y preprocesamiento
python -m src.trainer            # Entrenamiento de los 5 modelos
python -m src.evaluator          # Evaluación, métricas y selección del mejor modelo
python -m src.predict            # Inferencia con el modelo seleccionado
```

Los artefactos generados (modelos serializados, gráficos, reportes) quedan en `models/` y `outputs/`.

---

### Opción 2 — Ejecución vía GitHub Actions

El flujo completo está automatizado mediante un workflow de GitHub Actions. Los pasos que ejecuta:

1. Descarga de datos
2. Preprocesamiento de datos
3. Entrenamiento de modelos
4. Evaluación de modelos
5. Generación de evidencias
6. Generación de tag con release

Para lanzarlo, accede a la pestaña **Actions** del repositorio y ejecuta el workflow `[TODO: nombre del workflow]`.

---

### Opción 3 — Ejecución vía API FastAPI (bonus)

La API (`Hotel Cancellation ML API v0.1.0`) expone los endpoints principales del pipeline. Documentación interactiva disponible en `http://localhost:8000/docs` (Swagger UI) una vez arrancado el servidor.

```bash
# Arrancar el servidor
uvicorn api.main:app --reload --port 8000
```

#### Endpoints

| Endpoint | Método | Body | Descripción |
|---|---|---|---|
| `/train` | `POST` | Empty | Entrena el modelo y lo guarda en `models/best_model.pkl` |
| `/predict` | `POST` | `features` | Carga el `.pkl` y devuelve predicciones para las features recibidas |
| `/evaluate` | `POST` | `dataset_path` (opcional) | Evalúa el modelo entrenado con el dataset en disco |

El flujo esperado es: **`/train` → `/predict` → `/evaluate`**.

---

**`POST /train`**

No requiere cuerpo. Lee el CSV del dataset local, entrena el pipeline completo y serializa el mejor modelo en `models/best_model.pkl`.

```bash
curl -X POST http://localhost:8000/train
```

Respuesta exitosa (`200`):
```json
{
  "status": "success",
  "message": "Modelo entrenado y guardado en archivo pkl."
}
```

Errores: `404` si no se encuentra el CSV, `500` para cualquier otro error.

---

**`POST /predict`**

Recibe una o varias reservas con sus datos originales (tal como llegan en el momento de la reserva, sin data leakage) y devuelve la predicción de cancelación. Carga internamente el Pipeline completo serializado en `models/best_model.pkl` (preprocesado + modelo).

Los campos excluidos respecto al dataset original son los que solo se conocen _después_ de la reserva: `reservation_status`, `reservation_status_date`, `assigned_room_type`, `booking_changes` e `is_canceled`.

```json
{
  "records": [
    {
      "hotel": "City Hotel",
      "lead_time": 120,
      "arrival_date_year": 2025,
      "arrival_date_month": "July",
      "arrival_date_week_number": 27,
      "arrival_date_day_of_month": 1,
      "stays_in_weekend_nights": 1,
      "stays_in_week_nights": 3,
      "adults": 2,
      "children": 0,
      "babies": 0,
      "meal": "BB",
      "country": "ESP",
      "market_segment": "Online TA",
      "distribution_channel": "TA/TO",
      "is_repeated_guest": 0,
      "previous_cancellations": 0,
      "previous_bookings_not_canceled": 0,
      "reserved_room_type": "A",
      "deposit_type": "No Deposit",
      "agent": "9",
      "company": null,
      "days_in_waiting_list": 0,
      "customer_type": "Transient",
      "adr": 150.5,
      "required_car_parking_spaces": 0,
      "total_of_special_requests": 1
    }
  ]
}
```

Respuesta exitosa (`200`):
```json
{
  "predictions": [0]
}
```

`0` = no cancela, `1` = cancela.

Errores: `400` si el modelo no ha sido entrenado aún, `500` para errores de inferencia.

---

**`POST /evaluate`**

Evalúa el modelo entrenado contra el dataset en disco. Llama internamente a `src.evaluator.evaluate_all`.

```json
{
  "dataset_path": "data/raw/dataset.csv"
}
```

El campo `dataset_path` es opcional — si se omite, usa la ruta por defecto configurada en el proyecto.

Errores: `404` si no se encuentra el dataset, `500` para cualquier otro error.

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

**Métrica principal**: F1-score

**Justificación**: En este problema, los dos tipos de error tienen un coste significativo pero simétrico:

- Un **falso negativo** (predecir que el cliente *no* cancela cuando sí lo hará) significa que el hotel no tomará medidas preventivas — no ofrecerá descuentos ni reasignará la habitación a tiempo. En periodos de baja ocupación, esto puede traducirse en ingresos perdidos difíciles de recuperar.
- Un **falso positivo** (predecir cancelación cuando el cliente *no* va a cancelar) lleva a ofrecer descuentos o incentivos innecesarios. En temporada alta, donde la demanda es alta, esto supone pérdidas directas y potenciales reclamaciones de clientes que pagaron tarifa completa.

Dado que ninguno de los dos errores es claramente más costoso que el otro — el impacto depende de la temporada y la ocupación del hotel —, se busca un **equilibrio entre precision y recall**. El **F1-score**, al ser la media armónica de ambas métricas, penaliza por igual los falsos positivos y los falsos negativos, lo que lo convierte en la métrica más adecuada para este caso de uso.

Además de la métrica principal se reportan para cada modelo:

- **Accuracy**, **Precision**, **Recall**, **F1-score**, **AUC-ROC**
- Matriz de confusión
- Curva ROC comparativa entre modelos
- Importancia de variables (feature importance / SHAP)

---

## Resultados y conclusiones

### Comparativa de modelos

| Modelo | Accuracy | F1-score | ROC-AUC |
|---|---|---|---|
| Logistic Regression | [TODO] | [TODO] | [TODO] |
| Decision Tree | [TODO] | [TODO] | [TODO] |
| Random Forest | [TODO] | [TODO] | [TODO] |
| LightGBM | [TODO] | [TODO] | [TODO] |
| Deep Neural Network (Keras) | [TODO] | [TODO] | [TODO] |

### Modelo seleccionado

`[TODO: nombre del modelo ganador]` — seleccionado por obtener el mejor valor de `[TODO: métrica principal]` con un valor de `[TODO: valor]`.

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
