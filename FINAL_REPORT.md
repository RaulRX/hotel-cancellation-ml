# Informe final de evaluación de modelos de clasificación binaria

## Definición de roles

* **Angel Pérez Izquierdo**
  * *MLOps Engineer*:
    * Responsable de la automatización del flujo de evaluación de modelos.
  * *Ingeniero IA*:
    * Implementación del entrenamiento de los algoritmos:
      * Random Forest
      * Red Neuronal multicapa con Keras
    * Evaluación de los modelos
    * Generación de evidencias

* **Raúl Sánchez Serrano**
  * *MLOps Engineer*:
    * Responsable de la automatización del flujo de evaluación de modelos.
  * *Ingeniero IA*:
    * Implementación del entrenamiento de los algoritmos:
      * Regresión Logística
      * Arbol de decisión
      * Gradient Boosting
    * Evaluación de los modelos
    * Generación de evidencias

## Justificación del problema

El problema seleccionado para este proyecto de evaluación de modelos de clasificación binaria es **predecir la cancelación de reservas hoteleras**. La elección de este problema se justifica por varias razones:

* **Relevancia empresarial**: La cancelación de reservas es un problema común en la industria hotelera que genera pérdidas económicas significativas. La capacidad de anticipar estas cancelaciones puede ayudar a los hoteles a optimizar sus ingresos y recursos.
* **Disponibilidad de datos**: Existen datasets públicos disponibles para este problema, como el "Hotel Booking Demand Dataset" de Kaggle, que permite realizar un análisis completo y comparativo de diferentes modelos.
* **Complejidad interesante**: El problema presenta una combinación de variables numéricas y categóricas, lo que permite explorar diferentes técnicas de preprocesamiento y evaluación de modelos.
* **Aplicabilidad de técnicas avanzadas**: Este problema permite aplicar técnicas de aprendizaje automático supervisado, incluyendo modelos tradicionales como la regresión logística y árboles de decisión, así como modelos más avanzados como Random Forest, Gradient Boosting y redes neuronales artificiales.

## Análisis exploratorio de datos

## Diseño del sistema

El sistema está compuesto por dos capas: una capa de lógica de negocio (`src/`) y una capa de exposición HTTP (`api/`). Ambas capas comparten los mismos módulos centrales, pero difieren en el punto de entrada y en el grado de control sobre el flujo de ejecución.

### Descripción de los scripts

#### Capa `src/` — lógica de negocio

| Script | Responsabilidad |
|---|---|
| `config.py` | Registro central de rutas y constantes: directorios de datos, modelos y salidas, columna objetivo, tamaño del split y métrica principal. |
| `data_loader.py` | Lectura del CSV crudo desde disco y persistencia/carga del dataset procesado. |
| `preprocess_data.py` | Limpieza del dataset (duplicados, filas incoherentes) y construcción de los `Pipeline` de sklearn — uno por modelo — que encapsulan todos los transformadores y el estimador. |
| `trainer.py` | Dos puntos de entrada: `train_models()` para uso desde la API y `train_all()` para uso desde CLI. Orquesta la preparación de datos, el entrenamiento y la persistencia de los modelos. |
| `evaluator.py` | Carga los modelos guardados, calcula métricas sobre el conjunto de test, genera gráficas de salida (ROC, matriz de confusión, importancia de variables) y devuelve la tabla comparativa. |
| `predict.py` | `predict_dataset()` carga el mejor modelo desde disco, ejecuta inferencia batch sobre el dataset procesado y persiste las predicciones en `outputs/predictions.json`. |

#### Capa `src/api/` — exposición HTTP

| Script | Responsabilidad |
|---|---|
| `main.py` | Crea la aplicación FastAPI y registra los cuatro routers. |
| `routers/best_model.py` | `GET /best-model` — sin cuerpo. Ejecuta el pipeline completo (train + predict + evaluate) en una sola llamada delegando en `trainer.train_all()`. Pensado para obtener el mejor modelo sin gestionar el flujo manualmente. |
| `routers/train.py` | `POST /train` — acepta un payload JSON opcional con `hyperparams` por modelo y delega en `trainer.train_models()`. Los pipelines serializados se guardan en `models/tests/`. |
| `routers/predict.py` | `POST /predict` — sin cuerpo. Ejecuta inferencia batch de forma asíncrona sobre el dataset completo delegando en `predict.predict_dataset()` y persiste los resultados en `outputs/predictions.json`. |
| `routers/evaluate.py` | `POST /evaluate` — sin cuerpo. Delega en `evaluator.evaluate_all()`, genera métricas y gráficas, y promueve el mejor modelo a `models/best_model.pkl`. |

### Flujo de ejecución vía FastAPI

La API expone tres endpoints desacoplados que deben invocarse en orden. `POST /train` carga los datos procesados, ajusta todos los pipelines de modelos y los guarda en `models/tests/`. Después, `POST /evaluate` recarga esos modelos, los puntúa sobre el conjunto de test, guarda las gráficas en `outputs/` y promueve el mejor modelo a `models/best_model.pkl`. Por último, `POST /predict` carga `best_model.pkl`, ejecuta inferencia batch de forma asíncrona sobre el dataset y persiste las predicciones en `outputs/predictions.json`.

### Flujo de ejecución vía CLI (`python -m src.trainer`)

Ejecuta el pipeline completo de extremo a extremo en un único comando. Fuerza el reprocesado del dato crudo, entrena todos los modelos, los evalúa inmediatamente, selecciona el ganador por F1-score y escribe `best_model.pkl` — sin necesidad de un paso de evaluación separado.

### Diferencia clave entre ambos flujos

El flujo FastAPI separa el entrenamiento y la evaluación en dos llamadas HTTP explícitas, dando al cliente control sobre cuándo se ejecuta cada fase. El flujo CLI colapsa todo el proceso en una única ejecución bloqueante, pensada para uso directo desde terminal.

## Resultado y elección final

> Este resumen recoge las reflexiones y conclusiones obtenidas a partir de la implementación y comparación de los siguientes algoritmos: **Logistic Regression**, **Decision Tree**, **Random Forest**, **LightGBM** y **Deep Neural Network (Keras)**.

El modelo con mejor rendimiento fue **LightGBM**, alcanzando una **accuracy del 84.08%**, frente al **80.91%** del Random Forest, el **79.85%** de la Red Neuronal, el **76.99%** de la Regresión Logística y el **72.06%** del Árbol de Decisión.

La métrica principal elegida para la comparación fue la **Accuracy**, por ser la métrica más directa e interpretable: refleja la proporción global de predicciones correctas y permite comparar modelos de forma inmediata sin necesidad de ponderar tipos de error.

LightGBM supera al resto de modelos por varias razones inherentes a su arquitectura:

* Es un modelo de *gradient boosting* que construye árboles de forma **secuencial y correctiva**, aprendiendo de los errores del modelo anterior, lo que le permite capturar relaciones no lineales y complejas entre variables.
* A diferencia del Árbol de Decisión, incorpora **regularización implícita** a través del proceso de boosting, lo que lo hace más robusto frente al sobreajuste.
* El dataset de reservas hoteleras contiene variables con interacciones complejas (por ejemplo, la combinación de `lead_time` con `deposit_type`), terreno donde los modelos de *boosting* tienen ventaja natural.

| Modelo               | Accuracy |
|----------------------|----------|
| Logistic Regression  | 76.99%   |
| Decision Tree        | 72.06%   |
| Random Forest        | 80.91%   |
| Deep Neural Network  | 79.85%   |
| LightGBM             | **84.08%** |

## Reflexión crítica sobre limitaciones y mejoras

#### Calidad del dato

* **`company`** fue eliminada del conjunto de features por presentar un **94% de valores nulos**, lo que impide extraer información fiable. Esto supone la pérdida potencial de señal relacionada con reservas corporativas.
* Se detectaron **valores negativos y extremos en `adr`** (precio promedio por noche), corregidos mediante ajuste al **percentil 99**. Este enfoque elimina outliers pero puede enmascarar casos reales atípicos con precios legítimamente altos o bajos.
* El dataset contenía una **gran proporción de filas duplicadas**, lo que podría amplificar el desbalance de clases e introducir sesgo en el entrenamiento si no se eliminan antes de la división train/test.
  * **Patrón no aleatorio:** los duplicados se concentran en reservas corporativas con `company = 281.0` y `adr = 36.00`, lo que apunta a un error sistemático de ingesta por parte de una agencia o empresa concreta, no a ruido aleatorio.
  * **Impacto en el balance de clases:** casi todos los duplicados corresponden a reservas no canceladas (`is_canceled = 0`), por lo que su eliminación reduce proporcionalmente la clase mayoritaria y mejora ligeramente el balance de clases.
  * **Riesgo de data leakage entre train y test:** al haber filas 100% idénticas, si no se eliminan antes de la división, una misma reserva puede acabar simultáneamente en entrenamiento y test, inflando artificialmente las métricas de evaluación.

#### Desbalance de clases

* El dataset presenta más registros de no-cancelación que de cancelación. Se aplicó `class_weight='balanced'` durante la instanciación de los modelos, aunque el impacto observado fue limitado.
* Como mejora futura, se propone explorar técnicas más avanzadas como **SMOTE** (*Synthetic Minority Oversampling Technique*), *undersampling* de la clase mayoritaria, o el ajuste del **umbral de decisión** del clasificador para favorecer el recall sobre la clase minoritaria.

#### Riesgo de data leakage

* Las variables **`reservation_status`** y **`reservation_status_date`** fueron correctamente identificadas y eliminadas del conjunto de entrenamiento, ya que revelan el resultado final de la reserva y constituirían una fuga de información directa hacia el modelo.

#### Alcance y optimización

* No se realizó una **optimización formal de hiperparámetros** (mediante `GridSearchCV` o `RandomizedSearchCV`), por lo que los modelos operan con configuraciones por defecto o ajustes manuales. Una búsqueda sistemática podría mejorar el rendimiento, especialmente en LightGBM.
* El **análisis cuantitativo de importancia de variables** no se completó durante el desarrollo del proyecto, quedando como línea de trabajo futuro junto con técnicas de interpretabilidad como SHAP o LIME.
