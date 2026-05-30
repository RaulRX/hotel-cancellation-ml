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

## Resultado y elección final

> Este resumen recoge las reflexiones y conclusiones obtenidas a partir de la implementación y comparación de los siguientes algoritmos: **Logistic Regression**, **Decision Tree** y **LightGBM**.

El modelo con mejor rendimiento fue **LightGBM**, alcanzando un **F1-score de ~70%**, frente al **65.5%** de la Regresión Logística y el **~63%** del Árbol de Decisión.

La métrica principal elegida para la comparación fue el **F1-score**, por encima de otras métricas como el Recall o la Accuracy. La justificación es de carácter de negocio: dado que el dataset presenta desbalance de clases, la Accuracy puede ser engañosa. El F1-score equilibra Precisión y Recall, penalizando de forma simétrica ambos tipos de error:

- **Falsos negativos** (no predecir una cancelación real): el hotel no toma medidas preventivas y pierde ingresos.
- **Falsos positivos** (predecir una cancelación que no ocurre): el hotel ofrece incentivos o descuentos innecesariamente.

LightGBM supera a los otros dos modelos por varias razones inherentes a su arquitectura:

- Es un modelo de *gradient boosting* que construye árboles de forma **secuencial y correctiva**, aprendiendo de los errores del modelo anterior, lo que le permite capturar relaciones no lineales y complejas entre variables.
- A diferencia del Árbol de Decisión, incorpora **regularización implícita** a través del proceso de boosting, lo que lo hace más robusto frente al sobreajuste.
- El dataset de reservas hoteleras contiene variables con interacciones complejas (por ejemplo, la combinación de `lead_time` con `deposit_type`), terreno donde los modelos de *boosting* tienen ventaja natural.

| Modelo               | F1-score (aprox.) |
|----------------------|-------------------|
| Regresión Logística  | 65.50%            |
| Árbol de Decisión    | ~63%              |
| LightGBM             | ~70%              |

## Reflexión crítica sobre limitaciones y mejoras

#### Calidad del dato

- **`company`** fue eliminada del conjunto de features por presentar un **94% de valores nulos**, lo que impide extraer información fiable. Esto supone la pérdida potencial de señal relacionada con reservas corporativas.
- Se detectaron **valores negativos y extremos en `adr`** (precio promedio por noche), corregidos mediante ajuste al **percentil 99**. Este enfoque elimina outliers pero puede enmascarar casos reales atípicos con precios legítimamente altos o bajos.
- El dataset contenía una **gran proporción de filas duplicadas**, lo que podría amplificar el desbalance de clases e introducir sesgo en el entrenamiento si no se eliminan antes de la división train/test.
  - **Patrón no aleatorio:** los duplicados se concentran en reservas corporativas con `company = 281.0` y `adr = 36.00`, lo que apunta a un error sistemático de ingesta por parte de una agencia o empresa concreta, no a ruido aleatorio.
  - **Impacto en el balance de clases:** casi todos los duplicados corresponden a reservas no canceladas (`is_canceled = 0`), por lo que su eliminación reduce proporcionalmente la clase mayoritaria y mejora ligeramente el balance de clases.
  - **Riesgo de data leakage entre train y test:** al haber filas 100% idénticas, si no se eliminan antes de la división, una misma reserva puede acabar simultáneamente en entrenamiento y test, inflando artificialmente las métricas de evaluación.


#### Desbalance de clases

- El dataset presenta más registros de no-cancelación que de cancelación. Se aplicó `class_weight='balanced'` durante la instanciación de los modelos, aunque el impacto observado fue limitado.
- Como mejora futura, se propone explorar técnicas más avanzadas como **SMOTE** (*Synthetic Minority Oversampling Technique*), *undersampling* de la clase mayoritaria, o el ajuste del **umbral de decisión** del clasificador para favorecer el recall sobre la clase minoritaria.

#### Riesgo de data leakage

- Las variables **`reservation_status`** y **`reservation_status_date`** fueron correctamente identificadas y eliminadas del conjunto de entrenamiento, ya que revelan el resultado final de la reserva y constituirían una fuga de información directa hacia el modelo.

#### Alcance y optimización

- No se realizó una **optimización formal de hiperparámetros** (mediante `GridSearchCV` o `RandomizedSearchCV`), por lo que los modelos operan con configuraciones por defecto o ajustes manuales. Una búsqueda sistemática podría mejorar el rendimiento, especialmente en LightGBM.
- El **análisis cuantitativo de importancia de variables** no se completó durante el desarrollo del proyecto, quedando como línea de trabajo futuro junto con técnicas de interpretabilidad como SHAP o LIME.
