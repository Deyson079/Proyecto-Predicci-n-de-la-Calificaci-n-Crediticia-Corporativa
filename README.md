# Predicción de la Calificación Crediticia Corporativa

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-RNA-D00000?logo=keras&logoColor=white)
[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TU_USUARIO/TU_REPOSITORIO/blob/main/Proyecto_Corporate_Credit_Rating.ipynb)

Proyecto de aprendizaje automático que busca **estimar la calificación crediticia (credit rating) de grandes empresas estadounidenses a partir de sus indicadores financieros**. Incluye un análisis exploratorio completo, aprendizaje **no supervisado** (clustering) y aprendizaje **supervisado** con **Redes Neuronales Artificiales** combinadas con técnicas de **reducción de dimensión** (SFFS, PCA y LDA).

**Universidad de Medellín · Facultad de Ingenierías**

| | |
|---|---|
| **Autores** | Valeria Duque Moreno · Deyson Ortiz Restrepo |
| **Profesor** | Antonio Jesús Tamayo Herrera, PhD |

---

## Tabla de contenido

- [Motivación](#motivación)
- [Base de datos](#base-de-datos)
- [Metodología](#metodología)
- [Resultados principales](#resultados-principales)
- [Conclusiones](#conclusiones)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Cómo ejecutar el proyecto](#cómo-ejecutar-el-proyecto)
- [Tecnologías](#tecnologías)

---

## Motivación

Una **calificación crediticia** es la opinión de una agencia (S&P, Moody's, Fitch, Egan‑Jones…) sobre la capacidad de una empresa para pagar su deuda, en una escala que va de **AAA** (riesgo mínimo) a **D** (incumplimiento). El proceso de las agencias es **costoso y lento**, y muchas empresas no tienen calificación. Un modelo que estime el rating a partir de los estados financieros permite:

- Anticipar cambios de calificación (*upgrades / downgrades*).
- Evaluar el riesgo de empresas sin calificación oficial.
- Apoyar decisiones de inversión, otorgamiento de crédito y cumplimiento regulatorio (p. ej. Basilea).
- Entender qué indicadores (rentabilidad, flujo de caja, endeudamiento, liquidez) pesan más en el riesgo de crédito.

## Base de datos

**Fuente:** [Corporate Credit Rating – Kaggle (agewerc)](https://www.kaggle.com/datasets/agewerc/corporate-credit-rating). El notebook la descarga automáticamente con `kagglehub`.

| Característica | Valor |
|---|---|
| Registros | 2.029 calificaciones |
| Empresas | 593 (NYSE / NASDAQ) |
| Periodo | 2005 – 2016 (mayoritariamente 2010 – 2016) |
| Columnas | 31: 6 descriptivas + 25 indicadores financieros |
| Valores nulos / duplicados | 0 / 0 |
| Variable de salida | `Rating` (10 clases ordinales: AAA, AA, A, BBB, BB, B, CCC, CC, C, D) |

**Indicadores financieros** agrupados en: liquidez (`currentRatio`, `quickRatio`, `cashRatio`…), rentabilidad (`returnOnAssets`, `returnOnCapitalEmployed`, márgenes…), eficiencia (`assetTurnover`, `payablesTurnover`…), endeudamiento (`debtRatio`, `debtEquityRatio`…), flujo de caja (`operatingCashFlowPerShare`, `freeCashFlowPerShare`…) y otros (`effectiveTaxRate`, `enterpriseValueMultiple`).

**Desbalance de clases:** BBB, BB, A y B concentran el 91,7 % de los registros, mientras que AAA, CC, C y D tienen 7, 5, 2 y 1 registros (razón de desbalance **671 : 1**). Por eso el problema se estudia en **dos escenarios**:

| | Escenario A – Base original | Escenario B – Base agrupada |
|---|---|---|
| Variable de salida | `Rating` | `Grado` |
| Clases | 10 (AAA … D) | 2: **Grado de inversión** (AAA–BBB, 57,4 %) vs. **Grado especulativo** (BB–D, 42,6 %) |

## Metodología

### Parte 1 – EDA, preprocesamiento y aprendizaje no supervisado

1. **Análisis exploratorio (EDA):** tipos de datos, calidad, distribución del `Rating`, variables categóricas (`Sector`, agencia), variable temporal, estadística descriptiva, asimetría, atípicos (regla IQR) y correlaciones de Spearman.
2. **Preprocesamiento:**
   - Se eliminan identificadores (`Name`, `Symbol`), la fecha (`Date`) y la agencia (`Rating Agency Name`).
   - `Sector` se codifica con *one‑hot* (12 *dummies*).
   - Los 25 indicadores se **winsorizan** (percentiles 1–99) y se transforman con **Yeo‑Johnson** (`PowerTransformer`), que reduce la asimetría extrema y estandariza.
   - Entrada final: **37 variables** (25 financieras + 12 *dummies*).
3. **Clustering** sin usar la variable de salida: **K‑Means**, **Aglomerativo** (*ward*, *complete*, *average*) y **DBSCAN** (búsqueda en malla de `eps` × `min_samples`), probando k = 2 … 12.
4. **Evaluación:** métricas internas (silueta, Davies‑Bouldin, Calinski‑Harabasz) y externas frente a las clases reales (**ARI** como métrica principal, NMI, V‑measure, pureza y *accuracy* húngara).

### Parte 2 – Reducción de dimensión + Redes Neuronales Artificiales

- **Modelo:** perceptrón multicapa (MLP) en **Keras**, capas `Dense` con ReLU + `Dropout`, optimizador Adam y `EarlyStopping`. Salida *softmax* (10 clases) o *sigmoide* (2 clases).
- **Validación:** `StratifiedKFold` de **5 folds**. Todo lo que se aprende de los datos (PCA, LDA, escalado y la RNA) se ajusta **dentro de cada fold** solo con entrenamiento.
- **Métrica principal:** **F1‑macro**, que da el mismo peso a todas las clases y no se infla con el desbalance.
- **Estrategias comparadas:**
  1. **SFFS** (*Sequential Forward Floating Selection*, `mlxtend`) con 4 *wrappers*: árbol de decisión, KNN, regresión logística y Random Forest, más ajuste de arquitectura de la RNA.
  2. Identificación de las **variables más discriminantes**.
  3. **PCA** (`mlxtend`) con distintas dimensiones *d*.
  4. **LDA** (`mlxtend`, con correcciones para *S_W* singular y el cálculo de *S_B*), *d* = 1…9 en A y *d* = 1 en B.

## Resultados principales

### Aprendizaje no supervisado

| Escenario | Mejor método | ARI | Pureza | Línea base (clase mayoritaria) |
|---|---|---|---|---|
| A – 10 clases | DBSCAN | 0,039 | 0,358 | 0,331 |
| B – 2 clases | DBSCAN | 0,082 | 0,652 | 0,574 |

Los ARI cercanos a 0 muestran que **ningún método de clustering recrea las calificaciones**: los ratings forman un gradiente continuo en el espacio financiero (la silueta de las clases reales es −0,08 en A y 0,05 en B) y no grupos separados.

### Redes Neuronales Artificiales (F1‑macro, validación cruzada de 5 folds)

| Estrategia | Escenario A (10 clases) | Dim. | Escenario B (2 clases) | Dim. |
|---|---|---|---|---|
| Línea base (clase mayoritaria) | 0,050 | – | 0,365 | – |
| Sin reducción | 0,195 ± 0,022 | 37 | 0,738 ± 0,025 | 37 |
| SFFS (regresión logística) + RNA | 0,196 ± 0,026 | 14 | 0,732 ± 0,026 | 11 |
| PCA + RNA | 0,207 ± 0,016 | 30 | **0,759 ± 0,013** | 25 |
| LDA + RNA | **0,232 ± 0,037** | 3 | 0,721 ± 0,030 | 1 |

- **Escenario A:** el mejor modelo es **LDA (3 discriminantes) + RNA**, con F1‑macro de 0,232 y *accuracy* de 0,428. Acierta sobre todo las clases grandes (A, BBB, BB, B) y casi nunca las extremas (AA, CC, C, D).
- **Escenario B:** el mejor modelo es **PCA (25 componentes) + RNA**, con F1‑macro de 0,759 y *accuracy* de 0,770 (F1 de 0,81 para grado de inversión y 0,71 para grado especulativo).

**Variables más discriminantes** (las que el SFFS eligió más veces): `operatingCashFlowSalesRatio`, `operatingCashFlowPerShare`, `cashRatio`, `returnOnCapitalEmployed` y los sectores *Finance*, *Transportation*, *Energy* y *Technology*. Es decir, el **flujo de caja**, la **rentabilidad** y el **sector** pesan más que la liquidez o la eficiencia.

## Conclusiones

- Distinguir **grado de inversión vs. especulativo** es un problema viable (≈ 77 % de *accuracy*), mientras que predecir las **10 calificaciones exactas** es muy difícil con estos datos por el desbalance extremo y el solapamiento entre clases vecinas.
- **LDA** logra la mayor compresión (de 37 a 1–3 ejes) porque usa la etiqueta; **PCA** necesita más componentes porque solo busca varianza; **SFFS** da el modelo más **interpretable**, con variables originales de sentido financiero y sin redundancias.
- El clustering no recrea los ratings, pero **DBSCAN** sirve para detectar empresas con estados financieros atípicos.
- El rating depende también de información que no está en estas razones (tamaño de la empresa, gerencia, entorno del sector, criterio de cada agencia). Agrupar clases, usar técnicas para desbalance o incorporar más variables son caminos naturales para mejorar el modelo.

## Estructura del repositorio

```
├── Proyecto_Corporate_Credit_Rating.ipynb   # Notebook completo (EDA, clustering y RNA)
├── requirements.txt                         # Dependencias
└── README.md
```

## Cómo ejecutar el proyecto

**Opción 1 – Google Colab (recomendada):** haz clic en el botón *Open in Colab* de arriba y ejecuta todas las celdas. Colab ya incluye TensorFlow/Keras; la primera celda instala `kagglehub`, `seaborn` y `mlxtend`.

**Opción 2 – Local:**

```bash
git clone https://github.com/TU_USUARIO/TU_REPOSITORIO.git
cd TU_REPOSITORIO
pip install -r requirements.txt
jupyter notebook Proyecto_Corporate_Credit_Rating.ipynb
```

> La base de datos se descarga sola con `kagglehub`. Si te pide autenticación, configura tus credenciales de Kaggle ([instrucciones](https://github.com/Kaggle/kagglehub#authenticate)).

> **Tiempo de ejecución:** la selección SFFS con Random Forest es la parte más lenta (≈ 10–13 minutos por escenario). El notebook completo tarda alrededor de una hora o algo más.

## Tecnologías

- **Python**, **Jupyter / Google Colab**
- **pandas**, **NumPy**, **SciPy**: manipulación de datos y pruebas estadísticas
- **Matplotlib**, **Seaborn**: visualización
- **scikit‑learn**: preprocesamiento, clustering, métricas y modelos *wrapper*
- **mlxtend**: SFFS, PCA y LDA
- **Keras / TensorFlow**: redes neuronales
- **kagglehub**: descarga de la base de datos
