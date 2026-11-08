# Ficha del modelo — `titanic_model_v1`

| | |
|---|---|
| **Nombre** | `titanic_model_v1` |
| **Algoritmo** | `HistGradientBoostingClassifier` (scikit-learn) |
| **Versión** | 1.0 · noviembre de 2026 |
| **Artefacto** | `models/titanic_model_v1.joblib` + `.metadata.json` |
| **Reproducir** | `make pipeline` |

---

## 1. Propósito y uso previsto

Estima la probabilidad de que un pasajero del RMS Titanic hubiera sobrevivido
al naufragio del 15 de abril de 1912, a partir de información **disponible al
embarcar**.

**Para qué sirve**

- Cuantificar y ordenar los patrones de supervivencia del naufragio.
- Ordenar pasajeros históricos por riesgo relativo, como insumo de un análisis
  retrospectivo.
- Material docente sobre evaluación sin fuga de datos.

**Para qué NO sirve** — ver la sección 7, que es parte obligatoria de esta
ficha y no un anexo.

---

## 2. Datos de entrenamiento

| | |
|---|---|
| **Fuente** | Conjunto Titanic, ingerido desde la URL registrada en `.env` |
| **Capa** | `data/silver/titanic_silver.parquet` |
| **Huella SHA-256** | `c2e1d0b71949330f…` (completa en `models/titanic_model_v1.metadata.json`) |
| **Tamaño** | 891 pasajeros con desenlace conocido |
| **Periodo** | Un único suceso: 15 de abril de 1912 |
| **Partición de evaluación** | 712 entrenamiento / 179 prueba, estratificada, `random_state=42` |
| **Entrenamiento del artefacto** | Los **891** pasajeros etiquetados |

El artefacto guardado se entrena con todos los datos etiquetados, prueba
incluida. Su desempeño esperado es el que midió la partición 712/179, porque
el procedimiento ya estaba validado cuando se reentrenó.

**Lo que los datos NO incluyen:** los 418 pasajeros sin desenlace conocido, la
tripulación completa —parte sustancial de las víctimas—, y cualquier variable
sobre la ubicación física del camarote, la condición del pasajero, si sabía
nadar o a qué bote accedió.

---

## 3. Variables

**Entran al modelo (10)**

| Tipo | Columnas |
|---|---|
| Numéricas | `Age` (original, con nulos), `SibSp`, `Parch`, `Fare`, `Familia` |
| Categóricas | `Pclass`, `Sex`, `Embarked`, `TTitulo`, `Alone` |

La imputación de la edad ocurre **dentro** del Pipeline, agrupando por `Sex` y
`Pclass`, y se recalcula en cada partición. Usar la `Age` ya imputada de
Silver habría filtrado estadísticas del conjunto de prueba.

**Quedan fuera, y por qué**

| Columna | Motivo |
|---|---|
| `Survived`, `Survivor` | Son el objetivo. |
| `PassengerId`, `Ticket`, `Name` | Identificadores únicos; la señal de `Name` está en `TTitulo`. |
| `NSex`, `NEmbarked`, `NCabin`, `NTitulo` | Imponen un orden inexistente; se reemplazan por one-hot. |
| `Age` imputada en Silver | Calculada con todos los pasajeros: fuga. |
| `Person` | Redundante con `Age` y `Sex`. |
| `Cabin` | Alta cardinalidad; su señal estaría en `TCabin`. |
| **`HasCabin`, `TCabin`** | **Descartadas por la ablación del Paso 61:** empeoran el modelo en 0.0020 e importancia por permutación 0.0000. Además, la cabina falta en el 97.6 % de tercera clase y se sospecha que se registró mejor para quienes sobrevivieron para declararla. |

---

## 4. Configuración congelada

```python
FINAL_ESTIMATOR    = "hgb"
FINAL_PARAMS       = {"model__l2_regularization": 1.0,
                      "model__learning_rate": 0.1,
                      "model__max_depth": 4,
                      "model__max_iter": 200}
USE_CABIN_FEATURES = False
THRESHOLD          = 0.50
```

El umbral **no es la convención**: se eligió porque es el que cumple el piso
de precisión del 80 % con el menor número de falsos positivos, que es el error
caro declarado antes de entrenar. Viaja con el modelo en los metadatos y
`predict()` lo usa por defecto.

Sobre los hiperparámetros: la búsqueda del Paso 57 anunció una mejora de
+0.0058, y la validación cruzada anidada mostró que **era íntegramente sesgo de
selección** (ganancia honesta +0.0039 ± 0.0266, 2 de 5 pliegues). Se adoptaron
igualmente porque no cuestan nada en la métrica principal y mejoran la
calibración un 40 % — y esa mejora es creíble precisamente porque el Brier no
era el objetivo de la búsqueda.

---

## 5. Desempeño

### Conjunto de prueba — 179 pasajeros, abierto una sola vez

| Métrica | Valor | IC 95 % (bootstrap) | Ancho |
|---|---|---|---|
| **ROC-AUC** | **0.8238** | [0.7469, 0.8939] | 0.147 |
| **Exactitud** | **0.8101** | [0.7486, 0.8659] | 0.117 |
| **F1** | 0.7424 | [0.6475, 0.8246] | 0.177 |
| Precisión | 0.7778 | — | — |
| Recall | 0.7101 | — | — |
| Brier | 0.1522 | — | — |

### Contra la vara, sobre el mismo conjunto de prueba

| | Exactitud | F1 |
|---|---|---|
| Regla de género ("toda mujer sobrevive") | 0.7765 | 0.6923 |
| **`titanic_model_v1`** | **0.8101** | **0.7424** |
| **Margen** | **+0.0336** | +0.0501 |

El criterio de éxito, declarado antes de entrenar, exigía superar a la regla de
género por al menos 2 puntos de exactitud. **Se cumple, con +3.4.**

### Validación cruzada (entrenamiento) frente a prueba

| Medición | ROC-AUC |
|---|---|
| CV anidada — procedimiento completo | 0.8783 ± 0.0239 |
| **Prueba** | **0.8238** [0.7469, 0.8939] |
| Diferencia | −0.0545 |

La anticipación **cae dentro** del intervalo de la prueba, de modo que el
procedimiento de evaluación se sostiene. La prueba salió algo por debajo; con
179 pasajeros no es posible distinguir entre azar de la partición y un resto de
optimismo acumulado por las decisiones tomadas con el entrenamiento.

> **El ancho del intervalo es el dato que debe acompañar siempre a la cifra.**
> Con 147 milésimas de amplitud en el ROC-AUC, esta muestra no puede
> distinguir dos modelos que difieran en menos de eso. Todas las diferencias
> discutidas durante el desarrollo —0.021 entre candidatos, 0.0058 del ajuste,
> 0.002 a 0.010 de las ablaciones— caben varias veces dentro de ella.

---

## 6. Desempeño por subgrupos

Medido con predicciones fuera de muestra sobre los 712 de entrenamiento
(Paso 54), umbral 0.5.

| Variable | Grupo | n | Tasa real | Exactitud | Precisión | **Recall** |
|---|---|---|---|---|---|---|
| Sexo | female | 253 | 0.7431 | 0.8063 | 0.8564 | **0.8883** |
| Sexo | male | 459 | 0.1852 | 0.8279 | 0.5556 | **0.3529** |
| Clase | 1ª | 171 | 0.6491 | 0.7895 | 0.8505 | 0.8198 |
| Clase | 2ª | 150 | 0.4467 | 0.9133 | 0.9219 | 0.8806 |
| Clase | 3ª | 391 | 0.2430 | 0.7980 | 0.6026 | **0.4947** |
| Edad | <16 | 76 | 0.5658 | 0.8684 | 0.8837 | 0.8837 |
| Edad | 16-30 | 247 | 0.3482 | 0.8016 | 0.7467 | 0.6512 |
| Edad | 30-50 | 202 | 0.4406 | 0.7822 | 0.7647 | 0.7303 |
| Edad | 50+ | 50 | 0.3000 | 0.8800 | 0.9091 | 0.6667 |
| Edad | sin registrar | 137 | 0.2920 | 0.8613 | 0.8000 | 0.7000 |

> ### El límite más serio de este modelo
>
> **El recall en pasajeros varones es 0.3529.** De cada tres hombres que
> sobrevivieron, el modelo encuentra uno; y cuando predice que un hombre
> sobrevive, acierta el 55.6 % de las veces. No es un subgrupo marginal: son
> **459 pasajeros, el 64 % del conjunto de entrenamiento**.
>
> Y su **exactitud es más alta** (0.8279) que la de las mujeres (0.8063). Como
> solo el 18.5 % de los hombres sobrevivió, responder "muere" a casi todos
> regala el 81.5 % sin aprender nada. La métrica global premia exactamente lo
> contrario de lo que el modelo hace bien.
>
> **Un modelo con 82 % de exactitud global es, aquí, dos modelos de calidad muy
> distinta compartiendo un promedio.**

Los errores de alta confianza lo confirman: de los diez en que el modelo estaba
más seguro, siete son hombres de segunda o tercera clase que **sobrevivieron**
con probabilidad asignada entre 0.0005 y 0.0156, y tres son mujeres de primera
o segunda que **murieron** con probabilidad entre 0.98 y 0.99. Son las
excepciones a la regla de género, y el modelo no dispone de ninguna variable
que las explique.

---

## 7. Limitaciones y usos no recomendados

### Usos explícitamente desaconsejados

| Uso | Razón |
|---|---|
| **Fijar primas** o cualquier valor monetario derivado de la probabilidad | El modelo está mal calibrado: un puntaje de 0.69 correspondía a una supervivencia real del 50 %. La regularización lo mejoró un 40 %, pero no lo corrigió. Sirve para **ordenar**, no para medir. |
| **Evaluar a pasajeros varones** | Recall 0.3529 sobre 459 pasajeros. |
| **Decidir sobre personas individuales** en cualquier contexto | Un recall del 35 % en un grupo y del 89 % en otro traslada un trato desigual a quien caiga del lado malo, de forma invisible para quien mire solo la métrica global. |
| **Extrapolar** a otros siniestros o a riesgo contemporáneo | Un único evento, con su tripulación y su protocolo. |
| **Cualquier decisión regulada** donde el sexo sea factor prohibido | Ver abajo. |

### Eliminar `Sex` no resolvería el problema regulatorio

La ablación del Paso 61 midió que quitar `Sex` del conjunto de variables cuesta
**−0.0057** de ROC-AUC, dentro del ruido. La razón: `TTitulo` multiplica su
importancia **por 5.3** y absorbe el papel completo de `Sex`, porque "Mr" es
hombre y "Mrs"/"Miss" son mujeres.

> Suprimir la columna `Sex` para cumplir una prohibición **no quitaría el sexo
> del modelo**. El cumplimiento sería aparente. Eliminarlo de verdad exigiría
> quitar también `TTitulo` y auditar qué variable toma el relevo después.

### Limitaciones de la medición

- **La muestra de prueba son 179 pasajeros.** Los intervalos de confianza
  abarcan entre 12 y 18 puntos según la métrica.
- **El conjunto de prueba se abrió una vez**, como exige el protocolo, pero
  todas las decisiones previas se tomaron con el entrenamiento: queda un resto
  de optimismo que la diferencia de −0.0545 frente a la CV anidada puede estar
  reflejando.
- **Un solo suceso histórico.** No hay segunda muestra contra la cual validar.
- **El 20 % de las edades está imputado**, y los niños sin edad registrada no
  se contaron como niños (unos 21, un subconteo cercano al 25 %).
- **Asociación no es causalidad.** Que `Sex` encabece la importancia no
  significa que el sexo causara sobrevivir: lo más probable es que mediara un
  protocolo de evacuación que no está en los datos.
- **El desempeño por subgrupos** de la sección 6 se midió sobre predicciones
  fuera de muestra del entrenamiento con los parámetros por defecto, no con la
  configuración final. Los valores son indicativos; la asimetría entre sexos,
  por su magnitud, no depende de ese detalle.

### Advertencia ética

El modelo se apoya en `Sex` y `Pclass`, las variables que gobernaron la
evacuación de 1912. **Predecir bien lo que ocurrió no implica que deba ocurrir
así**, ni que esas variables sean admisibles para decidir quién recibe un
servicio hoy. En la mayoría de jurisdicciones, tarificar por sexo es ilegal.

---

## 8. Entorno y reproducción

| | |
|---|---|
| Python | 3.14.4 |
| scikit-learn | 1.9.1 |
| pandas | 3.1.0 |
| `random_state` | 42 |

```bash
git clone <repositorio> && cd proyecto_ciencia_de_datos
python -m venv .venv && source .venv/bin/activate
make install
make test          # 152 pruebas
make lint
make pipeline      # ingesta -> silver -> gold -> modelo
```

`make pipeline` regenera `data/bronze`, `data/silver`, `data/gold` y
`models/titanic_model_v1.joblib` junto con sus metadatos.

> **Seguridad.** Un archivo `.joblib` es un `pickle`: puede ejecutar código al
> cargarse. No cargues este modelo desde una fuente que no controles.
> `load_model()` avisa si la versión de scikit-learn cambió respecto de la
> registrada, porque un modelo guardado con una versión puede comportarse
> distinto con otra.

---

## 9. Trazabilidad

Cada decisión de este modelo está justificada en un documento del repositorio:

| Decisión | Dónde se justifica |
|---|---|
| Métrica principal, costo de los errores, baselines | `reports/problema_ml.md`, secciones 1 a 3 |
| Variables incluidas y excluidas | `reports/problema_ml.md`, sección 4 |
| Protocolo de evaluación | `reports/problema_ml.md`, sección 5 |
| Elección del modelo | `reports/problema_ml.md`, sección 6 |
| Umbral, calibración, subgrupos, ética | `reports/problema_ml.md`, sección 7 |
| Hiperparámetros | `reports/problema_ml.md`, sección 8 |
| Interpretación y ablación | `reports/problema_ml.md`, sección 9 |
| Hallazgos del análisis exploratorio | `reports/conclusiones_eda.md` |
| Desarrollo completo | `notebooks/02_exploracion.ipynb`, `notebooks/04_modelado.ipynb` |
