# Formulación del problema de machine learning

**Fijado antes de entrenar ningún modelo.** El orden importa: definir la
métrica después de ver los resultados permite elegir la que mejor le quede al
modelo que ya se tiene, y eso invalida la comparación.

---

## 1. Definición del problema

| Decisión | Respuesta | Justificación |
|---|---|---|
| **Tipo de problema** | Clasificación binaria supervisada | El desenlace tiene dos valores conocidos para los 891 pasajeros de entrenamiento. |
| **Variable objetivo** | `Survived` (1 = sobrevivió) | Es el desenlace que la aseguradora necesita estimar. |
| **Unidad de análisis** | Un pasajero | Cada fila es una persona, y la predicción se emite por persona. |
| **Momento de predicción** | **Al embarcar** | Solo puede usarse información disponible en ese instante. Esto excluye cualquier dato generado durante o después del naufragio. |
| **Métrica principal** | **ROC-AUC** | No depende del umbral de decisión y mide la capacidad de **ordenar** pasajeros por riesgo, que es lo que una aseguradora usa para tarificar. Con un 38 % de positivos, la exactitud premia al modelo que ignora a los supervivientes. |
| **Métricas secundarias** | Precisión, recall, F1, exactitud | Se reportan juntas porque cada una oculta algo distinto. Ninguna se usa sola para decidir. |
| **Criterio de éxito** | Superar la regla de género en ROC-AUC, y no quedar por debajo de ella en exactitud | Ver sección 3. |

### Por qué ROC-AUC y no exactitud

Con un 38.4 % de supervivientes, un modelo que siempre responda "no sobrevive"
acierta el 61.6 % de las veces sin haber aprendido nada. Su recall es
exactamente 0: no identifica a ningún superviviente.

La exactitud, por sí sola, no distingue entre ese modelo inútil y uno que
realmente discrimina. ROC-AUC sí, porque evalúa el **ordenamiento** de las
probabilidades y no un corte arbitrario.

---

## 2. Costo de los errores

Los dos errores posibles no cuestan lo mismo, y para una aseguradora el
desequilibrio es claro.

| Error | Qué significa | Consecuencia para la aseguradora |
|---|---|---|
| **Falso positivo** — predecir que sobrevive quien no sobrevivió | El modelo subestima el riesgo | **Es el error caro.** Se emite una póliza con prima insuficiente, o se clasifica como seguro un perfil que no lo es. La pérdida es directa. |
| **Falso negativo** — predecir que muere quien sí sobrevivió | El modelo sobreestima el riesgo | Se cobra de más o se rechaza a un cliente viable. El costo es comercial: ingresos no percibidos, no pérdidas. |

**Implicación operativa.** Si hubiera que mover el umbral de decisión, debería
moverse hacia **reducir falsos positivos**, aun a costa de más falsos
negativos. En términos de métricas, eso significa privilegiar la **precisión
sobre la clase "sobrevive"** cuando haya que elegir.

Esta asimetría se declara aquí y **no se revisará después de ver los
resultados**.

### Una advertencia sobre el uso de este modelo

Las variables más predictivas son el **sexo** y la **clase**. Un modelo de
tarificación que use directamente el sexo sería, en la mayoría de
jurisdicciones actuales, ilegal además de éticamente cuestionable.

Este modelo sirve para **entender qué ocurrió en 1912**, no para fijar primas
hoy. Cualquier uso comercial exigiría revisar qué variables son admisibles como
factor de tarificación.

---

## 3. Baselines: la vara a superar

Calculados sobre el conjunto de entrenamiento, antes de entrenar ningún modelo.

| Baseline | Exactitud | Precisión | Recall | F1 |
|---|---|---|---|---|
| Ingenuo (siempre "no sobrevive") | ≈ 0.616 | 0.000 | **0.000** | 0.000 |
| **Regla de género** (toda mujer sobrevive) | **≈ 0.787** | ≈ 0.742 | ≈ 0.681 | ≈ 0.710 |

**El baseline ingenuo existe para demostrar el problema de la exactitud.**
Acierta casi dos de cada tres veces sin aprender nada, y no encuentra ni un
solo superviviente.

**La regla de género es la vara real.** Es lo que cualquiera deduce del EDA en
cinco minutos, sin entrenar nada. Un modelo que no la supere no justifica su
complejidad, su costo de mantenimiento ni su opacidad.

**Criterio de éxito declarado:** el modelo final debe superar los **0.787 de
exactitud** de la regla de género y alcanzar un ROC-AUC claramente por encima
de **0.5** (azar). Una mejora de exactitud menor a 2 puntos porcentuales no se
considerará suficiente para preferir un modelo complejo sobre la regla simple.

---

## 4. Variables: qué entra y qué no

### Entran al modelo

| Grupo | Columnas |
|---|---|
| Numéricas | `Age` (original, con nulos), `SibSp`, `Parch`, `Fare`, `Familia` |
| Categóricas | `Pclass`, `Sex`, `Embarked`, `TTitulo`, `Alone` |
| En evaluación | `HasCabin`, `TCabin` |

### Quedan fuera, y por qué

| Columna | Motivo |
|---|---|
| `Survived`, `Survivor` | Son el objetivo. Incluirlas es fuga directa. |
| `PassengerId`, `Ticket`, `Name` | Identificadores únicos: el modelo los memoriza sin generalizar. La información útil de `Name` ya está extraída en `TTitulo`. |
| `NSex`, `NEmbarked`, `NCabin`, `NTitulo` | Duplican a sus categóricas imponiendo un orden que no existe (¿"X" es mayor que "G"?). Se reemplazan por codificación *one-hot* dentro del Pipeline. |
| `Age` imputada | Calculada con el promedio de **todos** los pasajeros. Ver abajo. |
| `Person` | Derivada de `Age` y `Sex`: redundante con ambas. |
| `Cabin` | Texto libre de alta cardinalidad. Su información está en `TCabin` y `HasCabin`. |

### La fuga más sutil: la edad imputada

La columna `Age` de Silver se rellenó con el promedio por género calculado
sobre los **891 pasajeros**, incluidos los que acabarán en el conjunto de
prueba. Entrenar con ella le daría al modelo información que, en una situación
real, no tendría.

Por eso `load_dataset()` recupera **`AgeOriginal`**, con sus 177 nulos, y la
imputación se hace **dentro del Pipeline**: así se recalcula en cada partición
de la validación cruzada usando solo datos de entrenamiento.

### Las variables sospechosas: `TCabin` y `HasCabin`

Si `HasCabin` mejora la métrica, caben dos explicaciones:

1. Tener cabina registrada se asocia con mejores condiciones a bordo.
2. La cabina se registró con más detalle para quienes **sobrevivieron para
   declararla**.

La segunda convertiría a `HasCabin` en un **artefacto del registro**, no en un
factor de riesgo. Un modelo que la explote tendría buenas métricas y sería
inútil para estimar riesgo real, porque esa información no existiría al
embarcar.

Además, el EDA mostró que la ausencia de cabina no es aleatoria: falta en el
18.5 % de primera clase y en el 97.6 % de tercera. Buena parte de su poder
predictivo es `Pclass` disfrazada.

**Decisión:** se entrena con y sin estas dos variables, y se comparan los
resultados en el Paso 61. Si la mejora es marginal, se descartan.

---

## 5. Protocolo de evaluación

**División:** 80 % entrenamiento / 20 % prueba, estratificada por `Survived`
para mantener la proporción de supervivientes en ambos conjuntos. Semilla fija
(`RANDOM_STATE=42`) para que la división sea reproducible.

**El conjunto de prueba se abre una sola vez**, en la evaluación final. Todas
las decisiones intermedias —elección de modelo, hiperparámetros, umbral,
variables— se toman con **validación cruzada estratificada sobre el conjunto de
entrenamiento**.

Mirar el conjunto de prueba antes invalidaría la evaluación: cualquier decisión
tomada viéndolo ajustaría el modelo a ese conjunto concreto, y sus métricas
dejarían de estimar el rendimiento sobre datos nuevos.

---

## 6. Selección de modelos

**Resultado:** pasan a la fase de ajuste de hiperparámetros tres finalistas
—`hgb`, `logreg` y `forest`—. `hgb` lidera la métrica principal, `logreg` es el
único sin sobreajuste medible, y `forest` es el más estable entre particiones.

Todo lo que sigue se decidió con **validación cruzada estratificada de 5
pliegues sobre los 712 pasajeros de entrenamiento**. El conjunto de prueba
sigue cerrado.

### 6.1 Resultados de la validación cruzada

| Modelo | ROC-AUC | ± | Exactitud | F1 | Brecha | Ajuste (s) |
|---|---|---|---|---|---|---|
| **hgb** (HistGradientBoosting) | **0.8867** | 0.0194 | **0.8202** | **0.7553** | 0.1068 | 0.359 |
| **forest** (RandomForest) | 0.8657 | **0.0156** | 0.8034 | 0.7358 | 0.1327 | 0.245 |
| **logreg** (LogisticRegression) | 0.8585 | 0.0236 | 0.7921 | 0.7199 | **0.0062** | 0.017 |
| **tree** (DecisionTree) | 0.8468 | 0.0260 | 0.8047 | 0.7189 | 0.0465 | 0.010 |
| **knn** (KNeighbors) | 0.8408 | 0.0169 | 0.7865 | 0.7101 | 0.0873 | 0.010 |
| *dummy* (referencia) | 0.5000 | 0.0000 | 0.6166 | 0.0000 | 0.0000 | 0.010 |

`±` es la desviación del ROC-AUC entre los cinco pliegues. `Brecha` es el
ROC-AUC de entrenamiento menos el de validación. `Ajuste` es el tiempo medio de
un ajuste del pipeline completo.

### 6.2 Lectura por los cinco criterios

**1. Desempeño medio.** `hgb` encabeza con 0.8867, seguido de `forest`
(0.8657) y `logreg` (0.8585). El diagrama de cajas por pliegue muestra la caja
de `hgb` por encima de la de `forest`, con los bigotes apenas tocándose. Es
sugestivo, pero con cinco pliegues sobre 712 pasajeros **no constituye prueba
de superioridad**: la diferencia de 0.021 es del mismo orden que las
desviaciones de ambos.

**2. Estabilidad.** `forest` es el más consistente (±0.0156), seguido de `knn`
(±0.0169) y `hgb` (±0.0194). `tree` es el más errático (±0.0260), y su
desviación de precisión —0.0885, la mayor de la tabla— indica que las reglas
que aprende cambian sustancialmente de un pliegue a otro.

**3. Brecha entrenamiento-validación.** Separa dos familias con nitidez:

- `forest` alcanza **0.9984** de ROC-AUC en entrenamiento y cae a 0.8657 en
  validación: brecha de 0.1327. Un bosque sin profundidad limitada memoriza
  casi por completo los datos que vio.
- `hgb` repite el patrón, algo atenuado: 0.9936 contra 0.8867, brecha 0.1068.
- `logreg` tiene una brecha de **0.0062**, es decir, nula en la práctica: lo
  que rinde en entrenamiento lo rinde con datos nuevos.

Una brecha grande no descalifica a un modelo; señala dónde el ajuste de
hiperparámetros tiene margen para actuar.

**4. Interpretabilidad.** `logreg` entrega coeficientes convertibles en razones
de momios, directamente reportables a la gerencia. `tree` es visualmente
explicable, pero la inestabilidad de sus reglas (criterio 2) erosiona
precisamente esa ventaja: un árbol cuya estructura cambia entre particiones no
sostiene una explicación. `forest` y `hgb` exigen herramientas adicionales
—importancia por permutación o valores SHAP— para justificar una predicción
individual.

**5. Costo computacional.** `logreg` ajusta en 0.017 s frente a los 0.359 s de
`hgb`: **veintiuna veces más rápido** a cambio de 2.8 puntos de ROC-AUC. En
este volumen de datos la diferencia es irrelevante en términos absolutos, pero
se multiplica por cada combinación de hiperparámetros que se evalúe en la
siguiente fase.

### 6.3 Contraste con el criterio declarado en la sección 3

La sección 3 fijó, antes de entrenar, que una mejora de exactitud **menor a 2
puntos porcentuales** no bastaría para preferir un modelo complejo sobre la
regla de género (0.7893 de exactitud sobre el conjunto de entrenamiento).

| Modelo | Exactitud | Margen sobre la regla de género |
|---|---|---|
| hgb | 0.8202 | **+3.1 pts** |
| tree | 0.8047 | +1.5 pts |
| forest | 0.8034 | +1.4 pts |
| logreg | 0.7921 | +0.3 pts |
| knn | 0.7865 | −0.3 pts |

**Solo `hgb` cruza el umbral declarado.** Se deja constancia del hecho sin
revisar el criterio: redefinirlo ahora, viendo los resultados, es exactamente
lo que la sección 3 se comprometió a no hacer.

Dos matices atenúan la conclusión sin anularla. Primero, la métrica principal
es ROC-AUC, no la exactitud, y en ROC-AUC los cinco modelos reales superan con
amplitud el 0.5 del azar. Segundo, la exactitud de estos modelos se mide con
umbral 0.5 por defecto, que no está optimizado; la regla de género no tiene
umbral que ajustar. Aun así, el criterio se declaró sobre la exactitud y el
resultado es el que es.

### 6.4 Finalistas

| Modelo | Por qué pasa | Qué debe corregir el ajuste |
|---|---|---|
| **hgb** | Mejor ROC-AUC (0.8867) y el único que cruza el umbral de 2 puntos en exactitud | Brecha de 0.1068: limitar profundidad y hojas, y revisar la tasa de aprendizaje |
| **logreg** | Brecha nula (0.0062), interpretable sin herramientas auxiliares, 21× más rápido; a 2.8 puntos de ROC-AUC del líder | Poco margen: regularización y, si aporta, términos de interacción |
| **forest** | El más estable entre pliegues (±0.0156) | Brecha de 0.1327, la mayor: `max_depth` y `min_samples_leaf` son la palanca directa |

### 6.5 Descartados

**`knn`** — no supera la vara en exactitud (−0.3 pts), arrastra una brecha de
0.0873 y no admite explicación de una predicción individual. No aporta nada que
otro finalista no ofrezca mejor.

**`tree`** — su ROC-AUC (0.8468) es el más bajo de los modelos con estructura, y
la inestabilidad de sus reglas anula la interpretabilidad que sería su única
ventaja sobre `forest`. `logreg` cubre la necesidad de un modelo explicable con
mejor desempeño y sin sobreajuste.

**`dummy`** — se conserva como referencia, no como candidato. Su exactitud de
0.6166 con recall 0 documenta por qué la exactitud no puede ser la métrica
principal.

### 6.6 La agrupación del imputador de edad

Comparando el mismo modelo lineal y cambiando únicamente el agrupamiento del
imputador:

| Agrupación | ROC-AUC | ± |
|---|---|---|
| `("Sex",)` | 0.8584 | 0.0230 |
| `("Sex", "Pclass")` | 0.8585 | 0.0236 |

La diferencia es de **0.0001**, frente a una desviación entre pliegues de
0.023: una doscientas treintava parte del ruido. **Los datos no distinguen las
dos opciones.**

Se conserva `("Sex", "Pclass")`, y se deja explícito que la decisión **no está
respaldada por la métrica**. La sostiene el análisis exploratorio: la edad
media difiere de forma marcada entre clases, de modo que imputar a una mujer de
tercera clase con el promedio de todas las mujeres le asigna una edad sesgada
hacia arriba. Es un empate resuelto por razonamiento previo, no por evidencia
empírica, y así queda registrado.

### 6.7 Lo que estos resultados no demuestran

**El orden entre los tres primeros no es firme.** Las diferencias de ROC-AUC
entre `hgb`, `forest` y `logreg` —0.021 y 0.007— caben dentro de las
desviaciones entre pliegues. Afirmar que `hgb` es mejor que `logreg` exigiría
una comparación pareada por pliegue, que no se realizó.

**Ninguna cifra estima el desempeño final.** Todas provienen de validación
cruzada sobre el entrenamiento y están optimistamente sesgadas por el solo
hecho de que se usaron para elegir. La estimación del desempeño real se obtiene
una única vez, sobre el conjunto de prueba, en la evaluación final.

**Los hiperparámetros son los de fábrica.** El orden de la tabla puede
reordenarse tras el ajuste, sobre todo entre `hgb` y `forest`, que son los que
más margen tienen por corregir.
