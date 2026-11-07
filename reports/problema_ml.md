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
