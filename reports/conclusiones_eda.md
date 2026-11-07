# Conclusiones del análisis exploratorio del Titanic

**Destinatario:** Gerencia · **Objetivo:** identificar qué características
alteraron la probabilidad de sobrevivir, como insumo para modelos de riesgo y
protocolos de evacuación.

**Base de datos:** 891 pasajeros con desenlace conocido. Sobrevivieron 342
(38.4 %).

---

## 1. Hallazgos

### 1.1 El sexo es el factor dominante, por amplio margen

Sobrevivió el **74.2 % de las mujeres** (233 de 314) frente al **18.9 % de los
hombres** (109 de 577). La diferencia es estadísticamente inequívoca
(chi-cuadrado = 260.7; p ≈ 1.2 × 10⁻⁵⁸).

El caso más ilustrativo son cuatro miembros de la familia Fortune, en primera
clase, con la misma tarifa de 263.00 y el mismo camarote: **las dos mujeres
sobrevivieron, los dos hombres no**. Misma clase, misma riqueza, mismo grupo
familiar; la única variable que difiere es el sexo, y predice el resultado por
completo.

### 1.2 La clase no sustituye al sexo: lo modula

Las dos reglas actuaron a la vez. Tasas por clase y tipo de pasajero, con el
número de personas detrás de cada cifra:

| | 1ª clase | 2ª clase | 3ª clase |
|---|---|---|---|
| **Niños** (<16) | 83.3 % (5/6) | **100 % (19/19)** | 43.1 % (25/58) |
| **Mujeres** | 97.8 % (89/91) | 90.9 % (60/66) | 49.1 % (56/114) |
| **Hombres** | 35.3 % (42/119) | 8.1 % (8/99) | 11.9 % (38/319) |

Dos lecturas operativas:

- **"Mujeres y niños primero" se aplicó dentro de cada clase, no entre
  clases.** Los 19 niños de segunda sobrevivieron todos; de los 58 de tercera,
  25.
- **Una mujer de tercera clase (49.1 %) tuvo mejor pronóstico que un hombre de
  primera (35.3 %).** El sexo pesó más que la riqueza en los extremos de la
  jerarquía.

### 1.3 Los mayores de 65 años fueron el grupo más vulnerable

De los 11 pasajeros por encima de 64.8 años, **sobrevivió uno** (9.1 %), frente
al 38.8 % del resto. Es la tasa más baja de cualquier subgrupo identificado.

El número es pequeño, pero la dirección es consistente con el resto del
análisis: la evacuación favoreció a quienes podían desplazarse con autonomía.

### 1.4 Pagar más ayudó, pero solo en primera y segunda clase

Dividiendo la tarifa en cuartiles **dentro de cada clase**, para comparar
pasajeros equivalentes entre sí:

| Cuartil de tarifa | 1ª clase | 2ª clase | 3ª clase |
|---|---|---|---|
| Más barato | 46.3 % | 34.5 % | 22.7 % |
| Más caro | 73.6 % | 59.0 % | 28.3 % |
| **Diferencia** | **+27 pts** | **+25 pts** | **+6 pts** |

Dentro de primera y segunda clase, la tarifa aporta información que la clase
por sí sola no captura. En tercera clase prácticamente no la aporta.

Una explicación plausible —que estos datos no permiten confirmar— es que una
tarifa mayor significara un camarote mejor ubicado respecto de la cubierta de
botes, diferencia que en tercera clase no existía.

### 1.5 El tamaño del grupo familiar describe una U invertida

| Familiares a bordo | 0 | 1 | 2 | 3 | 4–6 | 7 o 10 |
|---|---|---|---|---|---|---|
| Tasa | 30.4 % | 55.3 % | 57.8 % | **72.4 %** | 13–33 % | **0 %** |
| Pasajeros | 537 | 161 | 102 | 29 | 49 | 13 |

Viajar acompañado por **uno a tres familiares** se asocia con el mejor
pronóstico. De los 13 pasajeros con siete o más familiares a bordo, **ninguno
sobrevivió**.

Los grupos de cuatro o más tienen entre 6 y 22 pasajeros, así que sus tasas
intermedias son poco confiables. La afirmación sólida es la de los extremos.

---

## 2. Decisiones de limpieza y su sesgo

Ninguna decisión sobre datos faltantes es neutral. Estas son las cuatro que
afectan la lectura de los resultados anteriores.

**Edad imputada en el 20 % de los registros.** Faltaban 177 de 891 edades. Se
rellenaron con el promedio del propio género. La consecuencia medida: la
desviación estándar cae de 14.53 a 13.01 y el rango intercuartílico se reduce
entre 26 % y 33 %. **Los datos parecen más precisos de lo que son.** Cualquier
intervalo de confianza calculado sobre la edad imputada sale artificialmente
estrecho. Por eso se conservó la edad original en una columna aparte, y todos
los análisis de edad de este informe usan los 714 valores reales.

**Los niños sin edad registrada no se contaron como niños.** La regla clasifica
como niño a quien tiene menos de 16 años; un registro sin edad conserva su
sexo. Se estima que esto deja fuera unos **21 niños**, un subconteo cercano al
25 % sobre los 83 identificados. La cifra del punto 1.2 para niños está, por
tanto, subestimada en número aunque no necesariamente en tasa.

**El título "Master" (niño varón) se fundió con "Mr".** La decisión mezcla
niños con adultos en la variable de título. Es admisible **solo porque existe
una variable separada que distingue a los menores de 16**; sin ella, borraría
una distinción que el análisis demostró decisiva.

**La cabina desconocida se convirtió en una categoría propia, "X", en lugar de
eliminarse.** Faltaba en 687 de 891 registros (77.1 %), así que eliminarlos
habría descartado tres cuartas partes de la muestra. Además, la ausencia del
dato **no es aleatoria**: falta en el 18.5 % de primera clase y en el 97.6 %
de tercera. El hecho de que no haya cabina registrada es, en sí mismo,
información sobre el pasajero.

---

## 3. Confusiones detectadas y cómo se separaron

Tres variables parecían explicar la supervivencia por sí solas y resultaron
estar entrelazadas con la clase. El método para separarlas fue **estratificar**:
comparar dentro de cada nivel de la clase, en lugar de comparar en bruto.

**Puerto de embarque — confusión parcial.** Sin controlar nada, Cherburgo
(55.4 %, n=168) parecía muy superior a Southampton (33.9 %, n=646). La causa
aparente: Cherburgo embarcó un 50.6 % de pasajeros de primera y Queenstown un
93.5 % de tercera.

Al estratificar, la brecha en primera clase se reduce de 21.5 a 10.5 puntos.
**Pero en tercera clase persiste y es grande:** 37.9 % en Cherburgo y 37.5 %
en Queenstown, frente a 19.0 % en Southampton, con 66, 72 y 354 pasajeros
respectivamente. La confusión explicaba parte del efecto, no todo. Queda un
residuo sin explicación que este análisis no puede resolver.

**Tarifa — confusión parcial.** La tarifa está fuertemente determinada por la
clase (correlación de 0.75 con la cubierta registrada, y distribuciones
claramente separadas por clase). Pero al dividirla en cuartiles dentro de cada
clase, conserva señal propia en primera y segunda, como muestra el punto 1.4.

**Título del nombre — redundancia casi total con el sexo.** La correlación
entre ambas variables codificadas es de −0.80: "Mr" es hombre, "Mrs" y "Miss"
son mujeres. El título aparece como segundo mejor predictor sin aportar
información nueva. **Para un modelo, incluir ambas sería duplicar la misma
variable.**

**Una advertencia metodológica adicional.** La edad muestra una correlación
lineal de −0.08 con la supervivencia, prácticamente nula, y sin embargo sí
predice: los varones del tramo más joven sobrevivieron en torno al 52 % y los
del tramo de los 20 años al 14 %. La relación existe, pero no es una línea
recta, y la correlación no la detecta. **Descartar variables por su correlación
baja habría eliminado una variable informativa.**

---

## 4. Qué NO podemos afirmar

**Asociación no es causalidad.** Todo lo anterior describe patrones observados,
no mecanismos. Que ser mujer se asocie con sobrevivir no demuestra que el sexo
fuera la causa: lo más probable es que mediara un protocolo de evacuación
aplicado por la tripulación, que no está en los datos.

**Un solo naufragio.** Los resultados describen un evento ocurrido en 1912, con
su tripulación, su protocolo y sus circunstancias. No son extrapolables a otros
siniestros marítimos ni a escenarios de riesgo contemporáneos sin evidencia
adicional.

**La muestra está incompleta.** Se analizaron 891 de los 1 309 pasajeros
registrados; los 418 restantes no tienen desenlace conocido y quedaron fuera.
**La tripulación no está incluida en absoluto**, pese a haber sido parte
sustancial de las víctimas.

**Faltan variables determinantes.** Los datos no registran la ubicación física
del camarote, la condición física del pasajero, si sabía nadar, su posición en
el barco al momento del impacto, ni a qué bote accedió. Cualquiera de esas
variables podría explicar buena parte de lo que aquí se atribuye a la clase o
a la tarifa.

**Algunos subgrupos son demasiado pequeños para sostener conclusiones.** Las
celdas de Queenstown en primera y segunda clase tienen 2 y 3 pasajeros. Los
niños de primera clase son 6. Los grupos familiares de cuatro o más tienen
entre 6 y 22. Las tasas de esos subgrupos se reportan por completitud, no como
evidencia.

**Los valores atípicos no son errores.** Los 116 pasajeros con tarifa atípica y
los 11 con edad atípica corresponden a personas reales. No se eliminó ninguno:
la regla del rango intercuartílico señala candidatos a revisar, no registros a
descartar.

---

# Anexo — Bitácora de hallazgos

Notas de trabajo registradas a medida que avanzó el análisis. Sustentan las
afirmaciones del informe anterior y conservan el detalle metodológico.
## H1. Los datos faltantes de `Cabin` no son aleatorios (Paso 5)

`Cabin` falta en 687 de 891 pasajeros (77.10 %), `Age` en 177 (19.87 %) y
`Embarked` en 2 (0.22 %).

El cruce con la clase muestra que la ausencia de `Cabin` tiene estructura:

| Clase | Falta `Cabin` |
|---|---|
| 1ª | 18.5 % |
| 2ª | 91.3 % |
| 3ª | 97.6 % |

**Interpretación.** El dato ausente no significa que el pasajero no tuviera
camarote: significa que no quedó registrado. Lo más probable es que refleje
cómo se compilaron los manifiestos, con más detalle para primera clase, y no
una diferencia física en el alojamiento. Es una asociación, no una causa.

**Consecuencia.** `Cabin` no se imputa. Se derivan `TCabin` (cubierta, `'X'`
si no hay dato) y `HasCabin` (0/1), que separa el hecho de que el dato exista
de su contenido.

---

## H2. Clase y sexo están entrelazados (Paso 6)

Proporción de mujeres por clase: 43.5 % en primera, 41.3 % en segunda,
**29.3 % en tercera**. Dicho al revés, la tercera clase era 70.7 % masculina.

**Consecuencia.** Si las mujeres sobrevivieron más y la tercera clase
sobrevivió menos, parte del efecto aparente de la clase puede ser composición
por sexo, y viceversa. Hay que estratificar antes de atribuir el efecto a una
de las dos.

---

## H3. El conteo de niños subestima en torno al 25 % (Paso 8, Reto 8.1)

`add_person()` marca `'child'` si `Age < 16`. En NumPy, `NaN < 16` es `False`,
así que los 177 pasajeros sin edad registrada **nunca** se marcan como niños:
conservan su sexo.

Entre los 714 con edad conocida, el 11.6 % son menores de 16. Aplicando esa
misma proporción a los 177 sin edad, faltarían unos **21 niños** sobre los 83
contados: un subconteo cercano al 25 %.

**Advertencia.** El cálculo supone que los pasajeros sin edad tenían la misma
proporción de niños que los demás, y eso es discutible. Si la edad de los
niños se registró con más cuidado que la de los adultos, 21 sería una
sobreestimación. La cifra es un orden de magnitud, no un dato.

---

## H4. Tercera clase concentraba a los niños (Paso 9)

| Clase | Niños | Total | % de niños |
|---|---|---|---|
| 1ª | 6 | 216 | 2.8 % |
| 2ª | 19 | 184 | 10.3 % |
| 3ª | 58 | 491 | 11.8 % |

**58 de los 83 niños del barco (70 %) viajaban en tercera clase.** Un niño en
primera era una rareza: seis en total.

**Consecuencia.** Prepara la tensión central del análisis de supervivencia:
"mujeres y niños primero" contra el acceso desigual a los botes por clase. Las
dos reglas apuntan en direcciones opuestas para esos 58 niños.

---

## H5. El puerto de embarque es casi un sustituto de la clase (Paso 13)

Composición de clases de cada puerto (normalizado por filas):

| Puerto | 1ª | 2ª | 3ª |
|---|---|---|---|
| Cherburgo (C) | **50.6 %** | 10.1 % | 39.3 % |
| Queenstown (Q) | 2.6 % | 3.9 % | **93.5 %** |
| Southampton (S) | 19.7 % | 25.5 % | 54.8 % |

**Consecuencia.** `Embarked` no es una variable independiente: saber el puerto
informa buena parte de la clase. Si los pasajeros de Cherburgo muestran mayor
supervivencia, la explicación más probable no es el puerto sino la clase que
embarcó allí. Cualquier efecto del puerto debe controlarse por clase antes de
interpretarse.

Es la segunda variable confundida con la clase, después del sexo (H2).

---

## H6. `Alone` pierde el tamaño del grupo familiar (Paso 14)

El 60.3 % de los pasajeros viajaba solo. La distribución de `Familia`
(`SibSp + Parch`) es:

| Familiares | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 10 |
|---|---|---|---|---|---|---|---|---|---|
| Pasajeros | 537 | 161 | 102 | 29 | 15 | 22 | 12 | 6 | 7 |

No hay nadie con 8 ni 9 familiares, y aparecen 7 pasajeros con exactamente 10:
casi con seguridad miembros de una misma familia numerosa, con el resto del
grupo en el conjunto sin etiquetar.

**Consecuencia.** `Alone` trata igual a quien viaja en pareja y a quien viaja
en un grupo de once. Se conservan las dos columnas para poder comprobar si el
tamaño del grupo importa.

---

## Cubiertas (Paso 11–12)

Solo 204 de 891 pasajeros (22.9 %) tienen cabina registrada. La cubierta `T`
aparece con **un solo pasajero**: se excluye de los gráficos por ser un grupo
de tamaño 1, pero la fila se conserva intacta en Silver. Es un problema de
estadística, no de calidad del dato.

---

## H7. "Mujeres y niños primero" operó dentro de cada clase, no entre ellas (Paso 18)

| | 1ª | 2ª | 3ª |
|---|---|---|---|
| child | 0.833 (5/6) | 1.000 (19/19) | 0.431 (25/58) |
| female | 0.978 (89/91) | 0.909 (60/66) | 0.491 (56/114) |
| male | 0.353 (42/119) | 0.081 (8/99) | 0.119 (38/319) |

- Los 19 niños de segunda clase sobrevivieron todos; de los 58 de tercera,
  25.
- El peor grupo del barco son los hombres de segunda clase: 8 de 99 (8.1 %),
  por debajo incluso de los de tercera (11.9 %). Es contraintuitivo y requiere
  explicación, no solo reporte.
- Una mujer de tercera (49.1 %) tuvo mejores probabilidades que un hombre de
  primera (35.3 %).

**Reserva.** El grupo `child` de primera clase son 6 pasajeros: su intervalo de
confianza va de 0.50 a 1.00 y no sostiene ninguna afirmación.

---

## H8. La recta de regresión predice valores imposibles (Paso 19)

Con `Survived` binaria, `lmplot` ajusta una recta que cruza el cero: para
tercera clase, la línea cae por debajo de 0 pasados los 70 años, y el eje del
gráfico llega a −0.2.

Una tasa de supervivencia negativa no existe. Es la limitación de ajustar un
modelo lineal a una variable acotada en [0, 1], y la razón por la que el
modelado posterior usa regresión logística.

---

## H9. El patrón por edad no es lineal, y la recta lo oculta (Paso 19)

Agrupando con `x_bins`:

- **Varones:** el primer intervalo de edad está en torno a 0.52, muy por
  encima de la recta (~0.30), y cae a ~0.14 en el tramo de los 20. "Niños
  primero" aplicó a los varones, pero solo mientras fueron niños.
- **Mujeres:** la tasa sube con la edad, de ~0.65 a ~0.93.

**Hipótesis a verificar, no hallazgo.** El segundo patrón puede estar
confundido por la clase: las mujeres mayores tenían más probabilidad de viajar
en primera. Debe estratificarse antes de interpretarse.

Los intervalos de confianza del tramo de 80 años abarcan casi todo el rango
[0, 1]: son muy pocos pasajeros y el gráfico lo señala.

---

## H10. La tarifa está determinada por la clase (Paso 20)

En la dispersión tarifa-edad coloreada por clase, la tercera clase se agrupa
en tarifas bajas mientras la primera se dispersa por todo el rango. Hay dos o
tres pasajeros con tarifa cercana a 512 frente a una mediana de 14.45; se
tratan formalmente en el Paso 38.

`Fare` es la tercera variable confundida con `Pclass`, después de `Sex` (H2) y
`Embarked` (H5). Si `Fare` predice la supervivencia, buena parte de ese poder
es `Pclass` disfrazada.

---

## H11. Imputar la edad reduce artificialmente la dispersión (Paso 22, Reto 22.1)

La desviación estándar de `Age` cae de **14.53 a 13.01** (un 10.5 %) al imputar
177 valores con el promedio de su género.

La causa es mecánica: esos 177 pasajeros reciben uno de solo dos números,
ambos en el centro de la distribución. La desviación estándar mide el
alejamiento respecto de la media, y añadir puntos en la media suma cero
alejamiento mientras aumenta el total de casos.

En el histograma aparecen dos barras artificiales donde antes no había nada.

**Consecuencia.** La columna imputada parece más precisa de lo que los datos
son. Todo cálculo sensible a la dispersión —intervalos de confianza, errores
estándar, pruebas de hipótesis— sale artificialmente estrecho si se hace sobre
`Age` en lugar de `AgeOriginal`. Por eso Silver conserva las dos columnas y
marca las filas tocadas en `AgeImputada`.

## H12. Fundir `Master` con `Mr` es defendible, pero solo por `Person`

`consolidate_titles()` mapea `Master` (niño varón) a `Mr`, mezclando niños con
adultos en la misma categoría de título.

La decisión se sostiene **únicamente porque `Person` ya separa a los menores de
16 años**: la información sobre la edad no se pierde, solo vive en otra
columna. Si `Person` no existiera, esta fusión borraría una distinción que el
análisis de supervivencia demostró decisiva (los niños de segunda clase
sobrevivieron al 100 %, los hombres de segunda al 8.1 %).

**Riesgo a vigilar.** Al modelar, `NTitulo` y `Person` codifican información
parcialmente redundante. Conviene revisar su correlación antes de incluir las
dos.

## H13. `NTitulo` duplica a `NSex`, y `NCabin` duplica a `Pclass` (Paso 27)

La matriz de correlación revela dos redundancias fuertes entre predictores:

- **`NSex` ↔ `NTitulo` = −0.80.** `Mr` es hombre, `Mrs` y `Miss` son mujeres:
  el título codifica el sexo casi por completo. Esto confirma con un número el
  riesgo anotado en H12. `NTitulo` aparece como segundo mejor predictor (0.43)
  sin aportar información nueva.
- **`Pclass` ↔ `NCabin` = 0.75.** Como `NCabin` vale 8 para cabina desconocida
  y desconocerla es casi sinónimo de tercera clase (H1), su correlación de
  −0.30 con la supervivencia es `Pclass` filtrándose.

**Consecuencia para el modelado.** Incluir los cuatro predictores metería
información repetida. Hay que elegir entre cada par o usar un método que
tolere la colinealidad.

## H14. Una correlación cercana a cero no significa ausencia de relación (Paso 27)

`Age` correlaciona −0.08 con `Survived`: prácticamente nada. Pero el Paso 19
mostró que la relación existe y es fuerte — los varones del primer intervalo
de edad sobrevivieron en torno al 52 % y los del tramo de los 20 al 14 %.

La correlación de Pearson mide solo relaciones **lineales**. El patrón de la
edad es alto en la infancia, cae en la adultez y se aplana: no es una recta, y
los tramos se cancelan entre sí.

**Consecuencia.** Descartar variables por su correlación baja es un error
frecuente y costoso. `Age` habría quedado fuera del modelo pese a ser
informativa.

## H15. Viajar acompañado casi duplica la supervivencia (Paso 28)

| | Tasa | Sobrevivieron | Total |
|---|---|---|---|
| Solo | 0.304 | 163 | 537 |
| Con familia | 0.506 | 179 | 354 |

**Hipótesis a verificar, no hallazgo.** Mujeres y niños viajaban acompañados
con más frecuencia que los hombres solos, así que el efecto puede ser
composición y no causa. Debe estratificarse por sexo y clase en la Parte 9.

## H16. Las estadísticas de edad por supervivencia usan la edad imputada (Paso 28)

`age_by_survival` reporta desviaciones estándar de 12.46 y 13.77, por debajo
del 14.53 de los datos reales (H11): están calculadas sobre `Age` imputada.
La diferencia de medias entre grupos es de solo 2.15 años, coherente con la
correlación casi nula de H14.

Para un informe riguroso, estas estadísticas deberían recalcularse sobre
`AgeOriginal`, restringidas a los 714 pasajeros con edad conocida.

## H17. La imputación encoge la caja y fabrica atípicos (Paso 30)

Rango intercuartílico de la edad, antes y después de imputar:

| | Mujeres | Hombres |
|---|---|---|
| `AgeOriginal` | 19.00 | 18.00 |
| `Age` imputada | 14.00 | 12.00 |

El IQR cae un 26 % en mujeres y un 33 % en hombres, porque 177 pasajeros se
concentran en solo dos valores del centro de la distribución.

**Efecto secundario.** Un atípico se define como un punto fuera de 1.5 × IQR.
Al encoger el IQR, el umbral se estrecha y pasajeros que antes quedaban dentro
pasan a marcarse como anómalos: en el boxplot comparativo aparecen visiblemente
más puntos sueltos con la edad imputada, pese a que los datos reales no
cambiaron.

**Consecuencia.** La detección de atípicos del Paso 38 debe hacerse sobre
`AgeOriginal`. Hacerla sobre `Age` marcaría como anómalos a pasajeros normales.

## H18. El boxplot tiene el mismo punto ciego que la correlación (Paso 30)

Las cajas de `AgeOriginal` por supervivencia son casi idénticas: medianas en
torno a 28 años en ambos grupos. Coherente con la correlación de −0.08 (H14),
y engañoso por la misma razón.

El boxplot resume distribuciones completas y no detecta un patrón que sube en
la infancia y baja en la adultez. La única pista visible es el bigote inferior
del grupo que sobrevivió, que llega casi a 0: los bebés.

Dos herramientas distintas —correlación de Pearson y boxplot— comparten la
misma limitación. Detectar la relación exigió agrupar por tramos de edad
(Paso 19).

## H19. La tarifa está fuertemente sesgada a la derecha (Paso 31)

| | |
|---|---|
| Media | 32.20 |
| Mediana | 14.45 |
| Asimetría | **4.79** |
| Percentil 99 | 249.01 |
| Máximo | 512.33 |

La media más que duplica la mediana y la asimetría de 4.79 está muy por encima
del umbral de 1 que ya indica sesgo fuerte. El 1 % más caro empieza en 249,
casi 17 veces la mediana.

**Implicaciones para el modelado:**

1. **La media no representa al pasajero típico.** Un puñado de tarifas
   extremas arrastra el promedio muy por encima de lo que pagó la mayoría.
2. **Un modelo lineal sobrepondera esos extremos**, porque minimiza errores al
   cuadrado y un punto lejano pesa mucho más que uno cercano.
3. **El remedio habitual es transformar**, por ejemplo con `log(1 + Fare)`, que
   comprime la cola y acerca la distribución a la simetría.

## H20. La confusión puerto-clase es PARCIAL, no total (Paso 35)

Sin estratificar, Cherburgo (55.4 %, n=168) parece muy superior a Southampton
(33.9 %, n=646). Estratificando por clase:

| | 1ª | 2ª | 3ª |
|---|---|---|---|
| C | 0.694 (n=85) | 0.529 (n=17) | 0.379 (n=66) |
| Q | 0.500 (n=**2**) | 0.667 (n=**3**) | 0.375 (n=72) |
| S | 0.589 (n=127) | 0.463 (n=165) | 0.190 (n=354) |

**En primera clase la brecha C−S cae de 21.5 a 10.5 puntos**: buena parte del
efecto era composición por clase, como anticipaba H5.

**Pero en tercera clase persiste y es grande:** C 37.9 % y Q 37.5 % frente a
S 19.0 %, con 66, 72 y 354 pasajeros respectivamente. Esa diferencia no la
explica la clase, porque la comparación ya es dentro de la misma clase.

**Corrección a H5.** El puerto no era únicamente un indicador de la clase.
Queda un efecto residual en tercera que exige otra explicación y que este
análisis no puede dar.

**Reserva.** Las celdas Q×1ª y Q×2ª son 2 y 3 pasajeros: no sostienen ninguna
afirmación pese a mostrar tasas de 0.500 y 0.667.

## H21. La tarifa aporta señal propia, pero solo en primera y segunda (Paso 36)

Cuartiles de tarifa calculados dentro de cada clase:

| FareQ | 1ª | 2ª | 3ª |
|---|---|---|---|
| 0 (más barato) | 0.463 | 0.345 | 0.227 |
| 1 | 0.574 | 0.500 | 0.200 |
| 2 | 0.745 | 0.585 | 0.276 |
| 3 (más caro) | 0.736 | 0.590 | 0.283 |

Pagar más se asocia con **+27 puntos en primera** y **+25 en segunda**, pero
solo **+6 en tercera**, y de forma no creciente.

**Corrección a H10.** `Fare` no es solo `Pclass` disfrazada: dentro de las
clases altas tiene información propia. Una hipótesis plausible —que este
análisis no puede verificar— es que más tarifa significara un camarote mejor
ubicado respecto de la cubierta de botes. En tercera clase esa diferencia no
existía.

## H22. El tamaño familiar describe una U invertida (Paso 37)

| Familiares | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 10 |
|---|---|---|---|---|---|---|---|---|---|
| Tasa | 0.304 | 0.553 | 0.578 | **0.724** | 0.200 | 0.136 | 0.333 | 0.000 | 0.000 |
| n | 537 | 161 | 102 | 29 | 15 | 22 | 12 | 6 | 7 |

La supervivencia sube de 30 % viajando solo hasta **72 % con tres familiares**
y se desploma después: de los 13 pasajeros con 7 o 10 familiares a bordo, **no
sobrevivió ninguno**.

**Reserva sobre la confiabilidad.** A partir de 4 familiares los grupos tienen
entre 6 y 22 pasajeros y los intervalos de confianza del gráfico de puntos
abarcan rangos enormes. La única afirmación sólida en esa zona es el 0 de 13
en los grupos más numerosos.

**Confusión no resuelta.** Las familias más grandes viajaban en tercera clase,
así que el desplome puede ser clase y no tamaño familiar. Separarlo exigiría
estratificar también aquí, y los grupos quedarían demasiado pequeños.

## H23. Los atípicos no son errores, y los de edad fueron los más vulnerables (Paso 38)

| Columna | Atípicos | % | Tasa atípicos | Tasa resto |
|---|---|---|---|---|
| `Fare` | 116 | 13.0 % | 0.681 | 0.339 |
| `Age` (sobre `AgeOriginal`) | 11 | 1.5 % | **0.091** | 0.388 |
| `Familia` | 91 | 10.2 % | 0.341 | 0.389 |

Los atípicos de tarifa sobrevivieron al doble, pero eso es `Pclass`: son casi
todos de primera.

Los 11 atípicos de edad son los mayores de 64.8 años, y **sobrevivió solo 1**.
Es el grupo con peor resultado del barco.

**Ninguno de estos 218 registros es un error.** Una tarifa de 512 la pagaron
tres personas reales; una edad de 80 años la tenía un pasajero real. La regla
del IQR señala candidatos a revisar, no filas a eliminar.

## H24. El caso Fortune: la misma billetera, distinto desenlace (Paso 38)

Cuatro miembros de la familia Fortune viajaban en primera clase con la misma
tarifa de 263.00 y los mismos 5 familiares a bordo:

| Pasajero | Sexo | Sobrevivió |
|---|---|---|
| Fortune, Miss. Mabel Helen | female | Sí |
| Fortune, Miss. Alice Elizabeth | female | Sí |
| Fortune, Mr. Charles Alexander | male | No |
| Fortune, Mr. Mark | male | No |

Misma clase, misma tarifa, misma familia, mismo camarote. La única variable
que difiere es el sexo, y predice el resultado perfectamente.

Es la ilustración más limpia del hallazgo central del análisis: por encima de
la riqueza y de la clase, el protocolo de evacuación operó sobre el sexo.


