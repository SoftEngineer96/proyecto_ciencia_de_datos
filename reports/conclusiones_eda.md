# Conclusiones del análisis exploratorio

Bitácora de hallazgos sobre los 891 pasajeros etiquetados de la capa Silver.
Se escribe a medida que avanza el taller; la Parte 9 la reorganiza como
argumento, con la estratificación que exige separar efectos confundidos.

---

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