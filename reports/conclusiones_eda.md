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
