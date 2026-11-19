# Predicción de supervivencia en el Titanic

Proyecto del **Taller de Buenas Prácticas en Ciencia de Datos e IA**.
Fork de la plantilla de [dfmarin](https://github.com/dfmarin/proyecto_ciencia_de_datos).

Estima la probabilidad de que un pasajero del RMS Titanic sobreviviera al
naufragio del 15 de abril de 1912, usando **solo información disponible al
embarcar**.

---

## Resultado

| | |
|---|---|
| Modelo | `HistGradientBoostingClassifier` |
| **ROC-AUC en prueba** | **0.8238** · IC 95 % [0.7469, 0.8939] |
| Exactitud | 0.8101 · 179 pasajeros |
| Vara a superar | Regla de género, 0.7765 → margen **+0.0336** |

El conjunto de prueba se abrió **una sola vez**, al final. Todas las decisiones
—modelo, hiperparámetros, umbral y variables— se tomaron con validación cruzada
sobre el conjunto de entrenamiento.

> **El intervalo de confianza mide 15 puntos de ancho, y eso es parte del
> resultado**, no una nota al pie. Con 179 pasajeros de prueba, esta muestra no
> distingue dos modelos que difieran en menos de eso.

---

## Cómo ejecutarlo

```bash
cp .env.example .env               # revisa TITANIC_URL
python -m venv .venv && source .venv/bin/activate
make install
make test                          # 152 pruebas
make lint
make pipeline                      # ingesta -> silver -> gold -> modelo
```

`make pipeline` genera `data/bronze`, `data/silver`, `data/gold` y
`models/titanic_model_v1.joblib` junto con sus metadatos.

Verificado desde un clon limpio en un entorno virtual nuevo: los resultados
coinciden hasta el último decimal.

---

## Estructura

```
flows/                      Código de producción, una etapa por archivo
  00_ingestion.py             fuente -> Bronze (copia cruda, nunca se altera)
  01_data_engineering.py      Bronze -> Silver (limpieza y variables derivadas)
  03_aggregation.py           Silver -> Gold (11 tablas de agregación)
  05_ml_models.py             Silver -> modelo entrenado, evaluado y guardado
  job.py                      Orquesta las cuatro etapas
  common/                     Rutas, configuración, logging y huellas SHA-256
notebooks/                  Exploración y modelado; sin lógica productiva
tests/                      152 pruebas
reports/                    Los tres informes escritos
```

Arquitectura **medallón**: cada capa tiene un contrato de entrada y de salida
que las pruebas verifican. Los cuadernos leen a través de los módulos, nunca
los datos directamente.

Los números de los nombres marcan el orden de las etapas; saltan 02 y 04
porque esas son cuadernos. Un módulo que empieza por dígito no se puede
importar con `import`, así que se carga con `importlib.import_module()`.

---

## Informes

| Documento | Qué contiene |
|---|---|
| [`reports/conclusiones_eda.md`](reports/conclusiones_eda.md) | Hallazgos del análisis exploratorio, el sesgo que introduce cada decisión de limpieza, y qué **no** permiten afirmar los datos |
| [`reports/problema_ml.md`](reports/problema_ml.md) | Nueve secciones: métrica, costo de los errores, protocolo de evaluación, selección de modelo, umbral, calibración, hiperparámetros e interpretación |
| [`reports/model_card.md`](reports/model_card.md) | Ficha del modelo: datos con su huella, variables, métricas con incertidumbre, desempeño por subgrupos y **usos no recomendados** |

---

## Algunas decisiones, y por qué

**ROC-AUC como métrica principal, no exactitud.** Con 38 % de supervivientes,
responder siempre "no sobrevive" acierta el 62 % sin haber aprendido nada.

**La imputación de la edad ocurre dentro del Pipeline.** Usar la columna ya
imputada de Silver habría filtrado al entrenamiento estadísticas calculadas con
el conjunto de prueba.

**El umbral es 0.50 por una razón, no por costumbre.** Es el que cumple el piso
de precisión del 80 % con menos falsos positivos, que es el error caro
declarado *antes* de entrenar.

**El ajuste de hiperparámetros no aportó nada.** La búsqueda anunció +0.0058 de
mejora; la validación cruzada anidada mostró que era sesgo de selección
(ganancia honesta +0.0039 ± 0.0266). La configuración se adoptó igualmente,
pero por la calibración —que mejoró un 40 %—, no por la métrica principal.

**Las variables de cabina se descartaron.** No mejoran el modelo, y había una
sospecha de fondo: que la cabina se registrara con más detalle para quienes
sobrevivieron para declararla, lo que la convertiría en un artefacto del
registro y no en un factor de riesgo.

**Quitar `Sex` del modelo sería cosmético.** Cuesta solo 0.0057 de ROC-AUC,
porque `TTitulo` multiplica su importancia por 5.3 y toma su lugar: "Mr" es
hombre, "Mrs" y "Miss" son mujeres.

---

## Un límite que conviene conocer antes de usarlo

El modelo **no sirve para evaluar pasajeros varones**: encuentra solo el 35 %
de los que sobrevivieron, sobre un subgrupo que representa el 64 % de los
datos. Su exactitud en ese grupo (0.83) es *mayor* que en las mujeres (0.81),
porque acertar "muere" a casi todos regala el 81 % sin aprender nada — lo que
hace que la métrica global oculte el problema en vez de mostrarlo.

Las probabilidades tampoco están bien calibradas: sirven para **ordenar**
pasajeros por riesgo, no para fijar un valor monetario.

El detalle completo, con las cifras por subgrupo y la lista de usos
desaconsejados, está en la [ficha del modelo](reports/model_card.md).
