# Reconocimiento facial con PCA GO

Práctica de **Sistemas Biométricos (UMA)**. Es un sistema de reconocimiento facial basado en **PCA GO** (*PCA on Image Gradient Orientations*, IGO-PCA). Usa YuNet como detector y una base de datos en CSV cifrada con AES-256-GCM. Incluye los 9 métodos que pide el enunciado, la evaluación completa y una batería de ataques.

> **Objetivo de la práctica:** que el grupo rival no encuentre defectos que no conozcamos. Por eso la sección [Debilidades conocidas](#9-debilidades-conocidas) es la más importante de este README.

---

## Índice
1. [Uso rápido](#1-uso-rápido)
2. [Los 9 métodos](#2-los-9-métodos)
3. [Cómo funciona PCA GO](#3-cómo-funciona-pca-go)
4. [Decisiones de diseño](#4-decisiones-de-diseño)
5. [Entrenamiento y reproducibilidad](#5-entrenamiento-y-reproducibilidad)
6. [Experimentos (elección de parámetros)](#6-experimentos-elección-de-parámetros)
7. [Evaluación](#7-evaluación)
8. [Seguridad y ataques](#8-seguridad-y-ataques)
9. [Debilidades conocidas](#9-debilidades-conocidas)
10. [Pendiente](#10-pendiente)
11. [Estructura del repositorio](#11-estructura-del-repositorio)
12. [Referencias](#12-referencias)

---

## 1. Uso rápido

El repositorio ya trae el modelo entrenado (`models/`), así que **no hace falta LFW para usar el sistema**. Solo hace falta para reentrenar o reevaluar.

### Opción A: Python local (3.10 o superior)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m facerec init             # genera la clave e inscribe todas las fotos de data/
python -m facerec listar
python -m facerec verificar taylor_swift_2009 data/taylor_swift_2009.jpg
python -m facerec identificar data/adele_2016.jpg
```

### Opción B: Docker

```bash
mkdir -p db secrets fotos
docker compose build
docker compose run --rm facerec init
docker compose run --rm facerec identificar /fotos/una_foto.jpg     # pon la foto en ./fotos/
```

`db/` (la BD) y `secrets/` (la clave) se montan como volúmenes. **No se suben al repo**: cada instalación genera su propia clave con `init`.

### La clave de la BD

La clave se busca en este orden:
1. la variable `FACEREC_DB_KEY` (base64 de 32 bytes),
2. el fichero indicado en `FACEREC_DB_KEY_FILE`,
3. `secrets/db.key`, que crea `init`.

Para usar otra BD se puede usar `--db ruta.csv` o la variable `FACEREC_DB_PATH`.

---

## 2. Los 9 métodos

### Desde Python

```python
from facerec import SistemaBiometrico
s = SistemaBiometrico()

s.inscribir("ana", "ana.jpg")          # foto -> detector -> extractor -> BD
s.verificar("ana", "ana2.jpg")         # -> True / False
s.identificar("ana2.jpg")              # -> "ana" / None (desconocido)

s.inscribir_sd("ana", "cara.jpg")      # sin detector: la foto debe ser YA la cara
s.verificar_sd("ana", "cara.jpg")
s.identificar_sd("cara.jpg")

t = s.extraer("ana.jpg")               # template: 150 floats
s.inscribir_t("ana", t)                # sin detector ni extractor
s.verificar_t("ana", t)
s.identificar_t(t)
```

`foto` puede ser una ruta, `bytes` o un array de numpy (BGR o gris). `template` puede ser un array, una lista o un texto `"v1,v2,...,v150"`.

### Desde la línea de comandos

```bash
python -m facerec inscribir    <id> <foto>
python -m facerec verificar    <id> <foto>
python -m facerec identificar  <foto>
python -m facerec inscribir_sd <id> <foto>     # y verificar_sd / identificar_sd
python -m facerec inscribir_t  <id> <template> # y verificar_t / identificar_t
python -m facerec extraer [--sd] <foto> [--out t.npy]   # imprime el template
python -m facerec borrar <id>
python -m facerec listar
```

Un `<template>` puede ser un fichero `.npy`, un `.csv` o `.txt` con valores separados por comas, o los valores escritos directamente.

**Salida:** JSON por stdout. El código de salida es **0** si se acepta o todo va bien, **1** si se rechaza o la persona es desconocida, y **2** si hay un error.

### Comportamiento de cada operación

| Situación | Comportamiento |
|---|---|
| `verificar` | `True` si la similitud coseno con el template inscrito es ≥ **λ = 0.2035** |
| `identificar` | Identificación **abierta**: devuelve el más parecido si su similitud es ≥ **λ_id = 0.3313**; si no, devuelve `None` |
| Inscribir un id que ya existe | **Error**. Hay que `borrar` antes |
| `verificar` con un id que no existe | `False`, igual que un rechazo, para no revelar si el id existe |
| Foto sin cara | Error `FaceError` |
| Foto con varias caras | Se usa la de **mayor área** |
| Template `_t` con longitud distinta de 150, NaN, inf, valores de más de 1000 o vector nulo | Error `TemplateError` |
| Id con caracteres raros | Error. Se aceptan 1–64 caracteres `[A-Za-z0-9 _-().']`, empezando por letra o número |
| BD manipulada o clave incorrecta | Error `DBIntegrityError`: el sistema **no opera** (*fail closed*) |

---

## 3. Cómo funciona PCA GO

PCA GO aplica PCA, pero no sobre el brillo de los píxeles sino sobre **la dirección de los bordes**. La dirección del gradiente casi no cambia cuando cambia la luz, y en las zonas tapadas las diferencias de dirección son aleatorias y tienden a cancelarse.

```
foto ─► YuNet (cara + 5 landmarks) ─► alineación por los ojos ─► 64×64 gris
     ─► blur gaussiano σ=1.2 ─► Sobel 3×3 (Gx, Gy) ─► φ = atan2(Gy, Gx)
     ─► [cos φ, sin φ] por píxel = vector de 8.192 ─► PCA (k=150, whitening)
     ─► template de 150 floats ─► similitud coseno ─► ≥ λ ?
```

- Se usa **cos y sin** en lugar del ángulo directamente para que 1° y 359° salgan parecidos.
- El PCA se entrena con **4.065 caras de LFW** y se guarda en `models/pca_igo.npz` (media + 150 componentes + autovalores). **No se reentrena al inscribir a nadie**.
- Con **whitening**, cada componente se divide por la raíz de su autovalor, de modo que todas pesan lo mismo en el coseno. Mejora el EER en unos 2 puntos (ver §6).

---

## 4. Decisiones de diseño

| # | Decisión | Elección | Por qué |
|---|---|---|---|
| 1 | Detector | **YuNet** (`cv2.FaceDetectorYN`) | Viene con OpenCV, así que no añade dependencias pesadas. Da 5 landmarks (ojos) para alinear y detectó el 100 % de las caras del dataset y de LFW |
| 2 | Confianza mínima del detector | **0.7** | Todas las caras del dataset dan ≥ 0.86. Así se descartan detecciones dudosas |
| 3 | Varias caras en la foto | **La de mayor área** | `adam_pearson_2024` tiene 3 caras detectadas. Si se rechazaran las fotos con varias caras, no se podría inscribir a esa persona |
| 4 | Alineación | **Similitud** (rotación + escala + traslación) con los 2 ojos en posiciones fijas | PCA GO compara la posición de los bordes, así que sin alinear falla |
| 5 | Tamaño de la cara | **64×64**, recorte ajustado de cejas a barbilla (ojos en (0.30, 0.33) y (0.70, 0.33)) | La cara más pequeña del dataset mide 87 px, así que nunca se agranda. 32, 48, 64 y 80 dan resultados casi iguales (§6) |
| 6 | Pre-proceso del gradiente | **Blur gaussiano σ=1.2 + Sobel 3×3** | El blur reduce el ruido en zonas planas (el ángulo ahí es aleatorio). Mejora el EER de 36,7 % a 35,3 % |
| 7 | Datos para entrenar el PCA | **LFW**, personas con 1 sola foto (4.065), pasadas por **nuestro** pipeline | Con las 50 fotos del profe saldrían como mucho 49 componentes, sobreajustados |
| 8 | Famosos solapados | Se **excluyen** de entrenamiento y calibración las 9 personas que están en el dataset del profe y en LFW, más 3 gemelos o hermanos | Si no, las métricas salen infladas. Se usan como fotos de prueba reales |
| 9 | Nº de componentes | **k = 150** | Mejor EER en validación junto con k=100–200 (§6) |
| 10 | Distancia | **Similitud coseno** | Los vectores IGO tienen norma constante y la comparación teórica de IGO es una correlación |
| 11 | Umbral de verificación λ | **0.2035**, calibrado en LFW para **FAR = 1 %** | Sistema de seguridad: se prioriza no dejar pasar impostores |
| 12 | Umbral de identificación λ_id | **0.3313**, calibrado para **FPIR = 1 %** con 51 inscritos | Con λ, el 39 % de los desconocidos acaba identificado como alguien, porque se compara contra 51 personas y el FAR se acumula |
| 13 | Tipo de identificación | **Abierta** (puede devolver "desconocido") | La identificación cerrada siempre identifica a alguien, y es trivial de explotar |
| 14 | Id que ya existe | **Error** | Si se permitiera sobrescribir, se podría suplantar a alguien simplemente reinscribiéndose con su id |
| 15 | Ruta `_sd` | Escala de grises y redimensionado a 64×64, **sin alinear** | No hay detector, así que no hay landmarks. Se rechazan las imágenes de menos de 32 px |
| 16 | Ruta `_t` | Validar longitud, tipo, finitud, rango y norma | Un NaN o un vector nulo romperían el coseno |
| 17 | Formato de la BD | CSV `id,nonce,data` + una fila `__manifest__` | El enunciado exige CSV |
| 18 | Protección de la BD | **AES-256-GCM por fila**, con el id como dato asociado, más un **manifiesto** autenticado del conjunto de filas | Da confidencialidad e integridad, impide mover templates entre ids y detecta filas borradas, añadidas o duplicadas |
| 19 | Clave | Fuera del repo y de la imagen Docker (`secrets/` o variable de entorno) | Si la clave estuviera junto al CSV, cifrar no serviría de nada |
| 20 | Qué devuelve la CLI | Solo **sí/no** o el id, **nunca el score** | Con el score, un ataque *hill-climbing* entra en 19 consultas (§8) |
| 21 | Métricas | FAR/FRR/EER y DET para verificación; rank-k, CMC y FPIR para identificación | Son las estándar en biometría |
| 22 | Persona nº 51 | **Zohran Mamdani** | Elección del grupo |
| 23 | Docker | `python:3.12-slim` + `opencv-python-headless` + `numpy` + `cryptography` | Imagen ligera, sin compilar nada |

---

## 5. Entrenamiento y reproducibilidad

Todo lo que hay en `models/` y `docs/` se regenera con estos pasos:

```bash
pip install -r requirements-dev.txt

# 0. Descargar LFW (~230 MB). Queda en ~/scikit_learn_data/lfw_home/lfw_funneled
python -c "from sklearn.datasets import fetch_lfw_people; fetch_lfw_people()"   # requiere scikit-learn

# 1. Pasar LFW por nuestro detector + alineación y crear los splits (~1 min)
python scripts/prepare_lfw.py --lfw ~/scikit_learn_data/lfw_home/lfw_funneled

# 2. Experimentos: tamaño x k x whitening, blur y comparación con PCA de píxeles (~4 min)
python scripts/experiments.py

# 3. Entrenar el PCA definitivo y calibrar λ y λ_id
python scripts/train.py

# 4. Crear la BD
python -m facerec init --force

# 5. Evaluación y ataques
python scripts/evaluate.py --lfw ~/scikit_learn_data/lfw_home/lfw_funneled
python scripts/attacks.py

# Tests
python -m pytest -q
```

### Splits de LFW

Los splits se hacen **por persona**, de modo que nadie aparece en dos splits:

| Split | Fotos | Personas | Uso |
|---|---:|---:|---|
| `train` | 4.065 | 4.065 | Entrenar el PCA (personas con 1 sola foto) |
| `calib` | 4.118 | 836 | Elegir parámetros y calibrar λ y λ_id |
| `test` | 4.911 | 836 | **Solo** la evaluación final. Nunca se usó para decidir nada |
| `overlap` | 120 | 9 | Famosos que también están inscritos: pruebas reales contra el sistema |
| `twin` | 19 | 3 | Ashley Olsen, Oliver Phelps y Venus Williams: ataque de gemelos |

En `calib` y `test` se usan como mucho 10 fotos por persona, para que George W. Bush (530 fotos) no domine los resultados.

---

## 6. Experimentos (elección de parámetros)

EER en `calib` (cuanto más bajo, mejor). El PCA se entrena con `train`. Datos completos en `docs/experiments.json`.

**Tamaño × nº de componentes (con whitening, blur 0.8):**

| Tamaño | k=25 | k=50 | k=100 | k=150 | k=200 | k=300 | k=400 |
|---|---|---|---|---|---|---|---|
| 32×32 | 38,3 % | 36,4 % | **34,9 %** | 35,4 % | 35,8 % | 36,9 % | 37,7 % |
| 48×48 | 38,9 % | 37,2 % | 35,8 % | 35,5 % | 35,5 % | 36,4 % | 36,4 % |
| 64×64 | 39,2 % | 38,0 % | 36,5 % | 35,6 % | 35,8 % | 36,0 % | 36,5 % |
| 80×80 | 39,3 % | 38,1 % | 36,5 % | 36,2 % | 36,0 % | 36,6 % | 37,0 % |

- **Sin whitening** todo empeora unos 2 puntos (37–40 %).
- **Blur antes del Sobel** (64×64, k=150): σ=0 → 36,7 %; 0,5 → 36,4 %; 0,8 → 35,6 %; **1,2 → 35,3 %**; 1,5 → 35,2 %; 2,0 → 34,9 %. Se elige σ=1,2, porque más blur equivale a bajar la resolución y las mejoras ya son marginales.
- **PCA de píxeles** (*eigenfaces*, la línea base): 40,3 % sin whitening y 37,0 % con whitening. PCA GO es mejor, pero por poco en fotos sin alterar. Donde se nota la diferencia es con cambios de luz y oclusiones (§7.D).
- En una exploración puntual (no incluida en `experiments.py`) se probaron otros recortes. Uno algo más ajustado dio 35,1 %, una diferencia despreciable. Los recortes todavía más ajustados o sin frente fueron peores, entre 35,9 % y 36,7 %.

**Conclusión:** las diferencias entre configuraciones son de 1–2 puntos. El techo lo pone el método (PCA, lineal y no supervisado), no los parámetros. Se eligen **64×64, k=150, whitening y σ=1,2**, que está a menos de 0,5 puntos del mejor resultado.

---

## 7. Evaluación

Ejecutar `python scripts/evaluate.py`. Los resultados completos están en `docs/results_eval.json` y las figuras en `docs/figs/`.

### A. LFW `test` (836 personas nunca vistas)

![DET](docs/figs/lfw_det.png)

| Métrica | Valor |
|---|---|
| Pares genuinos / impostores | 7.195 / 5.159.310 |
| **EER** | **34,6 %** |
| Con λ: FAR / FRR | 1,0 % / **83,7 %** |
| Identificación cerrada (galería de 418): rank-1 / rank-5 / rank-10 | 8,6 % / 16,5 % / 22,7 % |
| Identificación abierta con λ_id: correctos / dice "desconocido" / FPIR | 3,6 % / 88,2 % / 7,8 %* |

\*λ_id está calibrado para una galería de 51 personas. Aquí la galería es de 418, por eso el FPIR sale por encima del 1 %.

![CMC](docs/figs/lfw_cmc.png)

**Lectura:** las distribuciones de genuinos e impostores se solapan muchísimo. Con un FAR del 1 % se rechaza a 5 de cada 6 genuinos. Es coherente con la literatura: *eigenfaces* ronda el 60 % de acierto en LFW. LFW tiene pose, expresión y luz libres y una resolución baja.

### B. Sistema real (50 inscritos)

**Genuinos:** 120 fotos de LFW de 9 personas inscritas, comparadas contra su foto de inscripción del dataset del profe.

| Persona | Fotos | Acepta `verificar` | Rank-1 | `identificar` correcto |
|---|---:|---:|---:|---:|
| serenna_williams_2017 | 52 | 9,6 % | 13,5 % | 0 % |
| elisabeth_II_1945 | 13 | 0 % | 7,7 % | 0 % |
| jackie_chan_2000 | 13 | 0 % | 0 % | 0 % |
| salma_hayek_2012 | 13 | 7,7 % | 7,7 % | 0 % |
| keanu_reeves_2021 | 12 | 8,3 % | 8,3 % | 0 % |
| michael_jackson_1982 | 12 | 0 % | 0 % | 0 % |
| mary-kate_olsen_2011 | 3 | 0 % | 0 % | 0 % |
| christian_bale_2012 | 1 | 0 % | 0 % | 0 % |
| james_phelps_(gemelo_weasley)_2011 | 1 | 0 % | 0 % | 0 % |
| **Total** | **120** | **5,8 %** | **8,3 %** | **0 %** |

**Impostores:** 4.911 fotos de personas no inscritas (LFW `test`).
- FAR de `verificar` por comparación: **0,88 %**.
- FPIR de `identificar` (con λ_id): **0,67 %**. Si se usara λ, sería del **33,8 %**.

**Lectura:** el sistema casi nunca deja pasar a un impostor, pero tampoco reconoce a los genuinos cuando la foto es distinta de la de inscripción. Las peores son las de **grandes diferencias de edad**: Isabel II (1945 frente a los años 2000), Michael Jackson (1982 frente a los 2000) y Jackie Chan.

### C. Gemelos y hermanos

| Foto | Objetivo | Resultado |
|---|---|---|
| Ashley Olsen (gemela) | mary-kate_olsen_2011 | Rechazada (score −0,03, rango 29) |
| Oliver Phelps (gemelo) | james_phelps_(gemelo_weasley)_2011 | Rechazado (score 0,04, rango 15) |
| Venus Williams (hermana, 17 fotos) | serenna_williams_2017 | Todas rechazadas. **En 2 fotos Serena sale como la más parecida (rango 1)** y en otra como rango 2 |

Ojo: se rechazan porque el sistema rechaza casi todo, **no porque distinga gemelos**. Con un umbral más permisivo, o en identificación cerrada, Venus sería identificada como Serena en el 12 % de sus fotos.

### D. Robustez: PCA GO frente a PCA de píxeles

Galería: una foto original por persona. Las fotos de prueba se alteran sintéticamente. La tabla muestra la aceptación de genuinos con FAR = 1 % (LFW `test`, 836 galería y 1.609 pruebas).

![ejemplos](docs/figs/robustez_ejemplos.png)
![robustez](docs/figs/robustez.png)

| Alteración | PCA GO | PCA píxeles |
|---|---:|---:|
| Original | 17,5 % | 14,8 % |
| Oscura (γ=2,5) | **17,5 %** | 12,8 % |
| Clara (γ=0,4) | **17,5 %** | 11,5 % |
| Contraste ×0,4 | **17,5 %** | 12,1 % |
| Luz lateral | 12,5 % | 11,9 % |
| Mascarilla | **8,9 %** | 6,7 % |
| Gafas de sol | **12,2 %** | 5,5 % |
| Media cara tapada | 4,4 % | **6,5 %** |
| Ruido σ=15 | **16,2 %** | 14,6 % |
| Desenfoque σ=2 | 16,5 % | **17,7 %** |
| Baja resolución (16 px) | 15,7 % | **16,9 %** |

**Lectura:** se confirma la teoría.
- PCA GO es **totalmente invariante a cambios globales de brillo y contraste**: el resultado no se mueve, mientras que PCA de píxeles pierde hasta 3 puntos.
- Aguanta mucho mejor las **gafas de sol** (más del doble) y algo mejor la **mascarilla**.
- Con **luz lateral** (sombras que crean bordes nuevos) pierde 5 puntos.
- Con **media cara tapada** y con **desenfoque o baja resolución** es **peor** que PCA de píxeles, porque se pierden bordes.

### E. Ruta `_sd` (sin detector)

- 120 fotos genuinas: con detector se acepta el **5,8 %**. Pasando a `verificar_sd` el recorte del detector **sin alinear**, el **2,5 %**. Se confirma que la alineación importa.
- Imágenes que no son caras (gris uniforme, negro, ruido, degradado, tablero) en `identificar_sd`: **ninguna se identifica**. Pero una imagen gris uniforme saca un score de **0,171 contra adam_pearson_2024**, muy cerca de λ = 0,2035. Ver la debilidad D17.

---

## 8. Seguridad y ataques

Ejecutar `python scripts/attacks.py`, que trabaja sobre una copia de la BD. Los resultados están en `docs/results_attacks.json`.

### Modelo de amenazas

- El atacante **conoce todo el código y el modelo PCA**, porque el repo es público para el rival.
- Puede **leer** y/o **escribir** el CSV. Para el caso peor, también se considera que **roba la clave**.
- Puede llamar a cualquiera de los 9 métodos, incluidos `_sd` y `_t`, saltándose así el detector y el extractor.

### Resultados

**1. Lectura de la BD**

| Ataque | Resultado |
|---|---|
| Leer el CSV sin la clave | Ve los **ids en claro** (50) y blobs cifrados. No obtiene ningún template |
| Leer el CSV **con la clave robada** | Descifra los 50 templates y, con el modelo PCA del repo, **reconstruye el mapa de bordes** de la cara inscrita, con un coseno medio de 0,88 respecto al real (figura de abajo) |

![reconstrucción](docs/figs/reconstruccion.png)

**2. Escritura en la BD (sin clave)**

| Ataque | Resultado |
|---|---|
| Modificar 1 bit de un template | ✅ Detectado |
| Intercambiar templates entre dos ids | ✅ Detectado |
| Renombrar el id de una fila | ✅ Detectado |
| Borrar una fila | ✅ Detectado |
| Duplicar una fila | ✅ Detectado |
| Añadir una fila fabricada | ✅ Detectado |
| Vaciar campos / dejar solo la cabecera | ✅ Detectado |
| **Rollback**: restaurar una copia antigua completa del CSV | ❌ **No detectado**. Una persona dada de baja vuelve a estar inscrita |
| **Con la clave robada**: poner mi template bajo el id de la víctima | ❌ **No detectado**. `verificar(víctima, foto_atacante) = True` |

**3. Saltarse partes del sistema**

| Ataque | Resultado |
|---|---|
| `_t` con el template filtrado de la víctima | ❌ Aceptado (es lo esperable: el template *es* la credencial) |
| `_t` con ceros, NaN, inf, longitud incorrecta, 1e9 o texto | ✅ Rechazado |
| `_t` con el **template medio** de todos los inscritos | ❌ `verificar_t` lo acepta para **19 de los 50 ids**. `identificar_t` devuelve `None` (score 0,330, justo por debajo de λ_id) |
| `_t` **fuerza bruta** con 200.000 templates aleatorios | ❌ **1.177 aciertos contra un id concreto (0,59 %, unos 170 intentos por acierto)**. 161 aciertos de `identificar_t` contra alguien (0,08 %) |
| `_t` *hill-climbing* **con acceso al score** | ❌ Entra en **19 consultas**. Por eso la CLI no devuelve el score |
| `identificar_sd` con imágenes que no son caras | ✅ Ninguna se identifica |
| `identificar` (con detector) con imágenes que no son caras | ✅ Rechazado (no se detecta ninguna cara) |
| `_sd` con una imagen de 10×10 | ✅ Rechazado |
| Ids maliciosos (`=cmd\|...`, `../../`, vacío, 100 caracteres, `__manifest__`) | ✅ Rechazados |
| Reinscribir un id existente con otra cara | ✅ Rechazado |
| `verificar` con un id que no existe | ✅ `False`, sin revelar que no existe |

---

## 9. Debilidades conocidas

Lista completa de los fallos que conocemos. Si el rival encuentra uno que no está aquí, es un fallo nuestro.

### Reconocimiento
- **D1. Precisión baja con fotos reales.** EER del 34,6 % en LFW. Con λ se rechaza al 84 % de los genuinos, y en el sistema real se acepta solo el 5,8 % de fotos genuinas distintas de la de inscripción. Es una limitación del propio PCA GO (lineal y no supervisado), no un bug. En fotos controladas (frontales, buena luz) funciona mejor.
- **D2. Edad.** Con diferencias grandes (Isabel II en 1945, Michael Jackson en 1982) el acierto es del 0 %.
- **D3. Pose.** Con la cara girada o de perfil, los bordes cambian de sitio y falla. No está medido aparte, pero LFW incluye pose libre.
- **D4. Expresión fuerte** (boca abierta, ojos cerrados): mueve los bordes, con el mismo efecto que la pose.
- **D5. Dependencia de la alineación.** Si YuNet sitúa mal los ojos, falla. Por la ruta `_sd` no hay alineación y el acierto baja del 5,8 % al 2,5 %.
- **D6. Oclusiones grandes.** Con media cara tapada, PCA GO es **peor** que PCA de píxeles (4,4 % frente a 6,5 %). La mascarilla también lo baja a la mitad.
- **D7. Luz lateral fuerte.** Las sombras crean bordes nuevos y el acierto baja de 17,5 % a 12,5 %. Solo es invariante a cambios **globales** de luz.
- **D8. Desenfoque y baja resolución.** Se pierden los bordes, y aquí PCA de píxeles es igual o mejor.
- **D9. Gemelos y hermanos.** No los distingue: Venus Williams sale como la persona más parecida a Serena en 2 de 17 fotos. Se rechazan solo porque el umbral es muy estricto.
- **D10. "Lobos".** Algunos inscritos atraen más falsos positivos que otros. Por ejemplo, `aitana_etxebarria_(twin_melody)_2024` y `christian_bale_2012` con λ_id, y `christian_bale_2012`, `leticia_sabater_1996` y `keanu_reeves_2021` con λ.
- **D11. Una sola foto de inscripción.** El template depende de esa foto concreta (luz, expresión, edad).
- **D12. Posible sesgo demográfico.** El PCA está entrenado con LFW, que tiene sobre todo hombres, personas de piel clara y políticos de EE. UU. de los años 2000. No está medido.

### Detector y presentación
- **D13. Sin detección de vida.** Una foto impresa o en pantalla de la víctima funciona igual que la víctima. No está medido (ver §10).
- **D14. Varias caras.** Se coge la más grande, así que un atacante puede poner delante una foto grande de otra persona.
- **D15. `_sd` acepta cualquier imagen como cara.** No comprueba que sea una cara: solo exige un mínimo de 32 px.

### Interfaz `_t`
- **D16. Fuerza bruta con templates aleatorios.** El 0,59 % de los intentos pasa `verificar_t` contra un id concreto, unos 170 intentos por acierto. **No hay límite de intentos.**
- **D17. Template medio ("wolf").** La media de los templates inscritos pasa `verificar_t` para 19 de los 50 ids. Con imágenes sin cara por `_sd` el score se queda cerca de λ (0,17 frente a 0,20).
- **D18. Hill-climbing.** Si el atacante tiene el score (`puntuar()` y `ranking()` de la librería Python, pensados para evaluación), entra en unas 19 consultas. La CLI no lo expone, pero cualquiera que importe la librería sí puede usarlo.
- **D19. Template robado = credencial.** Quien tenga un template válido pasa `verificar_t`. Un template no se puede "revocar": la cara es la misma.

### Base de datos y ficheros
- **D20. Rollback.** Restaurar una copia antigua completa del CSV no se detecta. Haría falta un contador monotónico guardado fuera del CSV.
- **D21. Ids en claro.** Leer el CSV revela quién está inscrito.
- **D22. Robo de la clave = control total.** Permite descifrar, reconstruir el mapa de bordes de cada cara inscrita (coseno 0,88) y meter templates propios. La clave está en `secrets/db.key`, en la misma máquina que la BD.
- **D23. Modelos no autenticados.** `models/threshold.json` y `models/pca_igo.npz` no van firmados. Quien pueda escribir en `models/` puede, por ejemplo, poner `"lambda": -1` y aceptar a todo el mundo. El hash del PCA se comprueba contra `threshold.json`, pero ese fichero también se puede modificar.
- **D24. Sin bloqueo de fichero.** Dos escrituras simultáneas pueden perder una inscripción. La escritura es atómica, pero no hay *lock*.
- **D25. Decodificación de imágenes no confiables.** Se usa OpenCV, y sus vulnerabilidades serían las nuestras. Se limita el tamaño a 20 MB y a 8000 px.

### Umbrales
- **D26. Compromiso de seguridad frente a usabilidad.** λ y λ_id son muy estrictos. Con λ_id, `identificar` no reconoce a ningún genuino del sistema real (0 %). Bajar los umbrales dispara los falsos positivos: con λ en identificación, 1 de cada 3 desconocidos se identifica como alguien.
- **D27. Umbrales calibrados en LFW.** Las fotos del dataset del profe son de mejor calidad, así que los scores pueden no estar en la misma escala.

---

## 10. Pendiente

- [ ] **Persona que falta:** el enunciado dice 50 fotos del profe + 1 nuestra = 51, pero `data/` tiene **49 del profe + Mamdani = 50**. Hay que confirmar con el profe si falta una foto.
- [ ] **Más fotos de Mamdani** (5–10, variadas) para probar al inscrito nº 51: ponerlas en `tests/mamdani/` (no en `data/`).
- [ ] **Ataques de presentación** (D13): foto impresa y foto en la pantalla del móvil de varios inscritos. Hay que hacerlos a mano con una cámara.
- [ ] **Construir la imagen Docker** (`docker compose build`). El Dockerfile está escrito, pero no se pudo construir en el entorno de desarrollo porque Docker Hub no estaba accesible. El código se ha probado con las mismas versiones de librerías.
- [ ] **Evaluar el sistema del grupo rival** con los mismos scripts de ataques, adaptando las llamadas.
- [ ] Posibles mejoras, si da tiempo: límite de intentos en `_t` (D16), contador anti-rollback (D20) y firma de `models/` (D23).

---

## 11. Estructura del repositorio

```
facerec/
  config.py      parámetros (tamaños, rutas, umbrales del detector)
  detector.py    carga de imágenes, YuNet, alineación, recorte _sd
  igo.py         vector IGO, PCA (entrenar/proyectar/reconstruir), coseno
  db.py          CSV cifrado AES-GCM + manifiesto
  system.py      SistemaBiometrico: los 9 métodos + borrar/listar
  metrics.py     FAR/FRR/EER, umbrales, CMC
  __main__.py    CLI (python -m facerec ...)
scripts/
  prepare_lfw.py detector + alineación sobre LFW, splits
  experiments.py elección de tamaño, k, whitening y blur
  train.py       PCA definitivo + calibración de λ y λ_id
  evaluate.py    evaluación (bloques A–E) + figuras
  attacks.py     batería de ataques
models/          YuNet (.onnx), PCA (.npz), umbrales (.json)
data/            fotos de inscripción (49 del profe + Mamdani)
docs/            resultados (.json) y figuras
tests/           pytest (21 tests)
Dockerfile, docker-compose.yml, requirements*.txt
```

---

## 12. Referencias

- G. Tzimiropoulos, S. Zafeiriou, M. Pantic. *Subspace Learning from Image Gradient Orientations*. IEEE TPAMI 34(12), 2012.
- G. B. Huang et al. *Labeled Faces in the Wild: A Database for Studying Face Recognition in Unconstrained Environments*. UMass Amherst TR 07-49, 2007.
- W. Wu et al. *YuNet: A Tiny Millisecond-level Face Detector*. Machine Intelligence Research, 2023 (modelo `face_detection_yunet_2023mar` de OpenCV Zoo).
- M. Turk, A. Pentland. *Eigenfaces for Recognition*. Journal of Cognitive Neuroscience, 1991.
