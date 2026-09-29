# HPC Tools for AI — Baseline BERT/SQuAD en una sola GPU

**Plataforma:** CESGA FinisTerrae III
**Modelo:** `google-bert/bert-base-uncased`
**Dataset:** SQuAD v1.1
**Framework:** PyTorch + Hugging Face Transformers
**Hardware de referencia:** 1 × NVIDIA A100-PCIE-40GB

---

## Resumen

Este informe establece una **baseline reproducible de una sola GPU** para el fine-tuning de BERT sobre SQuAD antes de pasar a entrenamiento distribuido. El estudio analiza el rendimiento de entrenamiento, la utilización de GPU, el uso de memoria, los modos de precisión, el tamaño de batch, la configuración del DataLoader, `torch.compile` y la estabilidad temporal utilizando exactamente una GPU NVIDIA A100.

La baseline final de rendimiento utiliza **BF16**, un **batch size por dispositivo de 160** y **una época completa de SQuAD**. En tres repeticiones válidas, el tiempo medio de entrenamiento fue de **234.58 s**, con un **coeficiente de variación del 0.42%**, lo que proporciona una referencia estable para futuros experimentos de escalado multi-GPU.

> **Baseline de referencia:** 1 × A100-PCIE-40GB · BF16 · batch 160 · 1 época · **234.58 s de tiempo medio de entrenamiento**

---

## 1. Objetivos

El objetivo de esta baseline es caracterizar el rendimiento del fine-tuning de BERT sobre un único acelerador antes de introducir ejecución distribuida.

La baseline fue diseñada para:

- hacer fine-tuning de `google-bert/bert-base-uncased` sobre SQuAD v1.1;
- ejecutarse sobre **exactamente una GPU NVIDIA A100**;
- medir el tiempo de pared de la fase de entrenamiento;
- utilizar una carga suficientemente larga para obtener medidas temporales significativas;
- identificar cuellos de botella de cómputo, memoria y entrada de datos;
- proporcionar una referencia reproducible para futuros experimentos de entrenamiento distribuido.

Una primera serie piloto fue descartada porque la configuración de SLURM hacía visibles las dos A100 de un nodo con dos GPUs al proceso de entrenamiento. Esas medidas se conservan únicamente con fines de diagnóstico y **no** forman parte de los resultados válidos de una sola GPU.

---

## 2. Entorno experimental

### 2.1 Stack software

| Componente | Versión |
|---|---|
| Python | 3.10.8 |
| PyTorch | 2.11.0+cu128 |
| CUDA build | 12.8 |
| Transformers | 5.17.0 |
| Datasets | 5.0.1 |
| Accelerate | 1.15.0 |
| GPU | NVIDIA A100-PCIE-40GB |

Crea el entorno virtual desde el directorio `BASELINE`:

```bash
module load python/3.10.8
python3 -m venv ../.venv
source ../.venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2.2 Preprocesamiento del dataset

El split completo de entrenamiento de SQuAD contiene **87,599 ejemplos originales**. Con la estrategia de tokenización mediante ventana deslizante utilizada, se generan **88,492 features de entrenamiento**.

Parámetros de preprocesamiento:

| Parámetro | Valor |
|---|---:|
| Longitud máxima de secuencia | 384 |
| Document stride | 128 |
| Padding | Fijo a la longitud máxima |

---

## 3. Configuración final de la baseline

La configuración de rendimiento seleccionada para una sola A100 es:

| Parámetro | Valor |
|---|---:|
| Modelo | `google-bert/bert-base-uncased` |
| Dataset | SQuAD v1.1 |
| GPU | 1 × NVIDIA A100-PCIE-40GB |
| Precisión | BF16 |
| Épocas | 1 |
| Batch size por dispositivo | 160 |
| Learning rate | `3e-5` |
| DataLoader workers | 0 |
| Gradient accumulation | 1 |
| Optimizador | AdamW fused (`adamw_torch_fused`) |
| Scheduler | Linear |
| `torch.compile` | Desactivado |
| TF32 | Comportamiento por defecto de PyTorch |

Una época completa ya dura claramente más de un minuto, por lo que una única época satisface el requisito de duración de la carga manteniendo razonables las repeticiones del benchmark.

---

## 4. Garantía de ejecución en una sola GPU en CESGA

La configuración oficial de SLURM solicita una A100 y 32 cores de CPU. El proceso Python se lanza mediante un paso `srun` explícito:

```bash
srun \
    --ntasks=1 \
    --cpus-per-task=32 \
    --gres=gpu:a100:1 \
    python train.py ...
```

Este detalle es importante en CESGA. Algunos trabajos piloto utilizaban `--exclusive` a nivel de nodo, lo que hacía visibles ambas A100 de un nodo con dos GPUs al shell del batch. Como consecuencia, Hugging Face Trainer podía detectar dos GPUs aunque el trabajo hubiera solicitado una sola.

La implementación final utiliza dos salvaguardas:

1. El aislamiento de GPU se aplica en el paso `srun`.
2. `train.py` aborta si no hay **exactamente un dispositivo CUDA** visible en ejecución.

Esto garantiza que todos los tiempos de baseline reportados corresponden a una única A100.

---

## 5. Metodología de temporización

El benchmark principal mide únicamente el tiempo de pared de:

```python
trainer.train()
```

Se realiza una sincronización CUDA inmediatamente antes y después de la región temporizada.

Por tanto, el tiempo de entrenamiento reportado excluye:

- carga del modelo;
- descarga y carga del dataset;
- tokenización y preprocesamiento;
- serialización final del modelo.

De este modo se aísla la fase de fine-tuning y se facilita la comparación entre repeticiones.

---

## 6. Estudio de optimización de rendimiento

### 6.1 Batch size

Con la configuración de precisión original, aumentar el batch size de 48 a 96 y 128 no mejoró de forma material el throughput. El tiempo se mantuvo cerca de 59 s sobre el subconjunto controlado de 4,096 ejemplos, mientras que el consumo de memoria GPU aumentó de forma considerable.

Tras habilitar BF16, el batch size fue ajustado de nuevo. Se seleccionó **160** como punto de equilibrio práctico entre throughput y memoria: batch 192 no aportó prácticamente ninguna mejora adicional de velocidad y consumió más memoria GPU.

### 6.2 DataLoader workers

Utilizar 0, 2 o 4 workers produjo tiempos prácticamente idénticos, alrededor de 59.2 s sobre el mismo subconjunto controlado.

**Conclusión:** el pipeline de entrada no constituía un cuello de botella relevante, por lo que se mantuvo `num_workers=0` por simplicidad y reproducibilidad.

### 6.3 BF16

BF16 produjo la mejora de rendimiento más importante medida.

| Precisión | Tiempo medio | Memoria GPU asignada máxima |
|---|---:|---:|
| Precisión por defecto | ~59.64 s | ~12.46 GiB |
| BF16 | ~12.26 s | ~8.61 GiB |

Esto corresponde aproximadamente a un **speedup de 4.87×** y a una reducción importante del uso de memoria.

### 6.4 TF32

TF32 aceleró considerablemente la ruta de precisión por defecto:

| Configuración | Tiempo medio |
|---|---:|
| TF32 desactivado | ~59.63 s |
| TF32 activado | ~19.55 s |

Sin embargo, BF16 siguió siendo más rápido y más eficiente en memoria, por lo que fue seleccionado para la baseline oficial.

### 6.5 `torch.compile`

`torch.compile` fue evaluado con BF16 y batch size 160.

| Modo | Tiempo medio |
|---|---:|
| Eager | ~87.30 s |
| Compilado | ~151.31 s |

Para la carga de una época evaluada, el coste de compilación dominó y volvió más lenta la versión compilada. Un experimento más largo de tres épocas redujo la penalización relativa, pero la ejecución compilada siguió siendo más lenta en tiempo total.

También se observó una recompilación cuando el último batch tenía una forma diferente. `drop_last=True` elimina ese batch irregular, pero la mejora no fue suficiente para justificar un cambio en la baseline oficial.

**Decisión:** mantener `torch.compile` desactivado en la ejecución de referencia.

---

## 7. Resultados de profiling

Una ejecución representativa BF16 con batch 160 fue muestreada aproximadamente una vez por segundo.

Observaciones principales:

| Métrica | Valor representativo |
|---|---:|
| Utilización GPU media | ~89.6% |
| Utilización GPU ≥80% | ~97.6% de las muestras GPU |
| Memoria GPU máxima | ~26.8 GiB |
| Potencia GPU media | ~238 W |

El profiling adicional recogió uso de memoria GPU, temperatura, clock de SM, consumo de potencia y memoria RSS del proceso de entrenamiento.

La utilización sostenida de GPU y la ausencia de mejoras medibles al aumentar los DataLoader workers apoyan la conclusión de que la fase temporizada de entrenamiento está **principalmente limitada por GPU**.

---

## 8. Resultados oficiales de la baseline

Se ejecutaron tres repeticiones válidas sobre SQuAD completo con la configuración final.

| Repetición | Tiempo de entrenamiento |
|---|---:|
| 1 | 233.48 s |
| 2 | 235.41 s |
| 3 | 234.85 s |

### Estadísticas resumen

| Estadística | Valor |
|---|---:|
| Media | **234.58 s** |
| Mediana | 234.85 s |
| Desviación estándar muestral | 0.99 s |
| Coeficiente de variación | **0.42%** |

Por tanto, la baseline final de una sola A100 es aproximadamente:

> **234.6 segundos = 3.91 minutos por época completa de SQuAD**

El bajo coeficiente de variación indica una buena estabilidad temporal entre repeticiones.

---

## 9. Reproducibilidad

Todas las ejecuciones válidas de rendimiento fuerzan exactamente una GPU CUDA visible.

La semilla aleatoria utilizada es **42**. Aun así, pueden existir pequeñas diferencias numéricas porque los kernels GPU y componentes internos del framework no están garantizados como deterministas bit a bit.

Los resultados generados, checkpoints, archivos TensorBoard, entornos virtuales y pesos de modelos se excluyen intencionadamente de Git.

### Estructura del repositorio

```text
BASELINE/
├── README.md
├── README_ES.md
├── requirements.txt
├── train.py
├── baseline.slurm
├── run_baseline.sh
├── profile.slurm
└── profile_run.sh
```

- `requirements.txt` — dependencias de Python fijadas para recrear el entorno virtual.
- `train.py` — implementación oficial de la baseline sobre una sola A100.
- `baseline.slurm` — trabajo SLURM oficial para la baseline.
- `run_baseline.sh` — wrapper de conveniencia para enviar el trabajo de baseline.
- `profile.slurm` — trabajo SLURM para la ejecución de profiling.
- `profile_run.sh` — wrapper de muestreo de recursos utilizado durante el profiling.

---

## 10. Ejecución

Desde el directorio `BASELINE`:

```bash
./run_baseline.sh
```

O directamente:

```bash
sbatch baseline.slurm
```

---

## 11. Conclusión

La configuración final de referencia es:

```text
BERT-base-uncased
SQuAD v1.1
1 × NVIDIA A100-PCIE-40GB
BF16
batch size 160
1 época
~234.6 s de tiempo de entrenamiento
```

La optimización principal fue BF16, que proporcionó aproximadamente un **speedup de 4.87×** sobre el subconjunto controlado. El paralelismo del DataLoader no produjo mejoras medibles y `torch.compile` no resultó ventajoso para este benchmark corto porque el coste de compilación dominó el tiempo total.

La baseline final es temporalmente estable, con un **coeficiente de variación del 0.42%**.

Por tanto, esta baseline proporciona simultáneamente:

- una referencia reproducible de **rendimiento HPC en una sola GPU**; y
