[English](GETTING_STARTED.md) | [简体中文](GETTING_STARTED_cn.md) | Español

# Comenzando con PaddleDetection

## Instalación

Para configurar el entorno de ejecución, consulta las [instrucciones de instalación](INSTALL_es.md).

> 💡 **¿Estás en Windows?** Consulta la [Guía de Configuración en Windows](WINDOWS_SETUP_es.md) para instrucciones paso a paso.

---

## Preparación de Datos

- Consulta [PrepareDetDataSet](./data/PrepareDetDataSet_en.md) para la preparación de datos.
- Configura la ruta de datos en los archivos de configuración del dataset en `configs/datasets`.

---

## Entrenamiento, Evaluación e Inferencia

PaddleDetection proporciona scripts para entrenamiento, evaluación e inferencia con diversas funcionalidades. Para detalles sobre entrenamiento distribuido, consulta [DistributedTraining](./DistributedTraining_en.md).

```bash
# Entrenamiento en una sola GPU
export CUDA_VISIBLE_DEVICES=0
python tools/train.py -c configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.yml

# Entrenamiento en múltiples GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 \
    tools/train.py -c configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.yml

# Evaluación con GPU
export CUDA_VISIBLE_DEVICES=0
python tools/eval.py \
    -c configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.yml \
    -o weights=https://paddledet.bj.bcebos.com/models/faster_rcnn_r50_fpn_1x_coco.pdparams

# Inferencia
python tools/infer.py \
    -c configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.yml \
    --infer_img=demo/000000570688.jpg \
    -o weights=https://paddledet.bj.bcebos.com/models/faster_rcnn_r50_fpn_1x_coco.pdparams
```

> 💡 **En Windows**, reemplaza `export VARIABLE=value` por `set VARIABLE=value` en el símbolo del sistema (CMD), o usa `$env:VARIABLE="value"` en PowerShell.

---

## Lista de Argumentos

Los siguientes argumentos pueden consultarse con `--help`:

| Argumento | Scripts | Descripción | Por defecto | Observaciones |
|:---------:|:-------:|:-----------:|:-----------:|:-------------:|
| `-c` | TODOS | Seleccionar archivo de configuración | Ninguno | **Obligatorio**, ej: `-c configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.yml` |
| `-o` | TODOS | Establecer parámetros en el archivo de configuración | Ninguno | `-o` tiene mayor prioridad que `-c`. Ej: `-o use_gpu=False` |
| `--eval` | train | Si realizar evaluación durante el entrenamiento | False | Usa `--eval` si es necesario |
| `-r/--resume_checkpoint` | train | Ruta del checkpoint para reanudar entrenamiento | Ninguno | Ej: `-r output/faster_rcnn_r50_1x_coco/10000` |
| `--slim_config` | TODOS | Archivo de configuración del método slim | Ninguno | Ej: `--slim_config configs/slim/prune/yolov3_prune_l1_norm.yml` |
| `--use_vdl` | train/infer | Si registrar datos con VisualDL | False | VisualDL requiere Python >= 3.5 |
| `--vdl_log_dir` | train/infer | Directorio de logs de VisualDL | train: `vdl_log_dir/scalar` infer: `vdl_log_dir/image` | VisualDL requiere Python >= 3.5 |
| `--output_eval` | eval | Directorio para almacenar la salida de evaluación | Ninguno | Ej: `--output_eval=eval_output` |
| `--json_eval` | eval | Si evaluar con bbox.json o mask.json existentes | False | Configura `--json_eval` si es necesario |
| `--classwise` | eval | Si evaluar AP por clase y trazar curva PR | False | Configura `--classwise` si es necesario |
| `--output_dir` | infer | Directorio para almacenar archivos de visualización | `./output` | Ej: `--output_dir output` |
| `--draw_threshold` | infer | Umbral para reservar resultados de visualización | 0.5 | Ej: `--draw_threshold 0.7` |
| `--infer_dir` | infer | Directorio de imágenes para inferencia | Ninguno | Se requiere `infer_dir` o `infer_img` |
| `--infer_img` | infer | Ruta de la imagen | Ninguno | `infer_img` tiene mayor prioridad que `infer_dir` |
| `--save_results` | infer | Si guardar resultados de detección en archivo | False | Opcional |

---

## Ejemplos

### Entrenamiento

#### Evaluar durante el entrenamiento

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 \
    tools/train.py -c configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.yml --eval
```

Realiza entrenamiento y evaluación alternadamente y evalúa al final de cada época. El mejor modelo con el MAP más alto se guarda en cada época con la misma ruta que `model_final`.

> Si el dataset de evaluación es grande, se sugiere modificar `snapshot_epoch` en `configs/runtime.yml` para reducir los tiempos de evaluación.

#### Fine-tuning (ajuste fino) para otra tarea

Cuando se usa un modelo pre-entrenado para fine-tuning, se puede usar `pretrain_weights` directamente. Los parámetros con forma diferente se ignorarán automáticamente:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 \
    tools/train.py -c configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.yml \
    -o pretrain_weights=output/faster_rcnn_r50_1x_coco/model_final
```

**Notas importantes:**
- `CUDA_VISIBLE_DEVICES` puede especificar diferentes números de GPU. Ej: `export CUDA_VISIBLE_DEVICES=0,1,2,3`
- El dataset se descarga automáticamente y se almacena en `~/.cache/paddle/dataset` si no se encuentra localmente
- El modelo pre-entrenado se descarga automáticamente y se almacena en `~/.cache/paddle/weights`
- Los checkpoints se guardan en `output` por defecto, y puede cambiarse desde `save_dir` en `configs/runtime.yml`

---

### Evaluación

#### Evaluar con ruta de pesos y dataset especificados

```bash
export CUDA_VISIBLE_DEVICES=0
python -u tools/eval.py \
    -c configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.yml \
    -o weights=https://paddledet.bj.bcebos.com/models/faster_rcnn_r50_fpn_1x_coco.pdparams
```

La ruta del modelo a evaluar puede ser tanto una ruta local como un enlace del [ZOO de Modelos](../MODEL_ZOO_cn.md).

#### Evaluar con JSON

```bash
export CUDA_VISIBLE_DEVICES=0
python tools/eval.py \
    -c configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.yml \
    --json_eval \
    --output_eval evaluation/
```

El archivo JSON debe llamarse `bbox.json` o `mask.json` y estar en el directorio `evaluation/`.

---

### Inferencia

#### Especificar directorio de salida y umbral

```bash
export CUDA_VISIBLE_DEVICES=0
python tools/infer.py \
    -c configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.yml \
    --infer_img=demo/000000570688.jpg \
    --output_dir=infer_output/ \
    --draw_threshold=0.5 \
    -o weights=output/faster_rcnn_r50_fpn_1x_coco/model_final \
    --use_vdl=True
```

`--draw_threshold` es un argumento opcional. El valor por defecto es 0.5. Diferentes umbrales producen diferentes resultados dependiendo del cálculo de [NMS](https://ieeexplore.ieee.org/document/1699659).

---

## Despliegue

Consulta la [documentación de despliegue](../../deploy/README.md).

## Compresión de Modelos

Consulta [slim](../../configs/slim/README_en.md).
