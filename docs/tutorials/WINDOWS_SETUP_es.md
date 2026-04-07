# 🖥️ Guía de Configuración en Windows — PaddleDetection desde Cero

> **Esta guía está dirigida a personas que nunca han usado PaddleDetection antes.**
> Sigue los pasos en orden y tendrás el entorno funcionando en menos de 30 minutos.

---

## 📋 Índice

1. [Requisitos del Sistema](#1-requisitos-del-sistema)
2. [Instalar Python](#2-instalar-python)
3. [Instalar Git](#3-instalar-git)
4. [Crear un Entorno Virtual (Recomendado)](#4-crear-un-entorno-virtual-recomendado)
5. [Clonar PaddleDetection](#5-clonar-paddledetection)
6. [Instalar PaddlePaddle](#6-instalar-paddlepaddle)
7. [Instalar Dependencias de PaddleDetection](#7-instalar-dependencias-de-paddledetection)
8. [Verificar la Instalación](#8-verificar-la-instalación)
9. [Ejecutar tu Primera Detección](#9-ejecutar-tu-primera-detección)
10. [Entrenamiento Rápido (Ejemplo)](#10-entrenamiento-rápido-ejemplo)
11. [Solución de Problemas Comunes](#11-solución-de-problemas-comunes)

---

## 1. Requisitos del Sistema

Antes de comenzar, verifica que tu computadora cumpla con los requisitos mínimos:

| Componente | Mínimo | Recomendado |
|-----------|--------|-------------|
| Sistema Operativo | Windows 10 64-bit | Windows 10/11 64-bit |
| RAM | 8 GB | 16 GB o más |
| Disco duro (espacio libre) | 10 GB | 20 GB o más |
| Python | 3.7 | 3.8 o 3.9 |
| GPU (opcional) | — | NVIDIA con CUDA 10.2+ |

> ✅ PaddleDetection funciona **sin GPU**. Si no tienes una GPU NVIDIA, puedes usar solo la CPU (el procesamiento será más lento, pero funciona perfectamente para aprender).

---

## 2. Instalar Python

### Paso 2.1 — Descargar Python

1. Ve a [https://www.python.org/downloads/windows/](https://www.python.org/downloads/windows/)
2. Descarga **Python 3.8.x** o **Python 3.9.x** (versión de 64 bits — busca el instalador que diga "Windows installer (64-bit)")

   > ⚠️ **Importante**: Se recomienda Python 3.8 o 3.9 para mayor estabilidad. Python 3.10 también es compatible según la documentación oficial, pero se han reportado problemas con algunas dependencias en Windows.

### Paso 2.2 — Instalar Python

1. Ejecuta el instalador descargado (doble clic)
2. **MUY IMPORTANTE**: Marca la casilla ✅ **"Add Python X.X to PATH"** en la pantalla inicial
3. Haz clic en **"Install Now"**
4. Espera a que termine la instalación
5. Haz clic en **"Close"**

### Paso 2.3 — Verificar Python

Abre el **Símbolo del sistema** (busca "cmd" en el menú inicio) y escribe:

```cmd
python --version
```

Deberías ver algo como:
```
Python 3.8.18
```

Si ves un error, asegúrate de haber marcado "Add Python to PATH" durante la instalación. Si el problema persiste, reinicia el computador.

---

## 3. Instalar Git

Git es necesario para clonar el repositorio de PaddleDetection.

### Paso 3.1 — Descargar Git

1. Ve a [https://git-scm.com/download/win](https://git-scm.com/download/win)
2. Descarga el instalador de **64 bits**

### Paso 3.2 — Instalar Git

1. Ejecuta el instalador
2. Acepta todas las opciones por defecto (simplemente haz clic en "Next" en cada pantalla)
3. En la pantalla "Choosing the default editor", selecciona el editor de tu preferencia (puedes dejarlo en Vim si no sabes cuál elegir)
4. Haz clic en **"Install"** y luego en **"Finish"**

### Paso 3.3 — Verificar Git

```cmd
git --version
```

Deberías ver algo como:
```
git version 2.42.0.windows.2
```

---

## 4. Crear un Entorno Virtual (Recomendado)

Un entorno virtual es un "espacio aislado" para instalar paquetes de Python sin afectar otras instalaciones. Esto es muy recomendable para evitar conflictos entre paquetes.

### Opción A: Usando `venv` (más sencillo)

Abre el **Símbolo del sistema** y ejecuta:

```cmd
# Crear el entorno virtual en una carpeta llamada 'paddle_env'
python -m venv C:\paddle_env

# Activar el entorno virtual
C:\paddle_env\Scripts\activate
```

Sabrás que el entorno está activo cuando veas `(paddle_env)` al inicio de la línea en el símbolo del sistema:
```
(paddle_env) C:\Users\TuNombre>
```

> 💡 **Para desactivar el entorno virtual:** escribe `deactivate`
> 💡 **Cada vez que abras una nueva ventana de CMD**, debes activar el entorno de nuevo con: `C:\paddle_env\Scripts\activate`

### Opción B: Usando Conda/Miniconda (para usuarios más avanzados)

Si prefieres usar Conda:

1. Descarga Miniconda desde [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)
2. Instala Miniconda (acepta las opciones por defecto)
3. Abre **Anaconda Prompt** y ejecuta:

```bash
# Crear entorno conda
conda create -n paddle_env python=3.8 -y

# Activar entorno
conda activate paddle_env
```

---

## 5. Clonar PaddleDetection

Con el entorno virtual activo, ejecuta estos comandos en el **Símbolo del sistema**:

```cmd
# Navegar a la carpeta donde quieres instalar PaddleDetection
# Por ejemplo, en el escritorio:
cd C:\Users\%USERNAME%\Desktop

# Clonar el repositorio
git clone https://github.com/PaddlePaddle/PaddleDetection.git

# Entrar a la carpeta del proyecto
cd PaddleDetection
```

> 💡 Si la descarga es lenta, puedes usar un mirror alternativo:
> ```cmd
> git clone https://gitee.com/paddlepaddle/PaddleDetection.git
> ```

---

## 6. Instalar PaddlePaddle

PaddlePaddle es el framework de deep learning en el que se basa PaddleDetection.

### Opción A: Sin GPU (solo CPU)

```cmd
python -m pip install paddlepaddle==2.3.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Opción B: Con GPU NVIDIA

Primero verifica qué versión de CUDA tienes instalada:

```cmd
nvcc --version
```

Luego instala la versión correspondiente:

**CUDA 10.2:**
```cmd
python -m pip install paddlepaddle-gpu==2.3.2.post102 -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html
```

**CUDA 11.2:**
```cmd
python -m pip install paddlepaddle-gpu==2.3.2.post112 -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html
```

**CUDA 11.6:**
```cmd
python -m pip install paddlepaddle-gpu==2.3.2.post116 -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html
```

> 📌 Para otras versiones de CUDA, visita: [https://www.paddlepaddle.org.cn/install/quick](https://www.paddlepaddle.org.cn/install/quick)

### Verificar la instalación de PaddlePaddle

```cmd
python -c "import paddle; paddle.utils.run_check()"
```

Si la instalación fue exitosa, verás:
```
PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.
```

---

## 7. Instalar Dependencias de PaddleDetection

Asegúrate de estar dentro de la carpeta `PaddleDetection` y con el entorno virtual activado.

### Paso 7.1 — Instalar pycocotools para Windows

En Windows, `pycocotools` requiere una versión especial. Instálala primero:

```cmd
pip install cython
pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
```

> ⚠️ Si el comando anterior falla porque no tienes Git configurado en pip, intenta:
> ```cmd
> pip install pycocotools-windows
> ```

### Paso 7.2 — Instalar el resto de dependencias

```cmd
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> 💡 El flag `-i https://pypi.tuna.tsinghua.edu.cn/simple` usa un espejo de China para descargas más rápidas. Si tienes buena conexión internacional, puedes omitirlo.

### Paso 7.3 — Compilar e instalar paddledet

```cmd
python setup.py install
```

Este proceso puede tardar 1-2 minutos. Verás varios mensajes de compilación, lo cual es normal.

---

## 8. Verificar la Instalación

Ejecuta las pruebas de arquitectura para confirmar que todo funciona:

```cmd
python ppdet/modeling/tests/test_architectures.py
```

Si todo está correcto, verás:
```
.......
----------------------------------------------------------------------
Ran 7 tests in 12.816s
OK
```

¡Si ves `OK` al final, la instalación fue exitosa! 🎉

---

## 9. Ejecutar tu Primera Detección

Ahora vamos a detectar objetos en una imagen de prueba que ya viene incluida en el repositorio.

### Con GPU:

```cmd
set CUDA_VISIBLE_DEVICES=0
python tools/infer.py -c configs/ppyolo/ppyolo_r50vd_dcn_1x_coco.yml -o use_gpu=true weights=https://paddledet.bj.bcebos.com/models/ppyolo_r50vd_dcn_1x_coco.pdparams --infer_img=demo/000000014439.jpg
```

### Sin GPU (CPU):

```cmd
python tools/infer.py -c configs/ppyolo/ppyolo_r50vd_dcn_1x_coco.yml -o use_gpu=false weights=https://paddledet.bj.bcebos.com/models/ppyolo_r50vd_dcn_1x_coco.pdparams --infer_img=demo/000000014439.jpg
```

> 📥 La primera vez que ejecutes este comando, se descargará automáticamente el modelo pre-entrenado (puede tardar unos minutos dependiendo de tu conexión).

El resultado se guardará en la carpeta `output/`. Abre el archivo `000000014439.jpg` en esa carpeta para ver los objetos detectados.

---

## 10. Entrenamiento Rápido (Ejemplo)

¡Vamos a entrenar un modelo de detección de señales de tráfico en solo 10 minutos!

### Paso 10.1 — Descargar el dataset de señales de tráfico

```cmd
python dataset/roadsign_voc/download_roadsign_voc.py
```

### Paso 10.2 — Entrenar el modelo

**Con GPU:**
```cmd
python tools/train.py -c configs/yolov3/yolov3_mobilenet_v1_roadsign.yml --eval -o use_gpu=true
```

**Sin GPU (CPU) — tomará más tiempo (~1 hora):**
```cmd
python tools/train.py -c configs/yolov3/yolov3_mobilenet_v1_roadsign.yml --eval -o use_gpu=false
```

### Paso 10.3 — Evaluar el modelo

```cmd
python tools/eval.py -c configs/yolov3/yolov3_mobilenet_v1_roadsign.yml -o use_gpu=false
```

### Paso 10.4 — Realizar inferencia con tu modelo entrenado

```cmd
python tools/infer.py -c configs/yolov3/yolov3_mobilenet_v1_roadsign.yml -o use_gpu=false --infer_img=demo/road554.png
```

---

## 11. Solución de Problemas Comunes

### ❌ Error: `'python' is not recognized as an internal or external command`

**Causa:** Python no está en el PATH del sistema.

**Solución:**
1. Desinstala Python
2. Reinstala marcando ✅ **"Add Python to PATH"**
3. Reinicia el símbolo del sistema

---

### ❌ Error: `pip is not recognized as an internal or external command`

**Causa:** pip no está en el PATH.

**Solución:** Usa `python -m pip` en lugar de solo `pip`:
```cmd
python -m pip install paddlepaddle==2.3.2
```

---

### ❌ Error al instalar `pycocotools`

**Causa:** pycocotools estándar no es compatible con Windows.

**Solución:**
```cmd
pip install pycocotools-windows
```
O bien:
```cmd
pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
```

---

### ❌ Error: `Microsoft Visual C++ 14.0 is required`

**Causa:** Falta el compilador de C++ de Microsoft necesario para compilar algunas dependencias.

**Solución:**
1. Descarga **Build Tools for Visual Studio** desde [https://visualstudio.microsoft.com/visual-cpp-build-tools/](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. Durante la instalación, selecciona **"C++ build tools"**
3. Instala y reinicia el equipo

---

### ❌ Error: `CUDA out of memory` / La GPU se queda sin memoria

**Causa:** El modelo o el batch size es demasiado grande para tu GPU.

**Solución:** Reduce el `batch_size` en el archivo de configuración:

Abre el archivo `.yml` del modelo y busca `batch_size`, luego redúcelo a la mitad:
```yaml
# Cambiar de:
batch_size: 8
# A:
batch_size: 4
```

O pasa el parámetro directamente:
```cmd
python tools/train.py -c configs/yolov3/yolov3_mobilenet_v1_roadsign.yml -o batch_size=2
```

---

### ❌ La descarga del modelo es muy lenta o falla

**Causa:** Conectividad lenta a servidores en China.

**Solución:** Descarga los modelos manualmente desde el [Zoo de Modelos](../../docs/MODEL_ZOO_en.md) y guárdalos localmente. Luego usa la ruta local en el parámetro `weights`:

```cmd
python tools/infer.py -c configs/ppyolo/ppyolo_r50vd_dcn_1x_coco.yml -o use_gpu=false weights=C:\ruta\al\modelo\ppyolo_r50vd_dcn_1x_coco.pdparams --infer_img=demo/000000014439.jpg
```

---

### ❌ Error en `setup.py install`: `error: command 'cl.exe' failed`

**Causa:** No tienes las herramientas de compilación de Visual C++ instaladas.

**Solución:** Igual que el error de `Microsoft Visual C++ 14.0` — instala las Build Tools de Visual Studio.

---

### ❌ `ImportError: DLL load failed` al importar paddle

**Causa:** Incompatibilidad entre la versión de PaddlePaddle y la versión de CUDA/cuDNN instalada.

**Solución:**
1. Verifica tu versión de CUDA: `nvcc --version`
2. Instala la versión de PaddlePaddle correspondiente a tu CUDA (ver [Paso 6](#6-instalar-paddlepaddle))
3. Asegúrate de tener instalado cuDNN compatible

---

## 🎓 Próximos Pasos

¡Felicitaciones! Ya tienes PaddleDetection funcionando en Windows. Aquí hay algunos recursos para continuar aprendiendo:

| Recurso | Descripción |
|---------|-------------|
| [GETTING_STARTED_es.md](GETTING_STARTED_es.md) | Guía completa de entrenamiento, evaluación e inferencia |
| [QUICK_STARTED_es.md](QUICK_STARTED_es.md) | Tutorial de 10 minutos con dataset real |
| [Zoo de Modelos](../../docs/MODEL_ZOO_en.md) (en inglés) | Todos los modelos disponibles con sus métricas |
| [Preparación de Datos](./data/PrepareDetDataSet_en.md) | Cómo preparar tu propio dataset |
| [README_es.md](../../README_es.md) | README principal en español |

---

## 💬 ¿Necesitas Ayuda?

Si tienes problemas que no están cubiertos en esta guía:

1. Revisa los [Issues de GitHub](https://github.com/PaddlePaddle/PaddleDetection/issues) — quizás alguien ya tuvo el mismo problema
2. Crea un nuevo Issue describiendo:
   - Tu versión de Windows
   - Tu versión de Python (`python --version`)
   - Tu versión de PaddlePaddle (`python -c "import paddle; print(paddle.__version__)"`)
   - El error completo que ves en la pantalla
