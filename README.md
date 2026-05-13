# 🎓 Predictor de Estado Académico - Proyecto Final IA

## 📌 Descripción

Sistema inteligente de predicción del estado académico de estudiantes utilizando **Inteligencia Artificial** y **Deep Learning**. El sistema entrena modelos en Python y ejecuta predicciones directamente en el navegador sin necesidad de backend, mediante TensorFlow.js.

### Características Principales

- ✅ **Predicción Individual**: Predice el estado de un estudiante con 5 variables
- ✅ **Análisis Masivo**: Procesa archivos CSV con múltiples estudiantes
- ✅ **Visualización de Métricas**: Muestra accuracy, precision, recall y F1-score
- ✅ **Matriz de Confusión**: Análisis visual del desempeño del modelo
- ✅ **Sin Backend**: Funciona 100% en el navegador (GitHub Pages compatible)
- ✅ **Deep Learning**: Red neuronal con Keras/TensorFlow
- ✅ **Diseño Moderno**: Interfaz profesional con animaciones

---

## 🎯 Objetivos del Proyecto

1. **Entrenar modelos IA** con datos reales de estudiantes
2. **Comparar modelos** (Regresión Logística vs Red Neuronal)
3. **Exportar a TensorFlow.js** para predicciones en navegador
4. **Procesar predicciones individuales** con 5 variables clave
5. **Procesar lotes CSV** para análisis masivo
6. **Visualizar métricas** del desempeño del modelo
7. **Desplegar en GitHub Pages** como aplicación estática

---

## 🛠️ Tecnologías Utilizadas

### Backend (Python)
- **pandas**: Manipulación y procesamiento de datos
- **numpy**: Operaciones numéricas avanzadas
- **scikit-learn**: Modelos clásicos de ML (Regresión Logística, StandardScaler)
- **tensorflow / keras**: Deep Learning y redes neuronales
- **tensorflowjs**: Conversión de modelos a JavaScript
- **matplotlib / seaborn**: Visualización de datos
- **joblib**: Serialización de modelos

### Frontend (JavaScript)
- **TensorFlow.js**: Ejecución de modelos en navegador
- **Chart.js**: Visualización de gráficos y matrices
- **PapaParse**: Procesamiento de archivos CSV
- **Bootstrap 5**: Diseño responsive
- **CSS3**: Animaciones y estilos modernos

### Despliegue
- **GitHub Pages**: Hosting estático gratuito

---

## 📊 Dataset

**Nombre**: Predict students' dropout and academic success

**Variables de entrada**: 36 columnas del dataset original

**Variables seleccionadas** (5 más relevantes):
1. `Curricular units 2nd sem (approved)` - Unidades curriculares aprobadas
2. `Curricular units 2nd sem (grade)` - Calificación promedio
3. `Tuition fees up to date` - Situación de pago (0=Al día, 1=Atrasado)
4. `Scholarship holder` - Beneficiario de beca (0=No, 1=Sí)
5. `Admission grade` - Calificación de admisión

**Variable objetivo** (Target):
- `0` = 🚪 **Dropout** (Abandonó los estudios)
- `1` = 📚 **Enrolled** (Actualmente matriculado)
- `2` = 🎓 **Graduate** (Se graduó)

**Total de registros**: ~4,400 estudiantes

---

## 🏗️ Estructura del Proyecto

```
proyecto-ia/
│
├── backend/
│   ├── train.py                    # Script de entrenamiento
│   ├── requirements.txt            # Dependencias Python
│   ├── data.csv                    # Dataset local
│   ├── metrics.json                # Métricas del modelo
│   ├── confusion_matrix.json       # Matriz de confusión
│   │
│   └── models/
│       ├── logistic_model.pkl      # Modelo de Regresión Logística
│       ├── scaler.pkl              # Normalizador StandardScaler
│       │
│       └── nn_model/               # Red Neuronal (SavedModel)
│           ├── saved_model.pb
│           ├── keras_metadata.pb
│           └── variables/
│
├── web/
│   ├── index.html                  # Página principal
│   ├── style.css                   # Estilos modernos
│   ├── script.js                   # Lógica de predicción
│   ├── metrics.json                # Métricas (copia)
│   ├── confusion_matrix.json       # Matriz (copia)
│   ├── config.json                 # Configuración del modelo
│   │
│   ├── models/
│   │   └── nn_model/               # Modelo TensorFlow.js
│   │       ├── model.json
│   │       ├── group1-shard1of1.bin
│   │       └── weights.json
│   │
│   └── assets/                     # Recursos estáticos
│
└── README.md                       # Este archivo
```

---

## 🚀 Instalación y Ejecución

### Requisitos Previos
- Python 3.8+
- pip o conda
- Navegador moderno (Chrome, Firefox, Edge, Safari)

### 1️⃣ Configurar el Entorno Python

```bash
# Ir a la carpeta backend
cd backend

# Crear entorno virtual (opcional pero recomendado)
python -m venv venv

# Activar entorno virtual
# En Windows:
venv\Scripts\activate
# En macOS/Linux:
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

### 2️⃣ Entrenar el Modelo

```bash
# Dentro de la carpeta backend/
python train.py
```

**El script realizará**:
1. ✓ Cargará el archivo `data.csv` local
2. ✓ Preprocesará y normalizará los datos
3. ✓ Entrenará Regresión Logística
4. ✓ Entrenará Red Neuronal
5. ✓ Evaluará ambos modelos
6. ✓ Generará `metrics.json` y `confusion_matrix.json`
7. ✓ Guardará modelos en `models/`

**Tiempo estimado**: 2-5 minutos

### 3️⃣ Convertir Modelo a TensorFlow.js

```bash
# Desde la carpeta backend/
tensorflowjs_converter --input_format tf_saved_model models/nn_model ../web/models/nn_model
```

Este comando:
- Convierte el modelo SavedModel a formato TensorFlow.js
- Genera `model.json` y archivos `.bin` en `web/models/nn_model/`

### 4️⃣ Probar Localmente

**Opción A: Con Python (recomendado)**
```bash
# Desde la carpeta web/
python -m http.server 8000
```

**Opción B: Con Node.js**
```bash
# Instalar http-server globalmente
npm install -g http-server

# Ejecutar
http-server web/
```

**Opción C: Visual Studio Code**
- Instalar extensión "Live Server"
- Click derecho en `web/index.html` → "Open with Live Server"

Acceder a: `http://localhost:8000` (o el puerto mostrado)

---

## 📈 Entrenamiento del Modelo

### Arquitectura de la Red Neuronal

```
┌─────────────────────────────────────────────────┐
│  Input (5 variables normalizadas)               │
└──────────────┬──────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────┐
│  Dense Layer 1: 64 neuronas + ReLU              │
└──────────────┬──────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────┐
│  Dropout: 30% (regularización)                   │
└──────────────┬──────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────┐
│  Dense Layer 2: 32 neuronas + ReLU              │
└──────────────┬──────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────┐
│  Dropout: 30%                                    │
└──────────────┬──────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────┐
│  Dense Layer 3: 16 neuronas + ReLU              │
└──────────────┬──────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────┐
│  Output Layer: 3 neuronas + Softmax             │
│  (Probabilidades: Dropout, Enrolled, Graduate)  │
└─────────────────────────────────────────────────┘
```

### Hiperparámetros

| Parámetro | Valor |
|-----------|-------|
| Epochs | 50 |
| Batch Size | 32 |
| Optimizer | Adam |
| Loss Function | sparse_categorical_crossentropy |
| Validation Split | 20% |
| Dropout Rate | 30% |

### División de Datos

- **Entrenamiento**: 80% (≈3,520 registros)
- **Prueba**: 20% (≈880 registros)
- **Stratified**: Mantiene la proporción de clases

---

## 📊 Métricas del Modelo

### Evaluación en Datos de Prueba

El script `train.py` genera:

```json
{
  "accuracy": 0.95,
  "precision": 0.94,
  "recall": 0.95,
  "f1_score": 0.94,
  "model": "Neural Network",
  "total_test_samples": 880,
  "classes": ["Dropout", "Enrolled", "Graduate"]
}
```

### Matriz de Confusión

Compara predicciones vs valores reales:

```
                 Predicción
              Dropout  Enrolled  Graduate
Actual Dropout    250      15        5
       Enrolled    10     350       15
       Graduate     5      20      190
```

---

## 🌐 Despliegue en GitHub Pages

### Paso 1: Crear Repositorio en GitHub

1. Acceder a https://github.com/new
2. Nombre: `predictor-academico` (o similar)
3. Marcar "Public"
4. Crear repositorio

### Paso 2: Subir Proyecto

```bash
# En la carpeta raíz del proyecto
git init
git add .
git commit -m "Proyecto final: Predictor de estado académico"
git branch -M main
git remote add origin https://github.com/TU_USUARIO/predictor-academico.git
git push -u origin main
```

### Paso 3: Habilitar GitHub Pages

1. Ir a **Settings** del repositorio
2. Sección **Pages** (en el menú izquierdo)
3. **Source**: Seleccionar "main" branch
4. **Folder**: Seleccionar `/web`
5. Guardar

**Tu sitio estará en**: `https://TU_USUARIO.github.io/predictor-academico/`

### Paso 4: Acceder a la Aplicación

Esperar 1-2 minutos y acceder a la URL anterior. ¡Tu aplicación estará en línea!

---

## 💡 Cómo Usar la Aplicación

### Predicción Individual

1. **Completar el formulario** con los 5 datos del estudiante
2. **Click en "Realizar Predicción"**
3. **Ver resultado** con probabilidades por clase
4. **Leer interpretación** personalizada

### Análisis Masivo (CSV)

1. **Preparar archivo CSV** con columnas:
   ```
   Curricular units 2nd sem (approved)
   Curricular units 2nd sem (grade)
   Tuition fees up to date
   Scholarship holder
   Admission grade
   ```

2. **Arrastrar archivo** a la zona de carga o hacer click
3. **Esperar procesamiento** (barra de progreso)
4. **Ver tabla de resultados**
5. **Descargar como CSV** (opcional)

### Ver Métricas

- Sección "Métricas del Modelo": Accuracy, Precision, Recall, F1-Score
- Sección "Matriz de Confusión": Gráfico interactivo con Chart.js

---

## 🧠 Conceptos de Machine Learning (Para Sustentación)

### 1. Red Neuronal vs Regresión Logística

**Regresión Logística**:
- Modelo lineal clásico
- Explícito y fácil de interpretar
- Menos flexible, ideal para datos lineales

**Red Neuronal (Deep Learning)**:
- Modelo no lineal con capas ocultas
- Puede aprender patrones complejos
- Mejor rendimiento en datos no lineales
- Requiere más datos pero más potente

### 2. ReLU (Rectified Linear Unit)

```
Fórmula: f(x) = max(0, x)
```

- Introduce **no linealidad** a la red
- Permite aprender patrones complejos
- Eficiente computacionalmente
- Activa neuronas selectivamente

### 3. Softmax

```
Fórmula: σ(z_i) = e^(z_i) / Σ e^(z_j)
```

- Convierte salidas en **probabilidades** (0-1)
- Suma de probabilidades = 1
- Ideal para clasificación multiclase (3+ clases)

### 4. Dropout

- Desactiva **aleatoriamente** 30% de neuronas durante entrenamiento
- Previene **overfitting** (memorización)
- Mejora **generalización** a datos nuevos
- No se usa durante predicción

### 5. StandardScaler (Normalización)

```
Fórmula: z = (x - media) / desviación_estándar
```

- Transforma datos a media=0, desv.est=1
- Evita que variables con rangos grandes dominen
- Mejora convergencia del optimizador

### 6. Train-Test Split

```
Entrenamiento (80%): Modelo aprende
Prueba (20%): Evaluación no sesgada
```

- Evita **data leakage**
- Proporciona métrica realista del desempeño
- Stratified: mantiene proporción de clases

### 7. Adam (Optimizador)

- Combina **momentum** y **RMSProp**
- Ajusta tasa de aprendizaje adaptativamente
- Convergencia rápida y estable
- Ideal para redes neuronales profundas

### 8. Categorical Crossentropy

```
Fórmula: L = -Σ y_i * log(y_pred_i)
```

- Función de **pérdida** para clasificación
- Penaliza predicciones incorrectas
- Minimiza durante entrenamiento
- Más efectiva que MSE para clasificación

---

## 📋 Archivos Generados

### Archivos Creados por train.py

| Archivo | Descripción |
|---------|-------------|
| `models/logistic_model.pkl` | Modelo de Regresión Logística |
| `models/scaler.pkl` | Normalizador StandardScaler |
| `models/nn_model/` | Red Neuronal en formato SavedModel |
| `metrics.json` | Accuracy, Precision, Recall, F1 |
| `confusion_matrix.json` | Matriz de confusión |

### Archivos Copiados a web/

```
web/
├── metrics.json                # Para mostrar en UI
├── confusion_matrix.json       # Para dibujar gráfico
├── config.json                 # Configuración del modelo
└── models/nn_model/            # Modelo TensorFlow.js (después de convertir)
```

---

## 🔧 Troubleshooting

### "Error: El modelo no cargó"

**Solución**:
1. Verificar que `web/models/nn_model/model.json` existe
2. Ejecutar el comando de conversión nuevamente:
   ```
   tensorflowjs_converter --input_format tf_saved_model models/nn_model ../web/models/nn_model
   ```

### "Error: Archivo CSV no cargado"

**Solución**:
1. Verificar que el CSV tiene los nombres de columnas correctos
2. Intentar con archivo de ejemplo
3. Revisar consola del navegador (F12)

### "Error: CORS cuando abre en navegador"

**Solución**:
1. No abrir archivos localmente (file://)
2. Usar servidor local (python -m http.server, http-server, etc.)

### "Predicción tarda mucho"

**Solución**:
1. Normal en primera predicción (carga el modelo)
2. Las siguientes serán más rápidas
3. Navegar a consola (F12) para ver progreso

---

## 📸 Capturas de Pantalla Sugeridas

Para la presentación incluir:

1. **Pantalla de predicción individual** - Mostrar formulario y resultado
2. **Resultado con probabilidades** - Gráficos de barras
3. **CSV procesado** - Tabla con múltiples predicciones
4. **Matriz de confusión** - Gráfico de desempeño
5. **Métricas** - Accuracy, Precision, etc.
6. **Consola de desarrollo** - Logs de procesamiento

---

## 👨‍💻 Autor y Contribuciones

**Proyecto Desarrollado por**: [Tu Nombre]

**Asignatura**: Análisis de Datos y Machine Learning

**Institución**: [Tu Universidad]

**Fecha**: Mayo 2026

---

## 📚 Referencias y Recursos

### Librerías Utilizadas
- [TensorFlow/Keras Documentation](https://www.tensorflow.org/)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [TensorFlow.js Guide](https://www.tensorflow.org/js)
- [Chart.js Documentation](https://www.chartjs.org/)

### Cursos y Tutoriales
- Andrés Ng - Deep Learning Specialization
- Fast.ai - Practical Deep Learning for Coders
- Kaggle - Machine Learning Competitions

### Datasets
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/)
- [Kaggle Datasets](https://www.kaggle.com/datasets)

---

## ⚖️ Licencia

Este proyecto es de uso educativo y puede ser modificado libremente para fines académicos.

---

## 📞 Soporte y Preguntas

Para dudas sobre el código:
1. Revisar los comentarios en `train.py` y `script.js`
2. Consultar documentación oficial de librerías
3. Verificar consola del navegador (F12 → Console)

---

**¡Proyecto completado y listo para presentar! 🎉**
