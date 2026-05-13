"""
================================================================================
PROYECTO FINAL - PREDICCIÓN DE ESTADO ACADÉMICO ESTUDIANTIL
Sistema de IA para clasificación multiclase: Dropout | Enrolled | Graduate
================================================================================

Este script entrena dos modelos de Machine Learning:
1. Regresión Logística (modelo clásico)
2. Red Neuronal (modelo profundo)

Todos los pasos incluyen explicaciones detalladas para defensa del proyecto.
================================================================================
"""

# ═══════════════════════════════════════════════════════════════════════════
# 0. VALIDACIONES PREVENTIVAS
# ═══════════════════════════════════════════════════════════════════════════

import sys
import os

print("\n" + "="*80)
print("VALIDACIONES PREVENTIVAS")
print("="*80 + "\n")

# Verificar versión de Python
python_version = sys.version_info
print(f"✓ Python versión: {python_version.major}.{python_version.minor}.{python_version.micro}")

if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
    print("❌ ERROR: Se requiere Python 3.8 o superior")
    sys.exit(1)

# Verificar existencia de data.csv
if not os.path.exists("data.csv"):
    print("❌ ERROR: No se encontró data.csv en la carpeta actual")
    print(f"   Ubicación esperada: {os.path.abspath('data.csv')}")
    sys.exit(1)
print("✓ data.csv encontrado")

# Verificar directorio web/
if not os.path.exists("../web"):
    print("❌ ERROR: No se encontró carpeta ../web")
    print(f"   Ubicación esperada: {os.path.abspath('../web')}")
    sys.exit(1)
print("✓ Carpeta ../web accesible")

# Verificar permisos de escritura
try:
    test_file = "test_write_permission.tmp"
    with open(test_file, 'w') as f:
        f.write("test")
    os.remove(test_file)
    print("✓ Permisos de escritura verificados")
except Exception as e:
    print(f"❌ ERROR: No hay permisos de escritura: {e}")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════════
# 1. IMPORTAR LIBRERÍAS ESENCIALES
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("IMPORTAR LIBRERÍAS")
print("="*80 + "\n")

try:
    import pandas as pd                          # Manipulación de datos
    import numpy as np                           # Operaciones numéricas
    import json                                  # Para exportar métricas
    from sklearn.preprocessing import LabelEncoder, StandardScaler  # Preprocesamiento
    from sklearn.model_selection import train_test_split           # División datos
    from sklearn.linear_model import LogisticRegression            # Modelo clásico
    from sklearn.metrics import (                                  # Evaluación
        accuracy_score, 
        precision_score, 
        recall_score, 
        f1_score,
        confusion_matrix,
        classification_report
    )
    import sklearn
    import tensorflow as tf                      # Framework de Deep Learning
    from tensorflow import keras                 # Alto nivel de TensorFlow
    from tensorflow.keras import layers          # Capas de red neuronal
    import joblib                                # Serialización de modelos
    import warnings
    import subprocess
    warnings.filterwarnings('ignore')
    
    print("✓ Todas las librerías importadas correctamente")
    print(f"   - pandas: {pd.__version__}")
    print(f"   - numpy: {np.__version__}")
    print(f"   - scikit-learn: {sklearn.__version__}")
    print(f"   - TensorFlow: {tf.__version__}")
    
except ImportError as e:
    print(f"❌ ERROR al importar librerías: {e}")
    print("   Ejecuta: pip install -r requirements.txt")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════
# 2. CONFIGURACIÓN DE DIRECTORIOS
# ═══════════════════════════════════════════════════════════════════════════

# Crear directorios si no existen
os.makedirs("models", exist_ok=True)
os.makedirs("../web/models", exist_ok=True)

print("✓ Directorios configurados")


# ═══════════════════════════════════════════════════════════════════════════
# 3. CARGAR DATASET LOCAL
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("PASO 1: CARGAR Y EXPLORAR DATASET")
print("="*80 + "\n")

try:
    # Leer el archivo CSV local
    # IMPORTANTE: El dataset usa ';' como separador
    df = pd.read_csv("data.csv", sep=';')
    
    print(f"📊 Dataset cargado: {df.shape[0]} registros, {df.shape[1]} columnas")
    print(f"\nPrimeras filas del dataset:")
    print(df.head())

    print(f"\n📈 Distribución de clases (Target):")
    print(df['Target'].value_counts())

    print(f"\n📋 Información del dataset:")
    print(df.info())
    
    # Validar que existan las columnas necesarias
    required_columns = [
        'Curricular units 2nd sem (approved)',
        'Curricular units 2nd sem (grade)',
        'Tuition fees up to date',
        'Scholarship holder',
        'Admission grade',
        'Target'
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"\n❌ ERROR: Faltan columnas en el dataset:")
        for col in missing_columns:
            print(f"   - {col}")
        sys.exit(1)
    
    print("\n✓ Todas las columnas requeridas están presentes")
    
except FileNotFoundError:
    print(f"❌ ERROR: No se encontró data.csv")
    print(f"   Ubicación esperada: {os.path.abspath('data.csv')}")
    sys.exit(1)
except Exception as e:
    print(f"❌ ERROR al cargar dataset: {e}")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════
# 4. MAPEO DE CLASES (Label Encoding para Target)
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("PASO 2: PREPARACIÓN DE DATOS - LABEL ENCODING")
print("="*80 + "\n")

# DICCIONARIO DE CLASES: Mapeo de Target a números
class_mapping = {
    'Dropout': 0,
    'Enrolled': 1,
    'Graduate': 2
}

# Mapeo inverso para mostrar predicciones
reverse_class_mapping = {v: k for k, v in class_mapping.items()}

print("🏷️  Mapeo de clases:")
for class_name, class_id in class_mapping.items():
    count = (df['Target'] == class_name).sum()
    print(f"   {class_name:12} → {class_id}  (Total: {count} registros)")

# Aplicar encoding al Target
df['Target'] = df['Target'].map(class_mapping)

print(f"\n✓ Target codificado numéricamente")


# ═══════════════════════════════════════════════════════════════════════════
# 5. SELECCIONAR VARIABLES PARA EL MODELO
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("PASO 3: SELECCIONAR CARACTERÍSTICAS (FEATURES)")
print("="*80 + "\n")

# Solo usar las 5 variables especificadas en el proyecto
selected_features = [
    'Curricular units 2nd sem (approved)',
    'Curricular units 2nd sem (grade)',
    'Tuition fees up to date',
    'Scholarship holder',
    'Admission grade'
]

print("📌 Variables seleccionadas para el modelo:")
for i, feature in enumerate(selected_features, 1):
    print(f"   {i}. {feature}")

# Separar características (X) y variable objetivo (y)
X = df[selected_features].copy()
y = df['Target'].copy()

print(f"\n✓ Características seleccionadas: {X.shape[1]} variables")
print(f"✓ Registros disponibles: {X.shape[0]}")

# Verificar valores faltantes
print(f"\n🔍 Valores faltantes en características:")
print(X.isnull().sum())

# Rellenar valores faltantes con la media (estrategia común)
X = X.fillna(X.mean())
print(f"✓ Valores faltantes rellenados con la media")


# ═══════════════════════════════════════════════════════════════════════════
# 6. DIVIDIR DATOS EN ENTRENAMIENTO Y PRUEBA
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("PASO 4: DIVISIÓN DE DATOS")
print("="*80 + "\n")

# train_test_split divide los datos en:
# - 80% para entrenamiento
# - 20% para validación/prueba
# - random_state=42 para reproducibilidad

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,           # 20% para prueba
    random_state=42,         # Mismo resultado cada ejecución
    stratify=y               # Mantener proporción de clases
)

print(f"📊 División de datos:")
print(f"   Entrenamiento: {X_train.shape[0]} registros (80%)")
print(f"   Prueba:       {X_test.shape[0]} registros (20%)")


# ═══════════════════════════════════════════════════════════════════════════
# 7. NORMALIZACIÓN DE DATOS (StandardScaler)
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("PASO 5: NORMALIZACIÓN (STANDARDSCALER)")
print("="*80 + "\n")

# StandardScaler transforma los datos a media=0 y desviación estándar=1
# Fórmula: z = (x - media) / desviación_estándar
# Esto es importante porque:
# - Algunos algoritmos funciona mejor con datos normalizados
# - Evita que variables con rangos grandes dominen el modelo

scaler = StandardScaler()

# Ajustar scaler SOLO con datos de entrenamiento (no test para evitar data leakage)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"📏 StandardScaler aplicado")
print(f"   Media de X_train_scaled: {X_train_scaled.mean():.6f}")
print(f"   Desv. Est de X_train_scaled: {X_train_scaled.std():.6f}")

# Guardar el scaler para uso en predicciones futuras
joblib.dump(scaler, 'models/scaler.pkl')
print(f"✓ Scaler guardado en: models/scaler.pkl")


# ═══════════════════════════════════════════════════════════════════════════
# 8. ENTRENAR MODELO 1: REGRESIÓN LOGÍSTICA
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("PASO 6: ENTRENAR REGRESIÓN LOGÍSTICA")
print("="*80 + "\n")

# Regresión Logística:
# - Modelo clásico de clasificación
# - Usa función logística (sigmoid) para probabilidades
# - max_iter aumentado porque dataset es grande
# - solver='lbfgs' compatible con versiones actuales de scikit-learn

logistic_model = LogisticRegression(
    max_iter=1000,
    solver='lbfgs'
)

print("🔄 Entrenando Regresión Logística...")
logistic_model.fit(X_train_scaled, y_train)
print("✓ Entrenamiento completado")

# Guardar modelo
joblib.dump(logistic_model, 'models/logistic_model.pkl')
print("✓ Modelo guardado en: models/logistic_model.pkl")


# ═══════════════════════════════════════════════════════════════════════════
# 9. ENTRENAR MODELO 2: RED NEURONAL (DEEP LEARNING)
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("PASO 7: ENTRENAR RED NEURONAL (DEEP LEARNING)")
print("="*80 + "\n")

print("🏗️  Arquitectura de la Red Neuronal:")
print("""
Capa de entrada: 5 neuronas (una por cada variable)
    ↓
Capa densa 1: 64 neuronas + ReLU (Rectified Linear Unit)
    ↓
Dropout: 30% (regularización para evitar overfitting)
    ↓
Capa densa 2: 32 neuronas + ReLU
    ↓
Dropout: 30%
    ↓
Capa densa 3: 16 neuronas + ReLU
    ↓
Capa de salida: 3 neuronas + Softmax (probabilidades para 3 clases)
""")

# Construir la red neuronal con Keras
neural_model = keras.Sequential([
    # Capa de entrada (5 características)
    # Input shape: (5,) porque tenemos 5 variables
    layers.Dense(64, activation='relu', input_shape=(5,)),
    # ReLU: max(0, x) - deja pasar valores positivos, bloquea negativos
    # Introduce NO linealidad, esencial para aprender patrones complejos
    
    # Dropout: desactiva 30% de neuronas aleatoriamente durante entrenamiento
    # Previene overfitting y mejora generalización
    layers.Dropout(0.3),
    
    # Segunda capa oculta
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    
    # Tercera capa oculta
    layers.Dense(16, activation='relu'),
    
    # Capa de salida: 3 neuronas (una por cada clase)
    # Softmax: convierte salidas a probabilidades que suman 1
    # Fórmula: softmax(x_i) = e^x_i / sum(e^x_j)
    layers.Dense(3, activation='softmax')
])

print("\n📋 Resumen de la arquitectura:")
neural_model.summary()

# Compilar el modelo
# - optimizer='adam': algoritmo de optimización adaptativo
# - loss='sparse_categorical_crossentropy': función de pérdida para clasificación multiclase
# - metrics=['accuracy']: métrica a monitorear
neural_model.compile(
    optimizer='adam',  # Adam: combina momentum y RMSProp, muy efectivo
    loss='sparse_categorical_crossentropy',  # Para clasificación multiclase (targets como enteros)
    metrics=['accuracy']
)

print("\n🔄 Entrenando Red Neuronal...")
history = neural_model.fit(
    X_train_scaled, y_train,
    epochs=50,              # 50 pasadas sobre los datos
    batch_size=32,          # Procesar 32 registros por actualización
    validation_split=0.2,   # 20% para validación durante entrenamiento
    verbose=1
)
print("✓ Entrenamiento completado")

# Guardar el modelo en formato .keras (formato nativo recomendado)
neural_model.save('models/nn_model.keras')
print("✓ Modelo guardado en: models/nn_model.keras")


# ═══════════════════════════════════════════════════════════════════════════
# 10. EVALUAR MODELOS EN DATOS DE PRUEBA
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("PASO 8: EVALUACIÓN DE MODELOS")
print("="*80 + "\n")

# ─────────────────────────────────────────────────────────────────────────
# Evaluación Regresión Logística
# ─────────────────────────────────────────────────────────────────────────

print("📊 REGRESIÓN LOGÍSTICA")
print("-" * 60)

y_pred_logistic = logistic_model.predict(X_test_scaled)

accuracy_lr = accuracy_score(y_test, y_pred_logistic)
precision_lr = precision_score(y_test, y_pred_logistic, average='weighted', zero_division=0)
recall_lr = recall_score(y_test, y_pred_logistic, average='weighted', zero_division=0)
f1_lr = f1_score(y_test, y_pred_logistic, average='weighted', zero_division=0)
cm_lr = confusion_matrix(y_test, y_pred_logistic)

print(f"✓ Accuracy:  {accuracy_lr:.4f}")
print(f"✓ Precision: {precision_lr:.4f}")
print(f"✓ Recall:    {recall_lr:.4f}")
print(f"✓ F1-Score:  {f1_lr:.4f}")
print(f"\nMatriz de Confusión:")
print(cm_lr)

# ─────────────────────────────────────────────────────────────────────────
# Evaluación Red Neuronal
# ─────────────────────────────────────────────────────────────────────────

print("\n📊 RED NEURONAL")
print("-" * 60)

y_pred_nn = neural_model.predict(X_test_scaled, verbose=0)
y_pred_nn_classes = np.argmax(y_pred_nn, axis=1)  # Índice de máxima probabilidad

accuracy_nn = accuracy_score(y_test, y_pred_nn_classes)
precision_nn = precision_score(y_test, y_pred_nn_classes, average='weighted', zero_division=0)
recall_nn = recall_score(y_test, y_pred_nn_classes, average='weighted', zero_division=0)
f1_nn = f1_score(y_test, y_pred_nn_classes, average='weighted', zero_division=0)
cm_nn = confusion_matrix(y_test, y_pred_nn_classes)

print(f"✓ Accuracy:  {accuracy_nn:.4f}")
print(f"✓ Precision: {precision_nn:.4f}")
print(f"✓ Recall:    {recall_nn:.4f}")
print(f"✓ F1-Score:  {f1_nn:.4f}")
print(f"\nMatriz de Confusión:")
print(cm_nn)

# Comparación
print("\n" + "="*60)
print("COMPARACIÓN DE MODELOS")
print("="*60)
print(f"{'Métrica':<15} {'Logística':<15} {'Red Neuronal':<15}")
print("-" * 60)
print(f"{'Accuracy':<15} {accuracy_lr:<15.4f} {accuracy_nn:<15.4f}")
print(f"{'Precision':<15} {precision_lr:<15.4f} {precision_nn:<15.4f}")
print(f"{'Recall':<15} {recall_lr:<15.4f} {recall_nn:<15.4f}")
print(f"{'F1-Score':<15} {f1_lr:<15.4f} {f1_nn:<15.4f}")

mejor_modelo = "Red Neuronal" if accuracy_nn > accuracy_lr else "Regresión Logística"
print(f"\n🏆 Mejor modelo: {mejor_modelo}")


# ═══════════════════════════════════════════════════════════════════════════
# 11. GENERAR REPORTE DETALLADO DE CLASIFICACIÓN
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("PASO 9: REPORTE DETALLADO")
print("="*80 + "\n")

print("📈 REGRESIÓN LOGÍSTICA - Reporte detallado:")
print(classification_report(y_test, y_pred_logistic, 
                          target_names=['Dropout', 'Enrolled', 'Graduate']))

print("\n📈 RED NEURONAL - Reporte detallado:")
print(classification_report(y_test, y_pred_nn_classes,
                          target_names=['Dropout', 'Enrolled', 'Graduate']))


# ═══════════════════════════════════════════════════════════════════════════
# 12. EXPORTAR MÉTRICAS A JSON PARA FRONTEND
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("PASO 10: EXPORTAR DATOS PARA FRONTEND")
print("="*80 + "\n")

# Usar métricas del mejor modelo (Red Neuronal)
metrics_data = {
    "accuracy": float(accuracy_nn),
    "precision": float(precision_nn),
    "recall": float(recall_nn),
    "f1_score": float(f1_nn),
    "model": "Neural Network",
    "total_test_samples": int(len(y_test)),
    "classes": ["Dropout", "Enrolled", "Graduate"]
}

# Guardar métricas en JSON
with open('metrics.json', 'w') as f:
    json.dump(metrics_data, f, indent=4)

# También copiar a web/
with open('../web/metrics.json', 'w') as f:
    json.dump(metrics_data, f, indent=4)

print("✓ metrics.json generado")
print(json.dumps(metrics_data, indent=2))


# ═══════════════════════════════════════════════════════════════════════════
# 13. EXPORTAR MATRIZ DE CONFUSIÓN A JSON
# ═══════════════════════════════════════════════════════════════════════════

confusion_matrix_data = {
    "matrix": cm_nn.tolist(),  # Convertir array de NumPy a lista
    "labels": ["Dropout", "Enrolled", "Graduate"],
    "model": "Neural Network"
}

with open('confusion_matrix.json', 'w') as f:
    json.dump(confusion_matrix_data, f, indent=4)

# También copiar a web/
with open('../web/confusion_matrix.json', 'w') as f:
    json.dump(confusion_matrix_data, f, indent=4)

print("✓ confusion_matrix.json generado")
print("\nMatriz de Confusión (JSON):")
print(json.dumps(confusion_matrix_data, indent=2))


# ═══════════════════════════════════════════════════════════════════════════
# 14. EXPORTAR MODELO A TENSORFLOWJS
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("PASO 11: EXPORTAR A TENSORFLOWJS")
print("="*80 + "\n")

print("🔄 Convirtiendo modelo a TensorFlow.js...")

# Usar tensorflowjs_converter desde la línea de comandos (mejor opción)
import subprocess
import sys

try:
    # Comando para convertir el modelo .keras a TensorFlow.js
    cmd = [
        sys.executable, '-m', 'tensorflowjs_converter',
        '--input_format', 'keras',
        'models/nn_model.keras',
        '../web/models/nn_model'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✓ Modelo convertido exitosamente a TensorFlow.js")
        print("✓ Archivos guardados en: ../web/models/nn_model/")
    else:
        print("⚠️  Error en conversión:")
        print(result.stderr)
        
except Exception as e:
    print(f"⚠️  No se pudo convertir con tensorflowjs_converter: {e}")
    print("Nota: Será necesario ejecutar en terminal:")
    print("tensorflowjs_converter --input_format keras models/nn_model.keras ../web/models/nn_model")


# ═══════════════════════════════════════════════════════════════════════════
# 15. CREAR ARCHIVO DE CONFIGURACIÓN PARA FRONTEND
# ═══════════════════════════════════════════════════════════════════════════

config_data = {
    "features": selected_features,
    "class_names": ["Dropout", "Enrolled", "Graduate"],
    "model_type": "neural_network",
    "input_shape": 5,
    "scaler_stats": {
        "mean": scaler.mean_.tolist(),
        "std": scaler.scale_.tolist()
    }
}

with open('../web/config.json', 'w') as f:
    json.dump(config_data, f, indent=4)

print("\n✓ config.json generado para frontend")


# ═══════════════════════════════════════════════════════════════════════════
# 16. RESUMEN FINAL
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("✅ ENTRENAMIENTO COMPLETADO")
print("="*80)

print(f"""
📁 ARCHIVOS GENERADOS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Backend (models/):
  ✓ logistic_model.pkl       - Modelo de Regresión Logística
  ✓ scaler.pkl               - Normalizador StandardScaler
  ✓ nn_model.keras           - Red Neuronal (formato Keras nativo)
  ✓ metrics.json             - Métricas de evaluación
  ✓ confusion_matrix.json    - Matriz de confusión

Frontend (../web/):
  ✓ metrics.json             - Métricas para mostrar
  ✓ confusion_matrix.json    - Matriz para visualizar
  ✓ config.json              - Configuración del modelo
  ✓ models/nn_model/         - Modelo para TensorFlow.js

📊 RESULTADOS DEL MODELO:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Red Neuronal (Modelo Recomendado):
  • Accuracy:  {accuracy_nn:.2%}
  • Precision: {precision_nn:.2%}
  • Recall:    {recall_nn:.2%}
  • F1-Score:  {f1_nn:.2%}

Próximos pasos:
1. Ejecutar en terminal:
   tensorflowjs_converter --input_format keras models/nn_model.keras ../web/models/nn_model
   
2. Abrir ../web/index.html en navegador
3. Hacer predicciones individuales o con archivos CSV

¡Listo para presentar el proyecto! 🎉
""")
