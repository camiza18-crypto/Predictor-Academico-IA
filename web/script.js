/* ═══════════════════════════════════════════════════════════════════════════
   PROYECTO FINAL - SCRIPT PRINCIPAL
   Predicción de Estado Académico usando TensorFlow.js
   ═══════════════════════════════════════════════════════════════════════════ */

// ═══════════════════════════════════════════════════════════════════════════
// 1. VARIABLES GLOBALES
// ═══════════════════════════════════════════════════════════════════════════

let model = null;               // Modelo de red neuronal
let scaler = null;              // Objeto para normalizar datos
let confusionMatrixChart = null; // Referencia al gráfico
let csvResults = [];            // Almacenar resultados del CSV

// Nombres de las clases
const CLASS_NAMES = ['🚪 Dropout', '📚 Enrolled', '🎓 Graduate'];
const CLASS_COLORS = ['#ef4444', '#3b82f6', '#10b981'];

// Índices de las variables en orden
const FEATURE_INDICES = {
    'Curricular units 2nd sem (approved)': 0,
    'Curricular units 2nd sem (grade)': 1,
    'Tuition fees up to date': 2,
    'Scholarship holder': 3,
    'Admission grade': 4
};

console.log('✓ Variables globales inicializadas');


// ═══════════════════════════════════════════════════════════════════════════
// 2. FUNCIONES DE UTILIDAD
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Muestra un mensaje de log con timestamp
 * @param {string} message - Mensaje a mostrar
 */
function logMessage(message) {
    const timestamp = new Date().toLocaleTimeString();
    console.log(`[${timestamp}] ${message}`);
}

/**
 * Muestra un error en la UI
 * @param {string} message - Mensaje de error
 */
function showError(message) {
    const errorCard = document.getElementById('errorCard');
    const errorMessage = document.getElementById('errorMessage');
    errorMessage.textContent = message;
    errorCard.style.display = 'block';
    
    // Ocultar resultado si existe
    document.getElementById('resultCard').style.display = 'none';
    
    logMessage(`❌ Error: ${message}`);
}

/**
 * Oculta el mensaje de error
 */
function hideError() {
    document.getElementById('errorCard').style.display = 'none';
}

/**
 * Muestra un loader/spinner
 * @param {boolean} show - Mostrar u ocultar
 */
function showLoader(show = true) {
    const spinner = document.getElementById('formSpinner');
    const button = document.querySelector('#predictionForm .btn-primary');
    
    if (show) {
        spinner.style.display = 'inline-block';
        button.disabled = true;
    } else {
        spinner.style.display = 'none';
        button.disabled = false;
    }
}


// ═══════════════════════════════════════════════════════════════════════════
// 3. CARGAR MODELO Y CONFIGURACIÓN
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Carga el modelo de TensorFlow.js
 */
async function loadModel() {
    try {
        logMessage('Cargando modelo de red neuronal...');
        
        // La ruta debe apuntar al model.json del modelo exportado
        const modelPath = './models/nn_model/model.json';
        model = await tf.loadGraphModel(modelPath);
        
        logMessage('✓ Modelo cargado exitosamente');
        // Para GraphModel, los inputs se acceden de forma distinta
        logMessage('Modelo listo para predicciones');
        
    } catch (error) {
        console.error('Error cargando modelo:', error);
        showError(`Error al cargar el modelo: ${error.message}`);
    }
}

/**
 * Carga las métricas del modelo desde JSON
 */
async function loadMetrics() {
    try {
        logMessage('Cargando métricas del modelo...');
        
        const response = await fetch('./metrics.json');
        const metrics = await response.json();
        
        // Actualizar elementos en la UI
        document.getElementById('metricAccuracy').textContent = (metrics.accuracy * 100).toFixed(2) + '%';
        document.getElementById('metricPrecision').textContent = (metrics.precision * 100).toFixed(2) + '%';
        document.getElementById('metricRecall').textContent = (metrics.recall * 100).toFixed(2) + '%';
        document.getElementById('metricF1').textContent = (metrics.f1_score * 100).toFixed(2) + '%';
        
        // Actualizar barras de progreso
        document.getElementById('barAccuracy').style.width = (metrics.accuracy * 100) + '%';
        document.getElementById('barPrecision').style.width = (metrics.precision * 100) + '%';
        document.getElementById('barRecall').style.width = (metrics.recall * 100) + '%';
        document.getElementById('barF1').style.width = (metrics.f1_score * 100) + '%';
        
        logMessage('✓ Métricas cargadas: ' + 
                  `Accuracy=${(metrics.accuracy*100).toFixed(2)}%, ` +
                  `Precision=${(metrics.precision*100).toFixed(2)}%, ` +
                  `Recall=${(metrics.recall*100).toFixed(2)}%, ` +
                  `F1=${(metrics.f1_score*100).toFixed(2)}%`);
        
    } catch (error) {
        console.error('Error cargando métricas:', error);
        logMessage('⚠️  No se pudieron cargar las métricas');
    }
}

/**
 * Carga la matriz de confusión y dibuja el gráfico
 */
async function loadConfusionMatrix() {
    try {
        logMessage('Cargando matriz de confusión...');
        
        const response = await fetch('./confusion_matrix.json');
        const data = await response.json();
        
        drawConfusionMatrix(data.matrix);
        logMessage('✓ Matriz de confusión cargada');
        
    } catch (error) {
        console.error('Error cargando matriz de confusión:', error);
        logMessage('⚠️  No se pudo cargar la matriz de confusión');
    }
}

/**
 * Dibuja la matriz de confusión usando Chart.js
 * @param {Array} matrix - Matriz de confusión (3x3)
 */
function drawConfusionMatrix(matrix) {
    const ctx = document.getElementById('confusionMatrixChart').getContext('2d');
    
    // Destruir gráfico anterior si existe
    if (confusionMatrixChart) {
        confusionMatrixChart.destroy();
    }
    
    // Preparar datos para Chart.js
    const labels = ['Dropout', 'Enrolled', 'Graduate'];
    const maxValue = Math.max(...matrix.flat());
    
    // Crear datasets para cada clase
    const datasets = [];
    const colors = ['#ef4444', '#3b82f6', '#10b981'];
    
    for (let i = 0; i < 3; i++) {
        datasets.push({
            label: labels[i],
            data: matrix[i],
            backgroundColor: colors[i],
            borderColor: '#fff',
            borderWidth: 2
        });
    }
    
    confusionMatrixChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            animation: {
                duration: 1000,
                easing: 'easeInOutQuart'
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Matriz de Confusión — Valores Predichos vs Reales',
                    color: '#e2e8f0',
                    font: { size: 14, weight: 'bold', family: "'Segoe UI', sans-serif" },
                    padding: { bottom: 16 }
                },
                legend: {
                    display: true,
                    labels: {
                        color: '#e2e8f0',
                        font: { size: 13 },
                        padding: 20
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: maxValue,
                    ticks: { color: '#cbd5e1', font: { size: 12 } },
                    grid: { color: 'rgba(148, 163, 184, 0.15)' },
                    border: { color: 'rgba(148, 163, 184, 0.2)' }
                },
                x: {
                    ticks: { color: '#cbd5e1', font: { size: 12 } },
                    grid: { color: 'rgba(148, 163, 184, 0.15)' },
                    border: { color: 'rgba(148, 163, 184, 0.2)' }
                }
            }
        },
        plugins: [{
            id: 'customCanvasBg',
            beforeDraw(chart) {
                const ctx = chart.canvas.getContext('2d');
                ctx.save();
                ctx.globalCompositeOperation = 'destination-over';
                ctx.fillStyle = 'transparent';
                ctx.fillRect(0, 0, chart.width, chart.height);
                ctx.restore();
            }
        }]
    });
}

/**
 * Crea un objeto normalizado para las predicciones
 * Almacena la media y desviación estándar para normalizar datos
 */
function initializeScaler() {
    // Estos valores se obtienen del archivo config.json generado por Python
    // Por ahora usaremos valores por defecto
    scaler = {
        mean: [0, 0, 0, 0, 0],      // Se pueden cargar desde config.json
        std: [1, 1, 1, 1, 1]        // Se pueden cargar desde config.json
    };
    
    logMessage('✓ Scaler inicializado');
}

/**
 * Carga la configuración del modelo (características, estadísticas, etc)
 */
async function loadConfig() {
    try {
        const response = await fetch('./config.json');
        const config = await response.json();
        
        // Actualizar scaler con valores reales
        scaler = {
            mean: config.scaler_stats.mean,
            std: config.scaler_stats.std
        };
        
        logMessage('✓ Configuración cargada desde config.json');
        
    } catch (error) {
        logMessage('ℹ️  Usando configuración por defecto (config.json no encontrado)');
    }
}


// ═══════════════════════════════════════════════════════════════════════════
// 4. NORMALIZACIÓN DE DATOS (StandardScaler)
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Normaliza un array de características usando StandardScaler
 * Fórmula: z = (x - media) / desviación_estándar
 * 
 * @param {Array} features - Array de 5 características
 * @returns {Array} Features normalizados
 */
function normalizeFeatures(features) {
    if (!scaler) {
        console.warn('Scaler no inicializado, usando normalización default');
        return features;
    }
    
    // Aplicar la fórmula de normalización a cada característica
    const normalized = features.map((value, index) => {
        const mean = scaler.mean[index] || 0;
        const std = scaler.std[index] || 1;
        return (value - mean) / std;
    });
    
    logMessage(`Datos normalizados: [${normalized.map(x => x.toFixed(3)).join(', ')}]`);
    return normalized;
}


// ═══════════════════════════════════════════════════════════════════════════
// 5. HACER PREDICCIONES
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Realiza una predicción con el modelo
 * @param {Array} features - Array de 5 características normalizadas
 * @returns {Object} Objeto con predicción, probabilidades e interpretación
 */
async function makePrediction(features) {
    if (!model) {
        throw new Error('Modelo no cargado');
    }
    
    try {
        // Convertir array a Tensor de TensorFlow
        // Shape: [1, 5] (1 muestra, 5 características)
        const inputTensor = tf.tensor2d([features], [1, 5]);
        
        logMessage('Realizando predicción...');
        
        // Obtener predicción del modelo
        // Output: tensor de shape [1, 3] (probabilidades para 3 clases)
        const prediction = model.execute(inputTensor);
        
        // Convertir tensor a array JavaScript
        const probabilities = await prediction.data();
        const probs = Array.from(probabilities);
        
        // Obtener la clase predicha (índice con mayor probabilidad)
        const predictedClassIndex = probs.indexOf(Math.max(...probs));
        const predictedClass = CLASS_NAMES[predictedClassIndex];
        const confidence = probs[predictedClassIndex] * 100;
        
        // Limpiar tensores (importante para evitar memory leaks)
        inputTensor.dispose();
        prediction.dispose();
        
        logMessage(`✓ Predicción completada: ${predictedClass} (${confidence.toFixed(2)}%)`);
        
        return {
            classIndex: predictedClassIndex,
            className: predictedClass,
            confidence: confidence,
            probabilities: {
                dropout: probs[0] * 100,
                enrolled: probs[1] * 100,
                graduate: probs[2] * 100
            }
        };
        
    } catch (error) {
        logMessage(`❌ Error en predicción: ${error.message}`);
        throw error;
    }
}


// ═══════════════════════════════════════════════════════════════════════════
// 6. VALIDACIÓN DE FORMULARIO
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Valida que el formulario tenga valores válidos
 * @returns {boolean} True si es válido
 */
function validateForm() {
    const form = document.getElementById('predictionForm');
    
    if (!form.checkValidity()) {
        showError('Por favor completa todos los campos correctamente');
        return false;
    }
    
    // Validaciones adicionales
    const unitsApproved = parseFloat(document.getElementById('units_approved').value);
    const unitsGrade = parseFloat(document.getElementById('units_grade').value);
    const admissionGrade = parseFloat(document.getElementById('admission_grade').value);
    
    if (unitsApproved < 0 || unitsApproved > 30) {
        showError('Unidades aprobadas debe estar entre 0 y 30');
        return false;
    }
    
    if (unitsGrade < 0 || unitsGrade > 20) {
        showError('Calificación debe estar entre 0 y 20');
        return false;
    }
    
    if (admissionGrade < 0 || admissionGrade > 200) {
        showError('Calificación de admisión debe estar entre 0 y 200');
        return false;
    }
    
    return true;
}

/**
 * Obtiene los valores del formulario como array
 * @returns {Array} Array de 5 valores en el orden correcto
 */
function getFormValues() {
    return [
        parseFloat(document.getElementById('units_approved').value),
        parseFloat(document.getElementById('units_grade').value),
        parseFloat(document.getElementById('tuition_fees').value),
        parseFloat(document.getElementById('scholarship').value),
        parseFloat(document.getElementById('admission_grade').value)
    ];
}

/**
 * Muestra el resultado de la predicción en la UI
 * @param {Object} result - Objeto con resultado de predicción
 */
function displayPredictionResult(result) {
    hideError();
    
    // Mostrar clase predicha
    const predictedClassElem = document.getElementById('predictedClass');
    predictedClassElem.textContent = result.className;
    
    // Actualizar probabilidades
    document.getElementById('prob-dropout').textContent = result.probabilities.dropout.toFixed(2) + '%';
    document.getElementById('prob-enrolled').textContent = result.probabilities.enrolled.toFixed(2) + '%';
    document.getElementById('prob-graduate').textContent = result.probabilities.graduate.toFixed(2) + '%';
    
    // Actualizar barras de probabilidad
    document.getElementById('bar-dropout').style.width = result.probabilities.dropout + '%';
    document.getElementById('bar-enrolled').style.width = result.probabilities.enrolled + '%';
    document.getElementById('bar-graduate').style.width = result.probabilities.graduate + '%';
    
    // Generar interpretación
    const interpretation = generateInterpretation(result);
    document.getElementById('interpretationBox').innerHTML = interpretation;
    
    // Mostrar tarjeta de resultado
    document.getElementById('resultCard').style.display = 'block';
}

/**
 * Genera una interpretación amigable del resultado
 * @param {Object} result - Objeto con resultado
 * @returns {string} HTML con interpretación
 */
function generateInterpretation(result) {
    const classIndex = result.classIndex;
    
    let interpretation = '<strong>Análisis de la Predicción:</strong><br>';
    
    if (classIndex === 0) {
        interpretation += '⚠️ Este estudiante tiene <strong>alto riesgo de abandono</strong>. ';
        interpretation += 'Se recomienda:<br>• Contacto de orientación académica<br>• Evaluación de dificultades financieras<br>• Seguimiento personalizado';
    } else if (classIndex === 1) {
        interpretation += '📚 El estudiante está <strong>actualmente matriculado</strong>. ';
        interpretation += 'Se recomienda:<br>• Mantener seguimiento académico<br>• Fomentar participación en actividades<br>• Monitoreo de rendimiento';
    } else {
        interpretation += '🎓 ¡Excelente! Este estudiante tiene <strong>tendencia a graduarse</strong>. ';
        interpretation += 'Se recomienda:<br>• Apoyo para finalizar trámites<br>• Preparación para salida laboral<br>• Reconocimiento de logros';
    }
    
    return interpretation;
}


// ═══════════════════════════════════════════════════════════════════════════
// 7. PROCESAMIENTO DE ARCHIVOS CSV
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Procesa un archivo CSV y realiza predicciones para todas las filas
 * @param {File} file - Archivo CSV
 */
async function processCSVFile(file) {
    return new Promise((resolve, reject) => {
        Papa.parse(file, {
            header: true,
            dynamicTyping: true,
            skipEmptyLines: true,
            complete: async function(results) {
                try {
                    logMessage(`CSV cargado: ${results.data.length} registros`);
                    
                    // Mostrar barra de progreso
                    document.getElementById('csvProgress').style.display = 'block';
                    document.getElementById('csvResults').style.display = 'none';
                    csvResults = [];
                    
                    // Procesar cada fila
                    for (let i = 0; i < results.data.length; i++) {
                        const row = results.data[i];
                        
                        // Actualizar barra de progreso
                        const progress = ((i + 1) / results.data.length) * 100;
                        document.getElementById('progressBar').style.width = progress + '%';
                        document.getElementById('progressText').textContent = 
                            `Procesando: ${i + 1}/${results.data.length} registros`;
                        
                        try {
                            // Obtener valores del CSV (nombres de columnas pueden variar)
                            // Intentar múltiples nombres de columnas posibles
                            const values = [
                                row['Curricular units 2nd sem (approved)'] || row['approved'] || 0,
                                row['Curricular units 2nd sem (grade)'] || row['grade'] || 0,
                                row['Tuition fees up to date'] || row['tuition'] || 0,
                                row['Scholarship holder'] || row['scholarship'] || 0,
                                row['Admission grade'] || row['admission'] || 0
                            ];
                            
                            // Normalizar y predecir
                            const normalizedValues = normalizeFeatures(values);
                            const prediction = await makePrediction(normalizedValues);
                            
                            // Guardar resultado
                            csvResults.push({
                                values: values,
                                prediction: prediction
                            });
                            
                        } catch (error) {
                            console.warn(`Error procesando fila ${i + 1}:`, error);
                        }
                        
                        // Dar tiempo al navegador para actualizar UI
                        await new Promise(resolve => setTimeout(resolve, 10));
                    }
                    
                    // Mostrar resultados
                    displayCSVResults();
                    resolve(csvResults);
                    
                } catch (error) {
                    reject(error);
                }
            },
            error: function(error) {
                reject(error);
            }
        });
    });
}

/**
 * Muestra los resultados del procesamiento CSV en la UI
 */
function displayCSVResults() {
    // Contar predicciones
    const countDropout = csvResults.filter(r => r.prediction.classIndex === 0).length;
    const countEnrolled = csvResults.filter(r => r.prediction.classIndex === 1).length;
    const countGraduate = csvResults.filter(r => r.prediction.classIndex === 2).length;
    
    // Actualizar estadísticas
    document.getElementById('totalProcessed').textContent = csvResults.length;
    document.getElementById('countDropout').textContent = countDropout;
    document.getElementById('countEnrolled').textContent = countEnrolled;
    document.getElementById('countGraduate').textContent = countGraduate;
    
    // Llenar tabla de resultados
    const tbody = document.getElementById('resultsBody');
    tbody.innerHTML = '';
    
    csvResults.forEach((result, index) => {
        const row = document.createElement('tr');
        const [approved, grade, tuition, scholarship, admission] = result.values;
        const pred = result.prediction;
        
        row.innerHTML = `
            <td>${index + 1}</td>
            <td>${approved.toFixed(1)}</td>
            <td>${grade.toFixed(2)}</td>
            <td>${tuition === 0 ? '✓ Sí' : '✗ No'}</td>
            <td>${scholarship === 1 ? '✓ Sí' : '✗ No'}</td>
            <td>${admission.toFixed(1)}</td>
            <td>${pred.className}</td>
            <td>${pred.confidence.toFixed(2)}%</td>
        `;
        
        tbody.appendChild(row);
    });
    
    // Mostrar resultados y ocultar progreso
    document.getElementById('csvProgress').style.display = 'none';
    document.getElementById('csvResults').style.display = 'block';
    
    logMessage(`✓ Procesamiento completado: ${csvResults.length} registros analizados`);
}

/**
 * Descarga los resultados como archivo CSV
 */
function downloadCSVResults() {
    if (csvResults.length === 0) {
        alert('No hay resultados para descargar');
        return;
    }
    
    // Crear CSV
    let csv = 'Índice,Unidades Aprobadas,Calificación,Pago Arancel,Beca,Calif. Admisión,Predicción,Confianza\n';
    
    csvResults.forEach((result, index) => {
        const [approved, grade, tuition, scholarship, admission] = result.values;
        const pred = result.prediction;
        
        csv += `${index + 1},${approved},${grade},${tuition === 0 ? 'Sí' : 'No'},${scholarship === 1 ? 'Sí' : 'No'},${admission},"${pred.className}",${pred.confidence.toFixed(2)}\n`;
    });
    
    // Descargar
    const element = document.createElement('a');
    element.setAttribute('href', 'data:text/csv;charset=utf-8,' + encodeURIComponent(csv));
    element.setAttribute('download', 'resultados_prediccion.csv');
    element.style.display = 'none';
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
    
    logMessage('✓ Resultados descargados como CSV');
}


// ═══════════════════════════════════════════════════════════════════════════
// 8. EVENT LISTENERS
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Evento: Envío del formulario de predicción individual
 */
document.getElementById('predictionForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    if (!validateForm()) return;
    
    showLoader(true);
    
    try {
        const formValues = getFormValues();
        logMessage(`Valores del formulario: [${formValues.join(', ')}]`);
        
        const normalizedValues = normalizeFeatures(formValues);
        const result = await makePrediction(normalizedValues);
        
        displayPredictionResult(result);
        
    } catch (error) {
        showError(error.message);
    } finally {
        showLoader(false);
    }
});

/**
 * Evento: Carga de archivo CSV (drag and drop)
 */
const csvUploadArea = document.getElementById('csvUploadArea');
const csvFile = document.getElementById('csvFile');

csvUploadArea.addEventListener('click', () => csvFile.click());

csvUploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    csvUploadArea.style.borderColor = '#8b5cf6';
    csvUploadArea.style.background = 'rgba(139, 92, 246, 0.15)';
});

csvUploadArea.addEventListener('dragleave', () => {
    csvUploadArea.style.borderColor = '#6366f1';
    csvUploadArea.style.background = 'rgba(99, 102, 241, 0.05)';
});

csvUploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    csvUploadArea.style.borderColor = '#6366f1';
    csvUploadArea.style.background = 'rgba(99, 102, 241, 0.05)';
    
    const file = e.dataTransfer.files[0];
    if (file && file.type === 'text/csv') {
        processCSVFileHandler(file);
    } else {
        alert('Por favor carga un archivo CSV válido');
    }
});

csvFile.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        processCSVFileHandler(file);
    }
});

/**
 * Maneja el procesamiento de archivo CSV
 */
async function processCSVFileHandler(file) {
    try {
        logMessage(`Cargando archivo: ${file.name}`);
        await processCSVFile(file);
    } catch (error) {
        alert(`Error procesando CSV: ${error.message}`);
        console.error(error);
    }
}

/**
 * Evento: Botón para descargar resultados
 */
document.getElementById('downloadBtn').addEventListener('click', downloadCSVResults);


// ═══════════════════════════════════════════════════════════════════════════
// 9. INICIALIZACIÓN AL CARGAR LA PÁGINA
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Función principal de inicialización
 */
async function initialize() {
    console.log('═══════════════════════════════════════════════════════════════');
    console.log('  PREDICTOR DE ESTADO ACADÉMICO - INICIALIZANDO');
    console.log('═══════════════════════════════════════════════════════════════');
    
    try {
        // Inicializar scaler básico
        initializeScaler();
        
        // Cargar configuración
        await loadConfig();
        
        // Cargar modelo
        await loadModel();
        
        // Cargar métricas
        await loadMetrics();
        
        // Cargar matriz de confusión
        await loadConfusionMatrix();
        
        logMessage('═══════════════════════════════════════════════════════════════');
        logMessage('✅ APLICACIÓN LISTA PARA USAR');
        logMessage('═══════════════════════════════════════════════════════════════');
        
    } catch (error) {
        console.error('Error durante inicialización:', error);
        showError('Error inicializando la aplicación. Verifica la consola.');
    }
}

// Esperar a que cargue el DOM y luego inicializar
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initialize);
} else {
    initialize();
}

logMessage('Script cargado correctamente');
