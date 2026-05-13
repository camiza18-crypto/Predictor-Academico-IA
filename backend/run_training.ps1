# ═══════════════════════════════════════════════════════════════════════════
# RUN_TRAINING.PS1
# Script para ejecutar el entrenamiento del modelo con el entorno virtual
# ═══════════════════════════════════════════════════════════════════════════

Write-Host ""
Write-Host "╔═══════════════════════════════════════════════════════════════════════════╗" -ForegroundColor Magenta
Write-Host "║       EJECUCIÓN DEL ENTRENAMIENTO - PROYECTO FINAL IA                    ║" -ForegroundColor Magenta
Write-Host "╚═══════════════════════════════════════════════════════════════════════════╝" -ForegroundColor Magenta

Write-Host ""

# ─────────────────────────────────────────────────────────────────────────
# Activar entorno virtual
# ─────────────────────────────────────────────────────────────────────────

Write-Host "Activando entorno virtual..." -ForegroundColor Yellow

if (-Not (Test-Path "venv")) {
    Write-Host "❌ ERROR: No se encontró el directorio 'venv'" -ForegroundColor Red
    Write-Host "   Primero ejecuta: .\setup_environment.ps1" -ForegroundColor Yellow
    exit 1
}

# Activar el entorno
& "$PSScriptRoot\venv\Scripts\Activate.ps1"

Write-Host "✓ Entorno activado" -ForegroundColor Green

# ─────────────────────────────────────────────────────────────────────────
# Ejecutar el entrenamiento
# ─────────────────────────────────────────────────────────────────────────

Write-Host ""
Write-Host "Iniciando entrenamiento..." -ForegroundColor Cyan
Write-Host "════════════════════════════════════════════════════════════════════════════" -ForegroundColor Gray

python train.py

$trainingExitCode = $LASTEXITCODE

# ─────────────────────────────────────────────────────────────────────────
# Verificar resultado
# ─────────────────────────────────────────────────────────────────────────

Write-Host ""
Write-Host "════════════════════════════════════════════════════════════════════════════" -ForegroundColor Gray

if ($trainingExitCode -eq 0) {
    Write-Host ""
    Write-Host "╔═══════════════════════════════════════════════════════════════════════════╗" -ForegroundColor Green
    Write-Host "║                   ✓ ENTRENAMIENTO COMPLETADO EXITOSAMENTE                 ║" -ForegroundColor Green
    Write-Host "╚═══════════════════════════════════════════════════════════════════════════╝" -ForegroundColor Green
    
    Write-Host ""
    Write-Host "ARCHIVOS GENERADOS:" -ForegroundColor Cyan
    Write-Host ""
    
    # Verificar archivos
    if (Test-Path "models\nn_model.keras") {
        Write-Host "  ✓ models/nn_model.keras" -ForegroundColor Green
    }
    if (Test-Path "models\logistic_model.pkl") {
        Write-Host "  ✓ models/logistic_model.pkl" -ForegroundColor Green
    }
    if (Test-Path "models\scaler.pkl") {
        Write-Host "  ✓ models/scaler.pkl" -ForegroundColor Green
    }
    if (Test-Path "..\web\metrics.json") {
        Write-Host "  ✓ ../web/metrics.json" -ForegroundColor Green
    }
    if (Test-Path "..\web\confusion_matrix.json") {
        Write-Host "  ✓ ../web/confusion_matrix.json" -ForegroundColor Green
    }
    if (Test-Path "..\web\models\nn_model\model.json") {
        Write-Host "  ✓ ../web/models/nn_model/ (modelo convertido a TensorFlow.js)" -ForegroundColor Green
    } else {
        Write-Host "  ⚠ ../web/models/nn_model/ (requiere conversión manual)" -ForegroundColor Yellow
    }
    
    Write-Host ""
    Write-Host "PRÓXIMOS PASOS:" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "1. Si ../web/models/nn_model/ no existe, ejecuta:" -ForegroundColor White
    Write-Host "   " -NoNewline
    Write-Host "tensorflowjs_converter --input_format keras models/nn_model.keras ../web/models/nn_model" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "2. Prueba la aplicación web localmente:" -ForegroundColor White
    Write-Host "   " -NoNewline
    Write-Host "cd ../web && python -m http.server 8000" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "3. Abre en navegador:" -ForegroundColor White
    Write-Host "   " -NoNewline
    Write-Host "http://localhost:8000" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "════════════════════════════════════════════════════════════════════════════" -ForegroundColor Gray
    
} else {
    Write-Host ""
    Write-Host "╔═══════════════════════════════════════════════════════════════════════════╗" -ForegroundColor Red
    Write-Host "║                      ❌ ERROR DURANTE EL ENTRENAMIENTO                     ║" -ForegroundColor Red
    Write-Host "╚═══════════════════════════════════════════════════════════════════════════╝" -ForegroundColor Red
    
    Write-Host ""
    Write-Host "Verifica los mensajes de error arriba. Si necesitas ayuda:" -ForegroundColor Yellow
    Write-Host "  • Lee los logs cuidadosamente" -ForegroundColor White
    Write-Host "  • Verifica que data.csv exista en esta carpeta" -ForegroundColor White
    Write-Host "  • Asegúrate de que hay espacio en disco" -ForegroundColor White
    Write-Host "  • Comprueba la conexión a internet para descargar pesos del modelo" -ForegroundColor White
    
    exit $trainingExitCode
}

Write-Host ""
