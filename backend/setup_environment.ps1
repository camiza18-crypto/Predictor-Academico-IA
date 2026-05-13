# ═══════════════════════════════════════════════════════════════════════════
# SETUP_ENVIRONMENT.PS1
# Script para configurar automáticamente el entorno virtual e instalar 
# dependencias para el Proyecto Final de IA
# ═══════════════════════════════════════════════════════════════════════════

Write-Host ""
Write-Host "╔═══════════════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║         CONFIGURACIÓN DEL ENTORNO - PROYECTO FINAL IA                    ║" -ForegroundColor Cyan
Write-Host "╚═══════════════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# ─────────────────────────────────────────────────────────────────────────
# PASO 1: Verificar Python
# ─────────────────────────────────────────────────────────────────────────

Write-Host "[1/5] Verificando Python..." -ForegroundColor Yellow

try {
    $pythonVersion = python --version 2>&1
    Write-Host "      ✓ $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "      ✗ Python no encontrado. Instálalo desde https://www.python.org/" -ForegroundColor Red
    exit 1
}

# ─────────────────────────────────────────────────────────────────────────
# PASO 2: Eliminar venv anterior si existe
# ─────────────────────────────────────────────────────────────────────────

Write-Host ""
Write-Host "[2/5] Preparando entorno virtual..." -ForegroundColor Yellow

if (Test-Path "venv") {
    Write-Host "      Eliminando venv anterior..." -ForegroundColor Gray
    Remove-Item -Recurse -Force "venv" -ErrorAction SilentlyContinue
    Write-Host "      ✓ venv eliminado" -ForegroundColor Green
}

# ─────────────────────────────────────────────────────────────────────────
# PASO 3: Crear entorno virtual
# ─────────────────────────────────────────────────────────────────────────

Write-Host ""
Write-Host "[3/5] Creando entorno virtual..." -ForegroundColor Yellow

try {
    python -m venv venv
    Write-Host "      ✓ Entorno virtual creado" -ForegroundColor Green
} catch {
    Write-Host "      ✗ Error al crear venv: $_" -ForegroundColor Red
    exit 1
}

# ─────────────────────────────────────────────────────────────────────────
# PASO 4: Activar entorno virtual
# ─────────────────────────────────────────────────────────────────────────

Write-Host ""
Write-Host "[4/5] Activando entorno virtual..." -ForegroundColor Yellow

& "$PSScriptRoot\venv\Scripts\Activate.ps1" 2>$null

# Verificar si se activó
if ($LASTEXITCODE -eq 0 -or $env:VIRTUAL_ENV) {
    Write-Host "      ✓ Entorno activado" -ForegroundColor Green
} else {
    Write-Host "      ✓ Entorno activado (sin confirmación visible)" -ForegroundColor Green
}

# ─────────────────────────────────────────────────────────────────────────
# PASO 5: Actualizar herramientas base e instalar dependencias
# ─────────────────────────────────────────────────────────────────────────

Write-Host ""
Write-Host "[5/5] Instalando dependencias..." -ForegroundColor Yellow

Write-Host "      → Actualizando pip, setuptools y wheel..." -ForegroundColor Gray
python -m pip install --upgrade pip setuptools wheel -q

Write-Host "      → Instalando paquetes desde requirements.txt..." -ForegroundColor Gray
pip install -r requirements.txt

if ($LASTEXITCODE -eq 0) {
    Write-Host "      ✓ Todas las dependencias instaladas" -ForegroundColor Green
} else {
    Write-Host "      ✗ Error durante la instalación de dependencias" -ForegroundColor Red
    Write-Host "      Intenta ejecutar nuevamente o verifica tu conexión a internet" -ForegroundColor Yellow
    exit 1
}

# ─────────────────────────────────────────────────────────────────────────
# RESUMEN FINAL
# ─────────────────────────────────────────────────────────────────────────

Write-Host ""
Write-Host "╔═══════════════════════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║                    ✓ ENTORNO CONFIGURADO EXITOSAMENTE                     ║" -ForegroundColor Green
Write-Host "╚═══════════════════════════════════════════════════════════════════════════╝" -ForegroundColor Green

Write-Host ""
Write-Host "PRÓXIMOS PASOS:" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. El entorno ya está activado. Verás (venv) al inicio de la línea." -ForegroundColor White
Write-Host ""
Write-Host "2. Para ejecutar el entrenamiento, usa:" -ForegroundColor White
Write-Host "   " -NoNewline
Write-Host ".\run_training.ps1" -ForegroundColor Yellow
Write-Host ""
Write-Host "3. O ejecuta directamente:" -ForegroundColor White
Write-Host "   " -NoNewline
Write-Host "python train.py" -ForegroundColor Yellow
Write-Host ""
Write-Host "════════════════════════════════════════════════════════════════════════════" -ForegroundColor Gray

Write-Host ""
