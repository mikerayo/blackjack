#!/bin/bash
# Script AUTOMÁTICO para configurar y entrenar en vast.ai
# Solo ejecuta: bash deploy_auto_vast.sh

set -e  # Detener si hay error

echo "╔════════════════════════════════════════════════════════════╗"
echo "║   🚀 ML BLACKJACK - AUTO DEPLOY VAST.AI                   ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# 1. Verificar GPU
echo "📊 Paso 1: Verificando GPU..."
nvidia-smi
echo ""

# 2. Crear directorio
echo "📁 Paso 2: Preparando directorio..."
mkdir -p /workspace/ML-BLACKJACK
cd /workspace/ML-BLACKJACK
echo "✓ Directorio listo: $(pwd)"
echo ""

# 3. Verificar si hay archivos
echo "📂 Paso 3: Verificando archivos..."
if [ -f "train_vast.py" ]; then
    echo "✓ Archivos encontrados"
else
    echo "⚠️  Los archivos NO están subidos aún."
    echo ""
    echo "📦 Tienes 2 opciones:"
    echo ""
    echo "OPCIÓN A - Subir por SCP desde tu Windows (PowerShell):"
    echo "  scp -P 22059 -r 'C:\Users\migue\Desktop\ML BLACKJACK' root@69.176.92.125:/workspace/"
    echo ""
    echo "OPCIÓN B - Subir por Jupyter:"
    echo "  1. En Jupyter, ve a /workspace/"
    echo "  2. Clic en 'Upload'"
    echo "  3. Sube estos archivos:"
    echo "     - src/ (toda la carpeta)"
    echo "     - requirements.txt"
    echo "     - train_vast.py"
    echo "     - train_massive.py"
    echo "     - *.md (todos los md)"
    echo ""
    echo " Luego ejecuta este script nuevamente."
    exit 1
fi
echo ""

# 4. Crear virtual environment
echo "🐍 Paso 4: Creando virtual environment..."
python3 -m venv venv
source venv/bin/activate
echo "✓ Virtual environment activado"
echo ""

# 5. Instalar dependencias
echo "📦 Paso 5: Instalando dependencias..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "✓ Dependencias instaladas"
echo ""

# 6. Verificar PyTorch + CUDA
echo "🔥 Paso 6: Verificando PyTorch + CUDA..."
python -c "
import torch
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('❌ ERROR: CUDA no disponible')
    exit(1)
"
echo ""

# 7. Verificar estructura del proyecto
echo "🔍 Paso 7: Verificando estructura..."
if [ -d "src" ] && [ -d "src/environment" ] && [ -d "src/agent" ]; then
    echo "✓ Estructura correcta"
else
    echo "❌ ERROR: Faltan carpetas de src/"
    echo "   Necesitas subir: src/"
    exit 1
fi
echo ""

# 8. Iniciar entrenamiento
echo "╔════════════════════════════════════════════════════════════╗"
echo "║   ✅ CONFIGURACIÓN COMPLETADA                              ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "🚀 Iniciando entrenamiento en 3 segundos..."
echo "   (Ctrl+C para detener)"
echo ""
sleep 3

python train_vast.py
