"""Create deployment package for server + Vast.ai"""

import zipfile
import os
from pathlib import Path

print("="*70)
print("CREANDO PAQUETE DE DESPLIEGUE")
print("="*70)

# Crear ZIP
print("\n[1/3] Creando ZIP...")
with zipfile.ZipFile('blackjack_server.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
    count = 0

    for root, dirs, files in os.walk('.'):
        # Filtrar directorios
        dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git', '.pytest_cache', 'venv', 'env', '.idea']]

        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, '.')

            # Incluir archivos Python y checkpoints
            if (file.endswith('.py') or
                file.endswith('.txt') or
                file.endswith('.md') or
                'checkpoint_ep500000.pt' in file_path):
                zf.write(file_path, rel_path)
                count += 1
                if count <= 10:
                    print(f"  + {rel_path}")

print(f"[OK] ZIP creado: blackjack_server.zip ({count} archivos)")
print(f"     Size: {Path('blackjack_server.zip').stat().st_size / 1024:.1f} KB")

# Crear instrucciones
print("\n[2/3] Creando instrucciones...")
instructions = """
# INSTRUCCIONES PARA DESPLEGAR EN HETZNER + VAST.AI

## PASO 1: Subir a tu servidor Hetzner

Desde tu máquina local (PowerShell):

```powershell
# Reemplaza con tu IP y usuario
scp C:\\Users\\migue\\Desktop\\ML-BLACKJACK\\blackjack_server.zip tu-usuario@tu-ip:/root/
```

## PASO 2: Conectar a tu servidor

```bash
ssh tu-usuario@tu-ip
```

## PASO 3: Preparar en servidor

```bash
# Descomprimir
cd /root
unzip blackjack_server.zip
cd blackjack

# Instalar dependencias
pip3 install torch numpy gymnasium tqdm -q

# Verificar instalación
python3 -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

## PASO 4: Crear cuenta Vast.ai

1. Ve a https://console.vast.ai
2. Crea cuenta y deposita fondos ($5-10 USD)
3. Click en "Create" → "Rent GPU"
4. Configura:
   - Image: PyTorch
   - GPU: RTX 4090 (o lo que esté disponible)
   - Disk: 20 GB
   - Max bid: $0.50/hr

## PASO 5: En Vast.ai (una vez conectado)

```bash
# Opción A: Subir archivo ZIP desde tu PC
# (Usa el botón upload en Jupyter)

# Opción B: Descargar desde tu Hetzner
# (En Vast.ai terminal)
wget http://tu-ip-servidor/blackjack_server.zip

# Descomprimir
unzip blackjack_server.zip
cd blackjack

# Instalar dependencias
pip install torch numpy gymnasium tqdm -q

# Verificar GPU
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"

# Iniciar entrenamiento (10M episodios)
nohup python train_vast.py > training.log 2>&1 &

# Monitorear
tail -f training.log
```

## PASO 6: Monitoreo desde tu Hetzner

```bash
# Conectarte por SSH a Vast.ai (desde Hetzner)
ssh root@vast-ai-ip

# Ver logs
tail -f /root/blackjack/training.log

# Ver progreso
ls -lh /root/blackjack/models/
```

## PASO 7: Descargar resultados

Cuando termine (desde Vast.ai):
```bash
cd /root/blackjack
tar -czf results_10M.tar.gz models/
exit
```

Descargar desde Vast.ai a tu Hetzner:
```bash
# En Hetzner
scp root@vast-ai-ip:/root/blackjack/results_10M.tar.gz /root/
```

## NOTA: El checkpoint de 500K se incluye automáticamente
El entrenamiento se reanudará desde el episodio 500,000 hacia 10,000,000
"""

with open('SERVER_DEPLOY.txt', 'w', encoding='utf-8') as f:
    f.write(instructions)

print("[OK] Instrucciones guardadas en: SERVER_DEPLOY.txt")

# Crear script de subida rápida
print("\n[3/3] Creando scripts de ayuda...")
with open('upload_to_server.ps1', 'w', encoding='utf-8') as f:
    f.write('# Script para subir archivo a servidor\n')
    f.write('# Modifica las variables abajo:\n\n')
    f.write('$serverIP = "TU-SERVER-IP"\n')
    f.write('$serverUser = "root"\n')
    f.write('$localPath = "C:\\Users\\migue\\Desktop\\ML-BLACKJACK\\blackjack_server.zip"\n\n')
    f.write('# Subir archivo\n')
    f.write('scp $localPath "$($serverUser)@$($serverIP):/root/"\n\n')
    f.write('# Conectar\n')
    f.write('ssh "$($serverUser)@$($serverIP)"\n')

print("[OK] Scripts creados")
print("\n" + "="*70)
print("RESUMEN:")
print("="*70)
print("1. Archivo creado: blackjack_server.zip")
print("2. Instrucciones: SERVER_DEPLOY.txt")
print("3. Script PowerShell: upload_to_server.ps1")
print("\nPasos:")
print("1. Modifica upload_to_server.ps1 con tu IP")
print("2. Ejecuta: .\\upload_to_server.ps1")
print("3. Sigue instrucciones en SERVER_DEPLOY.txt")
print("="*70)
