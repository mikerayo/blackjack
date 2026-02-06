# 🚀 GUIA RAPIDA - VAST.AI EN 3 PASOS

## PASO 1: Instalar Vast.ai CLI

```bash
pip install vast-ai
```

Luego inicia sesión:
```bash
vast login
```

---

## PASO 2: Buscar Mejores GPUs

### Opción A: Usar Script Automático
```bash
python vast_ai_search.py --search
```

### Opción B: Buscar Manualmente en Web
1. Ve a https://vast.ai/create
2. Filtra por:
   - **GPU**: RTX 3090, RTX 4090, o A100
   - **RAM**: Mínimo 16 GB
   - **Storage**: Mínimo 50 GB
   - **Internet**: Requerido
3. Ordena por precio más bajo
4. Elige una con buena reputación (> 0.95)

---

## PASO 3: Crear Instancia y Entrenar

### Opción A: Desde Web (RECOMENDADO - Más Fácil)

1. En https://vast.ai/create:
   - Elige la GPU (ej: RTX 3090 a $0.10/hora)
   - En "Image" escribe: `pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime`
   - En "Disk" escribe: `50` (GB)
   - Clic en "Rent!"

2. Una vez alquilada, verás detalles SSH:
   ```
   ssh -p 1234 root@xxx.xxx.xxx.xxx
   ```

3. Conéctate a la instancia:
   ```bash
   # Descarga tu clave SSH si es necesario
   # Luego conecta:
   ssh -p PUERTO root@IP
   ```

4. Sube los archivos:
   ```bash
   # En tu máquina local:
   scp -P PUERTO -r ML-BLACKJACK/ root@IP:/workspace/
   ```

5. En la instancia remota:
   ```bash
   cd /workspace/ML-BLACKJACK
   bash deploy_vast.sh
   ```

### Opción B: Desde CLI

```bash
# Buscar ofertas (toma el ID de la mejor)
python vast_ai_search.py --search

# Crear instancia (reemplaza ID)
vast create OFFER_ID --image pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime --disk 50

# Conectarte
vast ssh INSTANCE_ID

# Subir archivos (en otra terminal)
scp -P PUERTO -r ML-BLACKJACK/ root@IP:/workspace/

# En la instancia remota
cd /workspace/ML-BLACKJACK
bash deploy_vast.sh
```

---

## 📊 MONITOREAR PROGRESO

### Ver en Tiempo Real:
```bash
# En la instancia remota
tail -f models/logs/training.log
```

### O Vía Vast.ai Web:
1. Ve a https://vast.ai/console
2. Clic en tu instancia
3. Verás logs en tiempo real

---

## 💰 COSTOS ESTIMADOS

### 5 Millones de Episodios:

| GPU | Precio/hora | Tiempo | Costo Total |
|-----|-------------|--------|-------------|
| RTX 3090 | $0.10 | 2-3h | **$0.20 - $0.30** |
| RTX 4090 | $0.15 | 1-2h | **$0.15 - $0.30** |
| A100 | $0.50 | 1-1.5h | **$0.50 - $0.75** |

**Recomendación:** RTX 3090 o RTX 4090 por ~$0.20-0.30 total

---

## ⚠️ CONSEJOS IMPORTANTES

1. **Usa Docker Image Oficial de PyTorch:**
   ```
   pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
   ```

2. **Reserva suficiente disco:** 50 GB mínimo

3. **Monitoriza el progreso:** Asegúrate de que está entrenando

4. **Descarga checkpoints periódicamente:** Por si la instancia se cancela

5. **Detén la instancia cuando termines:** Para no seguir pagando

---

## 🎯 CONFIGURACIÓN OPTIMIZADA PARA GPU

El script `train_vast.py` ya está optimizado:

- **Batch Size:** 512 (vs 128 en CPU)
- **Buffer Size:** 1,000,000 (vs 500,000 en CPU)
- **Hidden Dims:** [1024, 512, 256] (vs [512, 256, 128] en CPU)
- **Checkpoint Interval:** 250,000 episodios

Esto aprovecha al máximo la GPU.

---

## 📥 DESCARGAR RESULTADOS

Cuando termine el entrenamiento:

```bash
# Desde tu máquina local
scp -P PUERTO -r root@IP:/workspace/ML-BLACKJACK/models ./
```

O comprimir primero:
```bash
# En la instancia remota
cd /workspace
tar -czf ML-BLACKJACK-results.tar.gz ML-BLACKJACK/models/

# En tu máquina local
scp -P PUERTO root@IP:/workspace/ML-BLACKJACK-results.tar.gz ./
```

---

## 🔧 SOLUCIÓN DE PROBLEMAS

### Error: "No module named 'torch'"

```bash
# En la instancia remota
pip install torch gymnasium numpy tensorboard
```

### Error: "CUDA out of memory"

Reduce el batch size en `train_vast.py`:
```python
'batch_size': 256,  # En lugar de 512
```

### Conexión SSH falla

1. Verifica que la instancia está corriendo
2. Revisa el puerto y IP correctos
3. Si usas firewall, permite el puerto

---

## ✅ CHECKLIST ANTES DE EMPEZAR

- [ ] vast.ai CLI instalado
- [ ] Cuenta creada y con saldo ($5-10)
- [ ] Script `train_vast.py` configurado
- [ ] Script `deploy_vast.sh` listo
- [ ] Entiendes cómo conectar por SSH
- [ ] Sabes cómo descargar los resultados

---

## 🎉 LISTO

Una vez que tengas tu instancia corriendo:

```bash
# 1. Conecta a la instancia
vast ssh INSTANCE_ID

# 2. Ve al directorio
cd /workspace/ML-BLACKJACK

# 3. Inicia entrenamiento
python train_vast.py

# 4. Espera 1-3 horas

# 5. Descarga resultados
# (En otra terminal en tu máquina)
scp -P PUERTO -r root@IP:/workspace/ML-BLACKJACK/models ./
```

**¡En 1-3 horas y por $0.20-0.30 tendrás tu modelo entrenado con 5M episodios!** 🚀
