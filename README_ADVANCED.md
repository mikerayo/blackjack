# ML Blackjack - Sistema Experto Avanzado + 5M Episodios

## 🚀 NUEVAS CARACTERÍSTICAS - VERSIÓN 2.0

### ✨ Lo Nuevo

1. **11 Estrategias Expertas Implementadas**
   - Basic Strategy
   - Hi-Lo con Index Plays (Illustrious 18)
   - KO Counting System
   - Ace-Five Count
   - Wizard of Odds Strategy
   - Thorp's Strategy
   - Wong Halves System
   - Zen Count System
   - Aggressive Strategy
   - Conservative Strategy
   - Adaptive Strategy

2. **7 Sistemas de Consenso**
   - Majority Voting (Votación por mayoría)
   - Weighted Voting (Votación ponderada por rendimiento)
   - Ranked Voting (Votación por ranking)
   - Borda Count (Sistema de puntos Borda)
   - Copeland Rule (Comparación por pares)
   - Meta-Learner (Meta-aprendizaje que elige la mejor estrategia)
   - **Hybrid Consensus** (Combinación de múltiples sistemas) ⭐ RECOMENDADO

3. **8 Sistemas de Apuestas Variables**
   - Flat Betting (Apuesta plana - más conservador)
   - Kelly Criterion (Óptimo matemático)
   - Hi-Lo Betting (Sistema clásico de conteo)
   - KO System Betting
   - Adaptive Betting (Se ajusta según rendimiento reciente)
   - Conservative Betting (Muy conservador)
   - Aggressive Betting (Máxima ganancia potencial)
   - Parlay Betting (Deja correr las ganancias)

4. **Entrenamiento Escalable para 5M+ Episodios**
   - Checkpointing automático cada 100K episodios
   - Capacidad de reanudar entrenamiento
   - Optimización de memoria
   - Integración con TensorBoard
   - Logging detallado de progreso

## 📁 Nueva Estructura del Proyecto

```
ML-BLACKJACK/
├── src/
│   ├── game/                    # Motor de blackjack (sin cambios)
│   ├── environment/             # Environment Gymnasium (sin cambios)
│   ├── agent/
│   │   ├── dqn.py               # Red neuronal
│   │   ├── replay_buffer.py     # Experience replay
│   │   ├── trainer.py           # Entrenador básico
│   │   └── scalable_trainer.py  # ⭐ NUEVO: Entrenador para 5M+ episodios
│   ├── strategies/              # ⭐ NUEVO MÓDULO
│   │   ├── expert_strategies.py # 11 estrategias expertas
│   │   ├── consensus_system.py  # 7 sistemas de consenso
│   │   ├── bankroll_management.py # 8 sistemas de apuestas
│   │   └── __init__.py
│   └── utils/                   # Métricas y visualización
├── train_massive.py             # ⭐ NUEVO: Script para entrenamiento masivo
├── evaluate_strategies.py       # ⭐ NUEVO: Evaluar todas las estrategias
└── README_ADVANCED.md           # Este archivo
```

## 🎯 Cómo Empezar Rápido

### 1. Evaluar Todas las Estrategias Expertas

```bash
python evaluate_strategies.py --episodes 10000 --type all
```

Esto comparará:
- Todas las 11 estrategias expertas
- Todos los 7 sistemas de consenso
- Todos los 8 sistemas de apuestas

**Resultado esperado:** Tabla comparativa con win rate, ROI, profit total, y Sharpe ratio.

### 2. Entrenamiento Masivo (5M Episodios)

#### Opción A: Configuración Recomendada
```bash
python train_massive.py \
    --episodes 5000000 \
    --use-consensus \
    --consensus-type hybrid \
    --use-variable-betting \
    --betting-system hilo \
    --initial-bankroll 100000 \
    --checkpoint-interval 100000 \
    --log-interval 10000
```

#### Opción B: Entrenamiento Más Rápido (1M episodios)
```bash
python train_massive.py \
    --episodes 1000000 \
    --use-consensus \
    --consensus-type hybrid \
    --betting-system kelly
```

#### Opción C: Entrenamiento Básico (sin expertos)
```bash
python train_massive.py \
    --episodes 1000000 \
    --no-use-consensus \
    --no-use-variable-betting
```

### 3. Reanudar Entrenamiento

```bash
python train_massive.py \
    --episodes 5000000 \
    --resume models/checkpoints/latest.pt
```

## 📊 Entender los Resultados

### Métricas Clave

1. **Win Rate**: Porcentaje de manos ganadas
   - 42-45% = Excelente
   - 40-42% = Bueno
   - <40% = Necesita mejorar

2. **ROI (Return on Investment)**: Porcentaje de retorno
   - >1% = Superando a la casa (objetivo)
   - 0-1% = Casi break-even
   - <0% = Perdiendo dinero

3. **Sharpe Ratio**: Retorno ajustado por riesgo
   - >1.0 = Excelente
   - 0.5-1.0 = Bueno
   - <0.5 = Demasiado volátil

4. **Profit Total**: Ganancia/pérdida total en dólares

## 🎓 Estrategias Expertas Explicadas

### 1. Basic Strategy
- **Qué es:** Estrategia matemática óptima sin contar cartas
- **Win Rate esperado:** ~42%
- **Mejor para:** Principiantes, base para comparación

### 2. Hi-Lo con Index Plays
- **Qué es:** Hi-Lo + Illustrious 18 (desviaciones óptimas)
- **Win Rate esperado:** ~43-44%
- **Mejor para:** Jugadores serios con conteo de cartas

### 3. KO System
- **Qué es:** Sistema de conteo desbalanceado (más fácil)
- **Win Rate esperado:** ~42-43%
- **Mejor para:** Quienes quieren un sistema más simple

### 4. Wong Halves
- **Qué es:** Sistema avanzado de conteo por fracciones
- **Win Rate esperado:** ~44-45%
- **Mejor para:** Contadores profesionales

### 5. Zen Count
- **Qué es:** Sistema balanceado de alta precisión
- **Win Rate esperado:** ~44-45%
- **Mejor para:** Máxima precisión en conteo

## 🤝 Sistemas de Consenso Explicados

### Hybrid Consensus (⭐ RECOMENDADO)

Combina múltiples sistemas de votación:
- 50% peso: Meta-learner (elige estrategia según contexto)
- 30% peso: Weighted voting (estrategias ponderadas)
- 20% peso: Majority voting (votación simple)

**Ventajas:**
- Adapta su elección según TC (true count), valor de mano, etc.
- Combina lo mejor de todos los sistemas
- Más robusto que cualquier sistema individual

### Meta-Learner Consensus

Elige automáticamente la mejor estrategia según la situación:
- TC >= 3: Hi-Lo con Index Plays (situación favorable)
- Mano <= 11: Wizard of Odds (agresivo con manos bajas)
- Mano >= 16: Conservative Strategy (proteger ganancias)
- Neutral: Basic Strategy

**Ventajas:** Máxima flexibilidad

## 💰 Sistemas de Apuestas Explicados

### Hi-Lo Betting (⭐ RECOMENDADO)

Sistema clásico de apuestas según true count:
- TC <= 0: 1 unidad (apuesta mínima)
- TC = 1: 2 unidades
- TC = 2: 4 unidades
- TC = 3: 6 unidades
- TC >= 4: 8+ unidades

**Ventajas:** Balance perfecto entre riesgo y recompensa.

### Kelly Criterion

Apuesta óptima matemática basada en edge:
```
Apuesta = (Edge / Odds) × Bankroll
```

**Ventajas:** Maximiza crecimiento a largo plazo.
**Riesgos:** Volátil si el edge es mal estimado.

### Parlay Betting

Deja correr las ganancias en rachas ganadoras:
- 1 victoria: 1× apuesta
- 2 victorias seguidas: 2× apuesta
- 3 victorias seguidas: 4× apuesta
- (Máximo 3 niveles)

**Ventajas:** Aprovecha rachas positivas.

## ⚙️ Configuración Avanzada

### Ajustar Agresividad del Entrenamiento

```bash
# Más exploración (aprende más lento pero mejor)
python train_massive.py --epsilon-decay 3000000

# Menos exploración (aprende más rápido)
python train_massive.py --epsilon-decay 1000000
```

### Ajustar Red Neuronal

```bash
# Red más grande (mejor pero más lento)
python train_massive.py --hidden-dims 1024,512,256

# Red más pequeña (más rápido)
python train_massive.py --hidden-dims 256,128
```

### Ajustar Apuestas

```bash
# Apuestas más agresivas (más riesgo, más recompensa)
python train_massive.py \
    --betting-system aggressive \
    --min-bet 25 \
    --max-bet 2000 \
    --initial-bankroll 250000

# Apuestas más conservadoras
python train_massive.py \
    --betting-system conservative \
    --min-bet 5 \
    --max-bet 100
```

## 📈 Esperar Resultados

### Con 5M Episodios y Sistema Híbrido

**Esperado:**
- Win Rate: 44-46%
- Ventaja sobre la casa: 1-2%
- ROI: +50-100% en bankroll inicial

**Factor Crítico:**
El true count (conteo de cartas) es ESPECIAL. Sin él, el máximo win rate es ~42%. Con true count + expert consensus + variable betting, puedes alcanzar 45%+.

## 🔍 Análisis de Resultados

### Después del Entrenamiento

1. **Verificar Checkpoints:**
```bash
ls -lh models/checkpoints/
```

2. **Revisar Métricas:**
```bash
cat models/metrics/metrics_ep_*.json
```

3. **Evaluar Modelo Final:**
```bash
python src/main.py --mode evaluate \
    --episodes 100000 \
    --model-path models/checkpoints/latest.pt
```

## ⚡ Optimizaciones de Rendimiento

### Para Entrenamiento Más Rápido

1. **Usar GPU (PyTorch la detecta automáticamente):**
```bash
# Verificar si PyTorch detecta GPU
python -c "import torch; print(torch.cuda.is_available())"
```

2. **Aumentar Batch Size:**
```bash
python train_massive.py --batch-size 256 --buffer-size 1000000
```

3. **Reducir Logging:**
```bash
python train_massive.py --log-interval 50000
```

## 🆘 Troubleshooting

### "Out of Memory"
```bash
# Reducir buffer size
python train_massive.py --buffer-size 200000

# O reducir batch size
python train_massive.py --batch-size 64
```

### "Entrenamiento muy lento"
```bash
# Reducir checkpointing
python train_massive.py --checkpoint-interval 500000

# Reducir logging
python train_massive.py --log-interval 50000
```

### Win Rate No Mejora
- Normal hasta 500K episodios
- Asegúrate de usar --use-consensus
- Asegúrate de usar --use-variable-betting
- Prueba diferentes --consensus-type

## 🎯 Metas Realistas

### Corto Plazo (100K episodios)
- Win Rate: 38-40%
- Todavía aprendiendo

### Medio Plazo (1M episodios)
- Win Rate: 42-44%
- Comienza a ser rentable

### Largo Plazo (5M+ episodios)
- Win Rate: 44-46%
- Ventaja consistente sobre la casa

## 📚 Referencias

- **Illustrious 18:** Don Schlesinger
- **Hi-Lo Count:** Stanford Wong
- **Kelly Criterion:** J. L. Kelly Jr.
- **Wizard of Odds:** Michael Shackleford
- **Beat the Dealer:** Edward O. Thorp

---

## 🚀 LISTO PARA ENTRENAR

**Comando recomendado:**
```bash
python train_massive.py \
    --episodes 5000000 \
    --use-consensus \
    --consensus-type hybrid \
    --use-variable-betting \
    --betting-system hilo \
    --initial-bankroll 100000 \
    --checkpoint-interval 100000 \
    --log-interval 10000
```

**Tiempo estimado:** 6-12 horas (depende del hardware)

¡Buena suerte! 🍀🎰
