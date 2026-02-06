# 🎯 QUICK REFERENCE - PRÓXIMA SESIÓN

## COMANDOS PARA ARRANCAR INMEDIATAMENTE

### 1️⃣ VERIFICAR BASELINE (1 minuto)
```bash
cd "C:\Users\migue\Desktop\ML BLACKJACK"
python test_basic_only.py
```
**Esperado:** Win Rate 42.3%
**Si no funciona:** Revisar environment

---

### 2️⃣ DIAGNOSTICAR BUG EN EXPERTOS (5 minutos)
```bash
python test_expert_actions.py
```
**Esperado:** Ver acciones que toman los expertos
**Si ves:** SPLIT con 3 cartas, SURRENDER con 18 = BUG confirmado

---

### 3️⃣ ENTRENAMIENTO SIMPLE DQN (30 minutos)
```bash
python src/main.py --mode train \
    --episodes 10000 \
    --epsilon-decay 10000
```
**Meta:** Verificar que DQN aprende
**Esperado:** Win rate subiendo de 30% → 40%+

---

### 4️⃣ LEER DOCUMENTACIÓN COMPLETA
```bash
NEXT_STEPS.md  # ← ESTE DOCUMENTO
```

---

## 📋 PRIORIDADES EXACTAS

### 🔥 CRÍTICO (HACER PRIMERO)
```
□ Arreglar expert strategies (src/strategies/expert_strategies.py)
  Revisar líneas 80-150 de cada expert
  Validar can_split(), can_double(), etc.
  Testear vs Basic Strategy del environment
```

### ⚡ IMPORTANTE
```
□ Corregir epsilon decay en scalable_trainer.py
  Cambiar epsilon_decay_steps de 2,000,000 a 50,000
□ Testear curriculum trainer
```

### 🚀 PROGRESIVO
```
□ Entrenar modelo DQN simple 100K episodios
□ Evaluar resultados
□ Si >40% win rate, escalar a vast.ai
```

---

## 🎯 OBJETIVO DE LA SESIÓN

**Meta:** Terminar con expert strategies funcionando

**Criterio de éxito:**
```bash
python evaluate_experts_only.py --episodes 1000
# Esperado: Win Rate 43-45%
```

---

## 📊 DÓNDE ESTAMOS AHORA MISMO

| Componente | Status | Win Rate |
|------------|--------|----------|
| Environment ✅ | Funciona | 42.3% (Basic Strategy) |
| DQN Agent ⚠️ | Parcialmente funcional | 15% (epsilon alto) |
| Expert Strategies ❌ | ROTO | 7-9% (debería 43%) |
| Betting Systems ✅ | Funciona | N/A |
| Consensus ❌ | Roto | Hereda bug de expertos |

---

## 🔑 CLAVE

**El problema NO es el architecture, es la implementación de los expertos.**

Fix los expertos → Todo el sistema funcionará → Podemos escalar a vast.ai

---

## 💡 TRUCO PARA DEBUGGEAR EXPERTOS

Comparar con `env.get_basic_strategy_action()`:

```python
from environment.blackjack_env import BlackjackEnv
from strategies.expert_strategies import BasicStrategy

env = BlackjackEnv()
state, _ = env.reset()
game_state = env.game.get_state()

# Environment basic strategy
env_action = env.get_basic_strategy_action()

# My basic strategy
my_bs = BasicStrategy()
valid_actions = [0,1,2,3,4,5]
my_action = my_bs.get_action(game_state, valid_actions)

if hasattr(my_action, 'value'):
    my_action = int(my_action.value)

print(f"Environment: {env_action}")
print(f"My Strategy: {my_action}")
print(f"Match: {env_action == my_action}")

# Si no match → está mal implementado
```

---

**¡LISTO PARA LA PRÓXIMA SESIÓN! 🚀**
