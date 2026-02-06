# 🎰 GUÍA: OBTENER DATOS REALES DE CASINOS ONLINE

## ⚠️ ADVERTENCIA IMPORTANTE

### ❌ NO HACER:
- **Web scraping** de casinos con dinero real → Ilegal + baneo
- **Bot automatizado** en casinos reales → Violación TOS
- **Obtener datos** sin permiso → Imposible (encriptados/privados)

### ✅ SÍ HACER:
- **Jugar en DEMO** y registrar manualmente
- **Crear bots** solo para DEMO/PLAY MONEY
- **Usar APIs** si el casino las ofrece públicas
- **Streaming data** mientras juegas tú mismo

---

## 🎲 ESTRATEGIAS PARA OBTENER DATOS REALES

### Opción 1: Manual en Casino DEMO (MÁS FÁCIL)

#### Casinos Fiat Recomendados:
1. **BetOnline** - https://www.betonline.ag/casino
   - Tiene versión DEMO gratis
   - Blackjack de 6 barajas
   - Reglas estándar

2. **Ignition Casino** - https://www.ignitioncasino.eu/
   - Requiere registro
   - Tiene modo "Practice"
   - Buena interfaz

3. **Bovada** - https://www.bovada.lv/
   - Versión "Play Money"
   - Muy popular en USA

#### Casinos Crypto Recomendados:
1. **Stake** - https://stake.com/casino/games/blackjack
   - Demo con fictitious coins
   - Muy fácil de usar
   - No requiere registro para demo

2. **BitStarz** - https://bitstarz.com/
   - Modo "Play Money"
   - Blackjack Evolution Gaming

3. **mBit Casino** - https://www.mbitcasino.com/
   - Versión demo disponible
   - Soporta múltiples cryptos

---

### Opción 2: Bot Automatizado (Requiere Selenium)

#### Paso 1: Instalar dependencias
```bash
pip install selenium webdriver-manager
```

#### Paso 2: Descargar ChromeDriver
- https://chromedriver.chromium.org/
- O usar `webdriver-manager` automático

#### Paso 3: Usar el bot
```bash
python casino_data_bot.py automated
```

⚠️ **ADVERTENCIA**: El modo automatizado requiere adaptación a CADA casino. No es plug-and-play.

---

### Opción 3: APIs Públicas (Si existen)

#### Algunos casinos tienen APIs:
- **Stake**: Tiene API pública (documentación en su sitio)
- **BetOnline**: API para afiliados (limitada)
- **Evolution Gaming**: Proveedores de software tienen APIs

Ejemplo con Stake API:
```python
import requests

# Obtener datos de mesa (si disponible)
response = requests.get('https://api.stake.com/casino/tables/blackjack')
data = response.json()
```

---

## 🤖 CÓMO USAR EL BOT

### Modo Interactivo (RECOMENDADO)

```bash
python casino_data_bot.py interactive
```

**Pasos:**
1. Abre el casino DEMO en tu navegador
2. Ejecuta el bot
3. Juega normalmente
4. Por cada mano, ingresa: `tu_total, carta_dealer, accion, resultado`
   - Ejemplo: `18,7,stand,win`
5. Cuando termines, presiona Enter sin datos

**Ejemplo de sesión:**
```
Mano #1
Tu total, carta dealer, acción, resultado: 18,7,stand,win
Apuesta (Enter para $10): 10
✅ Mano 1: 18 vs 7 → stand → win

Mano #2
Tu total, carta dealer, acción, resultado: 12,4,hit,loss
Apuesta (Enter para $10): 10
✅ Mano 2: 12 vs 4 → hit → loss

Mano #3
Tu total, carta dealer, acción, resultado: [Enter]

📊 RESUMEN DE LA SESIÓN
Total manos: 2
Victorias: 1 (50.0%)
Derrotas: 1 (50.0%)
Profit: $0.00
Guardado en: casino_data/fiat_20250206_143022.json
```

---

## 📊 FORMATO DE DATOS

Cada sesión guarda un JSON con esta estructura:

```json
{
  "session_id": "20250206_143022",
  "casino_name": "Stake",
  "casino_type": "crypto",
  "is_demo": true,
  "start_time": "2025-02-06T14:30:22",
  "end_time": "2025-02-06T15:45:10",
  "statistics": {
    "total_hands": 150,
    "wins": 68,
    "losses": 75,
    "pushes": 7,
    "blackjacks": 5,
    "win_rate": 0.453,
    "total_profit": -70.00
  },
  "hands": [
    {
      "timestamp": "2025-02-06T14:30:45",
      "player_total": 18,
      "dealer_card": 7,
      "action": "stand",
      "result": "win",
      "bet_amount": 10,
      "cards_player": ["K", "8"],
      "cards_dealer": ["7"],
      "split": false,
      "doubled": false,
      "blackjack": false
    },
    ...
  ]
}
```

---

## 🎯 OBJETIVOS

### Mínimo: 1,000 manos
- **Tiempo**: 2-4 horas
- **Mejora esperada**: +0.5-1% win rate
- **¿Vale la pena?**: Sí

### Óptimo: 5,000 manos
- **Tiempo**: 10-15 horas
- **Mejora esperada**: +1-2% win rate
- **¿Vale la pena?**: Probablemente

### Máximo: 10,000+ manos
- **Tiempo**: 20-30 horas
- **Mejora esperada**: +2-3% win rate
- **¿Vale la pena?**: Solo si te gusta jugar

---

## 💡 CONSEJOS

### ✅ HACER:
1. **Empezar con DEMO** - Sin riesgo
2. **Probar varios casinos** - Diferentes reglas
3. **Registrar TODO** - Incluso manos aburridas
4. **Tomar descansos** - Jugar cansado = malas decisiones
5. **Verificar datos** - Revisar el JSON guardado

### ❌ NO HACER:
1. **Jugar con dinero real solo para datos** - Pérdida de tiempo
2. **Usar bots en casinos reales** - Te banean
3. **Ignorar TOS del casino** - Puede tener consecuencias legales
4. **Exagerar datos** - Mejor 1000 reales que 10000 inventados

---

## 🔄 FLUJO COMPLETO

### Fase 1: Entrenar modelo base
```bash
# En vast.ai (ya lo estás haciendo)
python train_vast.py
```

### Fase 2: Recopilar datos (simultáneo)
```bash
# Mientras tanto, en tu PC:
python casino_data_bot.py interactive
```

### Fase 3: Fine-tuning
```bash
# Cuando terminen ambos:
python fine_tune_real_data.py
```

### Fase 4: Evaluar
```bash
python evaluate_strategies.py --model models/checkpoints/fine_tuned_real.pt
```

---

## 📋 CHECKLIST

- [ ] Elegir 2-3 casinos DEMO
- [ ] Crear cuenta en cada uno (si necesario)
- [ ] Probar la interfaz de cada casino
- [ ] Decidir cuál te gusta más
- [ ] Configurar el bot
- [ ] Jugar 1000 manos mínimo
- [ ] Revisar los datos guardados
- [ ] Fine-tunear el modelo
- [ ] Evaluar mejora

---

## 🆘 PROBLEMAS COMUNES

### Problema: "El casino no carga en DEMO"
**Solución**: Algunos casinos requieren registro. Crea cuenta falsa con datos genéricos.

### Problema: "No detecta las cartas"
**Solución**: Usa modo interactivo en lugar de automatizado.

### Problema: "Me aburro jugando"
**Solución**: Haz 100-200 manos por día, no todo de una vez.

### Problema: "Los datos no son consistentes"
**Solución**: Siempre juega en la misma mesa con las mismas reglas.

---

## 🎰 LISTA DE CASINOS POR TIPO

### 🇺🇸 USA (Fiat):
- Bovada
- Ignition Casino
- BetOnline
- Slots.lv

### 🌍 Internacional (Fiat):
- 888 Casino
- Bet365
- William Hill
- LeoVegas

### 💎 Crypto:
- Stake
- BitStarz
- mBit Casino
- 7Bit Casino
- FortuneJack

### 🎮 Evolution Gaming (proveedor):
- Muchos casinos usan su software
- Mismo juego en diferentes casinos
- Datos consistentes

---

## 📖 REFERENCIAS

- **Stake API**: https://stake.com/api (si está disponible)
- **Evolution Gaming**: https://www.evolution.com/ (contactar para API)
- **Documentación Selenium**: https://selenium-python.readthedocs.io/

---

**¡Recuerda: Siempre usar versiones DEMO para recopilar datos!** 🎰✅
