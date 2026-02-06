# 🎰 DATOS Y RECURSOS DE CASINOS PARA REENTRENAMIENTO

## 📊 RESUMEN EJECUTIVO

Los datos reales de casinos son **extremadamente raros** y casi inexistentes públicamente. Los casinos protegen esta información de forma proprietaria. Sin embargo, he recopilado los mejores recursos disponibles:

---

## ✅ DATASETS PÚBLICOS DISPONIBLES

### 1. Kaggle - 900,000 Hands of Blackjack Results
- **URL**: https://www.kaggle.com/datasets/mojocolors/900000-hands-of-blackjack-results
- **Tamaño**: 900,000 manos
- **Tipo**: Simulado (no datos reales de casino)
- **Descargar**:
  ```bash
  pip install kaggle
  kaggle datasets download -d mojocolors/900000-hands-of-blackjack-results
  ```

### 2. Kaggle - 50 Million Blackjack Hands
- **URL**: https://www.kaggle.com/datasets/dennisho/blackjack-hands
- **Tamaño**: 50 millones de manos
- **Tipo**: Simulado realista
- **Descargar**:
  ```bash
  kaggle datasets download -d dennisho/blackjack-hands
  ```

### 3. Kaggle - Simulated Blackjack Data
- **URL**: https://www.kaggle.com/datasets/flynn28/simulated-blackjack-data
- **Tipo**: Simulado con varios mazos

### 4. GitHub - kaseymallette/blackjack
- **URL**: https://github.com/kaseymallette/blackjack
- **Archivo**: `blackjack/data/hand_data.csv`
- **Tipo**: Simulado (para análisis de estrategia básica)

---

## 📈 ESTADÍSTICAS OFICIALES DE CASINOS

### Nevada Gaming Control Board
- **URL**: https://www.gaming.nv.gov/about-us/statistics-and-publications/
- **Contenido**: Datos oficiales de ingresos de casinos en Nevada
- **Datos históricos**: Disponibles desde 2004
- **Reportes**:
  - **GRI** (Gaming Revenue Information): Mensuales
  - **QSI** (Quarterly Statistical Information): Trimestrales
  - **Anuales**: Informes financieros completos

**Estadísticas clave**:
- Win rate de casinos en blackjack: ~14-15%
- House edge estándar: 0.5% - 2%
- Estrategia perfecta: 0.13% - 0.5% house edge

### UNLV Center for Gaming Research
- **URL**: https://gaming.library.unlv.edu/all-reports.html
- **Contenido**: Gráficos y análisis de blackjack
- **Incluye**: Datos de "Nevada Table Mix"

---

## 📚 PAPERS ACADÉMICOS CON DATOS

### 1. Fear and Loathing in Las Vegas (DATOS REALES)
- **URL**: https://www.cambridge.org/core/journals/judgment-and-decision-making/article/fear-and-loathing-in-las-vegas-evidence-from-blackjack-tables/A59038264979AC8C6A6B7D2436F958DA
- **Año**: 2009
- **Importancia**: **ÚNICO paper con datos reales de mesas de Las Vegas**
- **Citas**: 19 veces
- **Nota**: Los datos no están públicamente disponibles (propietarios)

### 2. Blackjack: Beating the Odds Using RL
- **URL**: https://tesi.luiss.it/40403/1/275041_AGUDIO_TOMMASO.pdf
- **Año**: Noviembre 2024
- **Enfoque**: Reinforcement learning en blackjack
- **Tipo**: Thesis, incluye simulaciones

### 3. An Analytic Derivation of Blackjack Win Rates
- **URL**: https://www.jstor.org/stable/170849
- **Año**: 1985
- **Contenido**: Teoría matemática de conteo de cartas
- **Análisis**: Escenarios con n cartas jugadas

### 4. Probability Models for Blackjack
- **URL**: https://www.sciencedirect.com/science/article/pii/S0898122109006543
- **Año**: 2010
- **Contenido**: Modelos probabilísticos y distribuciones

### 5. Expected Value of Advantage Players
- **URL**: https://digitalcommons.usu.edu/cgi/viewcontent.cgi?article=1528&context=gradreports
- **Año**: 2014
- **Contenido**: Efecto de la penetración del shoe

---

## 🗂️ REPOSITORIOS GITHUB

### Simuladores y Datos:
1. **kaseymallette/blackjack** - https://github.com/kaseymallette/blackjack
   - Archivo: `hand_data.csv`

2. **AntoniovanDijck/BlackJackRL** - https://github.com/AntoniovanDijck/BlackJackRL
   - Deep RL para blackjack

3. **jgayda/blackjack-simulator** - https://github.com/jgayda/blackjack-simulator
   - Análisis de estrategias

4. **GregSommerville/machine-learning-blackjack-solution** - https://github.com/GregSommerville/machine-learning-blackjack-solution
   - Algoritmos genéticos

---

## 🎯 ESTRATEGIA PARA OBTENER DATOS REALES

### Opción 1: Jugar y Registrar (RECOMENDADO)
```bash
# Usar el script incluido
python collect_real_data.py interactive

# O programáticamente
python collect_real_data.py
```

**Registra**:
- Tus manos jugadas en casinos online
- Estado, acción, resultado
- Reglas de la mesa
- Variaciones de apuestas

### Opción 2: Web Scraping de Casinos Online
⚠️ **Legalmente arriesgado** - Verifica TOS de cada casino

### Opción 3: Comprar Datos (No disponible)
❌ Los casinos NO venden datos de manos

---

## 📋 PLAN DE ACCIÓN

### FASE 1: Mientras tu modelo entrena (AHORA)
```bash
# 1. Instalar Kaggle API
pip install kaggle

# 2. Configurar API key
# Ve a https://www.kaggle.com/settings → API → Create New Token
# Descarga kaggle.json y ponlo en ~/.kaggle/

# 3. Descargar datasets públicos
python download_public_datasets.py
```

### FASE 2: Cuando termine el modelo de 10M
```bash
# 1. Evaluar modelo base
python evaluate_strategies.py --model models/checkpoint_10M.pt

# 2. Jugar 1000-10000 manos en casino online/real
python collect_real_data.py interactive

# 3. Re-entrenar con datos reales
python fine_tune_with_real_data.py --real-data casino_data/
```

### FASE 3: Fine-tuning con Datos Reales
```python
# Cargar modelo base
base_model = torch.load('models/checkpoint_10M.pt')

# Cargar tus datos reales
real_data = load_casino_sessions('casino_data/')

# Fine-tuning (solo 100K-500K episodios!)
fine_tuned_model = fine_tune(
    base_model,
    real_data,
    epochs=100,
    learning_rate=0.00001  # Muy bajo para no sobreajustar
)

# Guardar modelo mejorado
torch.save(fine_tuned_model, 'models/checkpoint_10M_finetuned.pt')
```

---

## 💡 CONSEJOS IMPORTANTES

### ✅ HACER:
1. **Registrar TODO** cuando juegues en casinos
2. **Usar múltiples datasets** para fine-tuning
3. **Validar** con estadísticas oficiales del Nevada Gaming Board
4. **Sobremuestrear** situaciones raras (blackjacks, splits)

### ❌ NO HACER:
1. **Confíar ciegamente** en datos simulados (no son reales)
2. **Comprar datos** de sources dudosas (probablemente falsos)
3. **Violatar TOS** de casinos online
4. **Fine-tunear excesivamente** (overfitting a tus datos específicos)

---

## 📊 ESTADÍSTICAS DE REFERENCIA

### Win Rates Esperados:
- **Random player**: 35-38%
- **Basic strategy**: 42-44%
- **Card counting**: 44-47%
- **Tu modelo (10M)**: 45-46% (esperado)
- **Tu modelo + fine-tuning**: 46-48% (esperado)

### House Edge por Reglas:
- **Reglas favorables**: 0.13% (3:2 BJ, DAS, surrender)
- **Reglas promedio**: 0.5-1%
- **Reglas desfavorables**: 2%+ (6:5 BJ, no DAS, no surrender)

---

## 🔗 ENLACES RÁPIDOS

### Descargar Datasets:
- Kaggle Blackjack: https://www.kaggle.com/datasets?search=blackjack

### Estadísticas Oficiales:
- Nevada Gaming: https://www.gaming.nv.gov/
- UNLV Research: https://gaming.library.unlv.edu/

### Papers Académicos:
- Cambridge (datos reales): https://bit.ly/cambridge-blackjack
- ArXiv RL: https://arxiv.org/pdf/math/0006017

---

## 📞 PRÓXIMOS PASOS

1. ✅ **Esperar a que termine 10M episodios** (6-12 horas)
2. ✅ **Descargar datasets públicos** mientras tanto
3. ✅ **Jugar y registrar 1000-10000 manos reales**
4. ✅ **Fine-tunear el modelo con datos reales**
5. ✅ **Evaluar mejora**

---

**¡Listo para recopilar datos mientras tu modelo entrena!** 🎰🚀
