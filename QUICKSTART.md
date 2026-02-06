# Quick Start Guide - ML Blackjack

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies (Already Done ✅)
```bash
pip install -r requirements.txt
```

### Step 2: Verify Installation
```bash
# Run environment tests
python tests/test_env.py

# You should see: "All tests passed! [OK]"
```

### Step 3: Choose Your Path

---

## 🎮 Path A: Just Want to See It Work?

### Test the Game (1 minute)
```bash
python src/main.py --mode test
```
This plays 5 random games and shows you the blackjack engine in action.

---

## 🏋️ Path B: Train Your Own Agent

### Quick Training Test (5 minutes)
```bash
python src/main.py --mode train --episodes 1000 --log-frequency 100
```

### Serious Training (1-2 hours)
```bash
python src/main.py --mode train --episodes 100000 --visualize --save-frequency 10000
```

**What happens:**
- Agent explores random actions initially
- Gradually learns from experience
- Saves model checkpoints every 10K episodes
- Generates visualizations at the end

**Expected output:**
```
Episode 1000/100000
  Average Reward: -0.15
  Win Rate: 42%
  Epsilon: 0.99
```

---

## 📊 Path C: Evaluate a Trained Model

### After Training, Evaluate Performance
```bash
python src/main.py --mode evaluate --episodes 10000 --model-path models/checkpoint_ep500.pt
```

**What you'll see:**
- DQN agent performance
- Basic Strategy baseline
- Random Strategy baseline
- Comparison metrics

**Sample output:**
```
DQN Results:
  Win Rate: 43.5%
  Mean Reward: -$0.05
  Total Profit: -$500

Basic Strategy:
  Win Rate: 47.0%
  Mean Reward: +$0.028
  Total Profit: +$280

Comparison:
  The DQN needs more training to beat basic strategy!
```

---

## 🎯 Understanding the Modes

### `--mode test`
- **Purpose:** Verify the game engine works
- **Duration:** ~30 seconds
- **Output:** 5 complete games with detailed state display

### `--mode train`
- **Purpose:** Train a DQN agent from scratch
- **Duration:** Depends on episodes
  - 1,000 episodes: ~1 minute
  - 10,000 episodes: ~5 minutes
  - 100,000 episodes: ~30-60 minutes
- **Output:** Trained model in `models/` folder

### `--mode evaluate`
- **Purpose:** Test a trained model against baselines
- **Duration:** ~2-3 minutes for 10K episodes
- **Output:** Performance metrics and comparisons

---

## 📈 Expected Training Progression

| Episodes | Expected Win Rate | Expected Advantage |
|----------|-------------------|-------------------|
| 1,000    | 35-38%           | Still losing      |
| 10,000   | 40-42%           | Approaching even  |
| 100,000  | 43-45%           | Slight edge       |
| 500,000+ | 44-46%           | Target advantage  |

**Goal:** Beat the 0.5% house edge with a 0.5-1.5% player advantage.

---

## 🔧 Common Commands

### Train with custom hyperparameters
```bash
python src/main.py --mode train \
    --episodes 50000 \
    --learning-rate 0.0001 \
    --batch-size 64 \
    --hidden-dims 512,256,128 \
    --network-type dueling
```

### Evaluate latest model
```bash
# First, find your latest model
ls models/*.pt

# Then evaluate it
python src/main.py --mode evaluate \
    --episodes 10000 \
    --model-path models/your_latest_model.pt
```

### Train with visualization
```bash
python src/main.py --mode train \
    --episodes 10000 \
    --visualize
```

Creates plots in `models/visualizations/`

---

## 📁 Where to Find Things

| What | Location |
|------|----------|
| Trained models | `models/*.pt` |
| Training metrics | `models/training_metrics_*.json` |
| Visualizations | `models/visualizations/*.png` |
| Logs | Console output |
| Source code | `src/` |

---

## 🎓 Learning Path

### 1. Understand the Game
```bash
python src/main.py --mode test
```
Watch how blackjack works, observe the state, actions, and rewards.

### 2. See Basic Strategy Performance
Edit `src/main.py` to run just basic strategy evaluation (no training needed).

### 3. Short Training Run
```bash
python src/main.py --mode train --episodes 1000
```
Observe how the agent's win rate changes (initially very poor).

### 4. Extended Training
```bash
python src/main.py --mode train --episodes 50000 --visualize
```
Let it train, then check the generated plots.

### 5. Analyze Results
```bash
python src/main.py --mode evaluate --episodes 10000 --model-path models/latest_model.pt
```
Compare against baselines to see if learning is working.

---

## ⚠️ Important Notes

1. **Training Time:** To get good results, you need 100K+ episodes
2. **Randomness:** Results vary between runs (RL is stochastic)
3. **Convergence:** The loss might not decrease monotonically (this is normal)
4. **Baselines:** Basic Strategy is strong - beating it requires significant training
5. **Expectations:** Don't expect to beat the casino with <10K episodes

---

## 🆘 Troubleshooting

### "Module not found" error
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

### "ImportError: attempted relative import"
```bash
# Make sure you're running from the project root
cd "C:\Users\migue\Desktop\ML BLACKJACK"
python src/main.py --mode test
```

### Low win rate initially
This is normal! The agent starts with random actions (epsilon=1.0) and learns gradually.

### Out of memory
Reduce `--buffer-size` or `--batch-size`:
```bash
python src/main.py --mode train --buffer-size 50000 --batch-size 32
```

---

## 🎉 Success Indicators

You'll know it's working when:
- ✅ Tests pass without errors
- ✅ Training completes and saves models
- ✅ Win rate improves from ~30% (random) to ~40%+
- ✅ Evaluation shows comparison between strategies
- ✅ Visualizations are generated (with --visualize flag)

---

**Ready to train your first RL agent? Start with:**
```bash
python src/main.py --mode train --episodes 1000 --log-frequency 100
```

Good luck! 🍀
