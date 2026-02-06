# ML Blackjack - Deep Q-Network for Profitable Blackjack

A Deep Reinforcement Learning system that learns to play Blackjack profitably using Deep Q-Networks (DQN).

## 🎯 Project Overview

This project implements a complete DQN-based agent that learns to play blackjack through experience. The system includes:

- **Complete Blackjack Engine**: All standard rules (hit, stand, double, split, insurance, surrender)
- **Card Counting Integration**: Hi-Lo counting system included in the state representation
- **Gymnasium Environment**: Standard RL environment interface
- **DQN Architecture**: Deep Q-Network with target networks and experience replay
- **Comprehensive Metrics**: Performance tracking, ROI, Sharpe ratio, and more
- **Visualization Tools**: Training curves, win rates, profit tracking

## 📁 Project Structure

```
ML-BLACKJACK/
├── src/
│   ├── game/              # Blackjack game engine
│   │   ├── blackjack.py   # Main game logic
│   │   ├── deck.py        # Card and deck management
│   │   └── rules.py       # Game rules and hand evaluation
│   ├── environment/       # RL environment
│   │   └── blackjack_env.py  # Gymnasium-compatible environment
│   ├── agent/             # DQN agent
│   │   ├── dqn.py         # Neural network architecture
│   │   ├── replay_buffer.py  # Experience replay
│   │   └── trainer.py     # Training loop
│   ├── utils/             # Utilities
│   │   ├── metrics.py     # Performance metrics
│   │   └── visualization.py  # Plotting functions
│   └── main.py            # Entry point
├── tests/                 # Unit tests
├── models/                # Saved model checkpoints
├── logs/                  # Training logs
└── requirements.txt       # Dependencies
```

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Setup

1. Clone the repository (or navigate to the project directory):
```bash
cd "C:\Users\migue\Desktop\ML BLACKJACK"
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🎮 Usage

### Quick Start

#### 1. Test the Game Engine

Verify everything works by running the game test:

```bash
python src/main.py --mode test
```

This will play 5 random games and display the output, showing you how the game works.

#### 2. Train the Agent

Train a DQN agent from scratch:

```bash
python src/main.py --mode train --episodes 100000
```

Training parameters:
- `--episodes`: Number of episodes to train (default: 100,000)
- `--num-decks`: Number of decks (default: 6)
- `--learning-rate`: Learning rate (default: 0.0001)
- `--batch-size`: Training batch size (default: 64)
- `--visualize`: Generate training plots

Example with custom parameters:
```bash
python src/main.py --mode train \
    --episodes 500000 \
    --learning-rate 0.0001 \
    --batch-size 64 \
    --network-type standard \
    --visualize
```

#### 3. Evaluate a Trained Model

Evaluate a trained model and compare with baselines:

```bash
python src/main.py --mode evaluate \
    --episodes 10000 \
    --model-path models/dqn_blackjack_20240101_120000.pt
```

This will:
- Evaluate the DQN agent
- Evaluate Basic Strategy (baseline)
- Evaluate Random Strategy
- Compare all strategies

## 📊 Metrics and Evaluation

The system tracks comprehensive metrics:

### Performance Metrics
- **Win Rate**: Percentage of hands won
- **ROI**: Return on Investment percentage
- **Total Profit/Loss**: Net monetary result
- **Sharpe Ratio**: Risk-adjusted returns
- **Advantage Over House**: Edge per hand

### Comparison Baselines
- **Basic Strategy**: Standard optimal play (no card counting)
- **Random Strategy**: Random action selection

### Visualization
Training generates these visualizations:
- Training rewards over time
- Win rate progress
- Cumulative profit
- Loss curves
- Strategy comparison charts

## 🔧 Configuration

### Environment Parameters
- `--num-decks`: Number of decks in shoe (1-8)
- `--penetration`: Shoe penetration (0-1)
- `--allow-surrender`: Enable surrender rule
- `--dealer-hits-soft-17`: Dealer hits on soft 17

### Training Parameters
- `--learning-rate`: Adam optimizer learning rate
- `--gamma`: Discount factor (default: 0.99)
- `--buffer-size`: Experience replay buffer size
- `--batch-size`: Mini-batch size for training
- `--target-update-frequency`: Steps between target network updates
- `--epsilon-start`: Initial exploration rate
- `--epsilon-end`: Final exploration rate
- `--epsilon-decay-steps`: Steps for epsilon decay
- `--hidden-dims`: Hidden layer sizes (comma-separated)
- `--network-type`: 'standard' or 'dueling' architecture

## 📈 Expected Results

With proper training (1M+ episodes):

- **Win Rate**: 42-44% (vs ~30% random, ~42% basic strategy)
- **House Edge**: Should reduce to near-zero or slightly positive
- **Advantage**: Target 0.5-1.5% edge over the house

**Note**: Beating the house consistently is extremely difficult. The goal is to minimize house edge through optimal play and card counting awareness.

## 🧪 Testing

Run unit tests:

```bash
# Test game engine
python tests/test_game.py

# Test environment
python tests/test_env.py

# Run all tests with pytest
pytest tests/ -v
```

## 🎓 How It Works

### State Representation
The agent observes 9 features:
1. Player hand value (normalized)
2. Dealer up card (normalized)
3. Is soft hand (boolean)
4. True count (card counting, normalized)
5. Cards remaining ratio
6. Can split (boolean)
7. Can double (boolean)
8. Can surrender (boolean)
9. Can insure (boolean)

### Action Space
6 discrete actions:
- HIT (0)
- STAND (1)
- DOUBLE (2)
- SPLIT (3)
- INSURANCE (4)
- SURRENDER (5)

### Network Architecture
```
Input (9) → Dense(256) → ReLU → Dense(256) → ReLU → Dense(128) → ReLU → Output(6)
```

### Training Algorithm
1. **Epsilon-Greedy Exploration**: Balance exploration/exploitation
2. **Experience Replay**: Store and sample past experiences
3. **Target Network**: Stable Q-value estimation
4. **Huber Loss**: Robust loss function
5. **Gradient Clipping**: Prevent exploding gradients

## 📝 Technical Details

### Card Counting
The system uses the **Hi-Lo** counting system:
- Cards 2-6: +1
- Cards 7-9: 0
- Cards 10-A: -1

The "true count" (running count / remaining decks) is included in the state, allowing the agent to learn betting and playing strategies based on deck composition.

### Blackjack Rules Implemented
- ✅ Hit and Stand
- ✅ Double Down
- ✅ Split pairs
- ✅ Insurance (when dealer shows Ace)
- ✅ Surrender (optional, configurable)
- ✅ Multiple decks (1-8, configurable)
- ✅ Blackjack natural (pays 3:2)
- ✅ Soft hands (Ace as 1 or 11)
- ✅ Dealer hits on soft 17 (configurable)

## 🔬 Research Background

This implementation is based on:
- **DQN**: Mnih et al. (2015) "Human-level control through deep reinforcement learning"
- **Card Counting**: Thorp's "Beat the Dealer"
- **Basic Strategy**: Standard blackjack optimal play

## 🤝 Contributing

To extend or improve this project:

1. **Try different architectures**: Double DQN, Dueling DQN, Rainbow
2. **Add more features**: Detailed hand composition, betting strategies
3. **Experiment with hyperparameters**: Learning rate schedules, network sizes
4. **Implement advanced strategies**: Kelly criterion betting, team play

## 📄 License

This project is for educational and research purposes.

## ⚠️ Disclaimer

This software is for educational purposes only. The authors are not responsible for any use of this software for actual gambling. Casino blackjack has built-in house advantages, and card counting may be prohibited by casinos.

## 🙏 Acknowledgments

- OpenAI Gym/Gymnasium for the RL environment interface
- PyTorch team for the deep learning framework
- The reinforcement learning research community

---

**Good luck, and may the odds be ever in your favor! 🎰🃏**
