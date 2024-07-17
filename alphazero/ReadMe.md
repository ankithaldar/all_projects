# AlphaZero for Tic-Tac-Toe and Connect 4

## (work In progress.)

### Overview
This repository contains an implementation of the AlphaZero algorithm applied to the games of Tic-Tac-Toe and Connect Four. AlphaZero is a reinforcement learning algorithm that learns to play games solely through self-play and without any human knowledge or data.

### Structure
The project is organized into the following main components:
- __tictactoe.py__: Defines the game environments for Tic-Tac-Toe.
- __connect_four.py__: Defines the game environments for Connect Four.
- __mcts.py__: Implements the Monte Carlo Tree Search (MCTS) algorithm.
- __model.py__: Defines the neural network architecture used for policy and value estimation.
- __alphazero.py__: Contains the main AlphaZero training loop, including self-play, training, and evaluation.
- __test_tictactoe.py__: Provides a basic script for playing a game against the trained AI.

### Requirements
```
Python 3.6+
NumPy
TensorFlow
Other dependencies (e.g., for Discord webhook integration, if used)
```

### Usage
- Install dependencies:
```Bash
pip install numpy tensorflow
```

- __Modify game-specific parameters__: Adjust hyperparameters and game-specific settings in alpha_zero.py for desired behavior.
- __Train the model__: Run the alpha_zero.py script with appropriate arguments.
- __Play against the AI__: Use the test.py script to play against the trained model.

### Training Process
The AlphaZero algorithm follows these steps:

- __Self-play__: The agent plays games against itself, collecting training data.
- __Training__: The neural network is trained on the collected data using supervised learning.
- __Evaluation__: The trained model is evaluated against a strong baseline or human players.

### Key Components
- __Game environment__: Provides the rules and state representation for the game.
- __MCTS__: Implements the tree search algorithm for selecting moves.
- __Neural network__: Estimates the policy (probability distribution over actions) and value (win probability) of a given game state.
- __Self-play__: Generates training data through self-play.
- __Training__: Trains the neural network using supervised learning.

### Hyperparameters
The performance of AlphaZero can be influenced by various hyperparameters, including:

- Learning rate
- Batch size
- Number of self-play iterations
- Number of MCTS simulations
- Exploration constant (C)
- Temperature

Experiment with different hyperparameter values to optimize performance for specific games.

### Additional Notes
- The provided code is a basic implementation and can be further optimized and extended.
- Consider using a GPU for faster training.
- Explore different neural network architectures and hyperparameters.
- Implement additional features like exploration bonuses and Dirichlet noise.

By understanding the core components and experimenting with different configurations, you can apply AlphaZero to various games and achieve strong performance.

---
### References
- [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/pdf/1712.01815)
- [Paper-Walkthrough](https://youtu.be/0slFo1rV0EM)
- [MCTS-Explained](https://youtu.be/UXW2yZndl7U)
- [AlphaZero-Explained](https://youtu.be/62nq4Zsn8vc)
- [AlphaZero from Scratch – Machine Learning Tutorial](https://www.youtube.com/watch?v=wuSQpLinRB4)
- [AlphaZeroFromScratch - Codes & Trained Models](https://github.com/foersterrobert/AlphaZeroFromScratch)
- [Codes - AlphaZero General](https://github.com/suragnair/alpha-zero-general)
