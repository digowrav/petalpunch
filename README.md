# petalpunch 🌸

A tic-tac-toe game powered by three distinct AI algorithms that demonstrate adversarial search, optimization, and reinforcement learning in a playable web experience.

**[Play it live →](https://digowrav.github.io/petalpunch/)**

---

## What This Project Demonstrates

This project is a hands-on showcase of three foundational AI paradigms, each powering a different difficulty level:

| Difficulty | Algorithm | Paradigm | Beatable? |
|:---:|:---:|:---:|:---:|
| Easy | Q-Learning | Reinforcement Learning | Yes |
| Medium | Depth-Limited Minimax | Adversarial Search | Yes |
| Hard | Minimax + Alpha-Beta Pruning | Optimized Adversarial Search | No |

The game also features a visual decision tree that lets you see the AI's thought process in real time, and an info panel that explains every metric the game tracks.

---

## The Algorithms

### 1. Minimax (Medium & Hard)

Minimax is the foundational algorithm for two-player zero-sum games. It models the game as a tree where two players take turns: one trying to maximize a score (MAX), the other trying to minimize it (MIN).

**Core idea:** You assume your opponent plays perfectly, then choose the move that gives you the best outcome even in that worst case.

**How the game tree works:**

Every possible game of tic-tac-toe can be represented as a tree. The root is the current board state. Each node's children represent the board states that result from each possible move. The tree ends at leaf nodes — positions where someone has won or the board is full (draw).

```
          Current Board (MIN's turn - AI choosing)
         /          |          \
    Move to A    Move to B    Move to C
      /   \        / \          |
   ...    ...    ...  ...      ...
    |      |      |    |        |
  +10    -10     0   +10      -10
  (you   (AI   (draw) (you   (AI
   win)   wins)        win)   wins)
```

**The recursive algorithm:**

At each node, the current player picks the child that's best for them. MAX picks the child with the highest score, MIN picks the child with the lowest score. This propagates scores from the leaves up to the root.

```
minimax(state, depth, is_maximizing):
    if terminal(state) or depth == 0:
        return evaluate(state)
    
    if is_maximizing:
        best = -infinity
        for each child of state:
            best = max(best, minimax(child, depth-1, false))
        return best
    else:
        best = +infinity
        for each child of state:
            best = min(best, minimax(child, depth-1, true))
        return best
```

**Utility function:**

The evaluation function assigns scores to terminal states:

| Outcome | Score | Why |
|:---:|:---:|:---|
| Human wins | +10 + remaining_depth | Positive (good for MAX); depth bonus prefers faster wins. |
| AI wins | -10 - remaining_depth | Negative (good for MIN); depth bonus prefers faster wins for AI too. |
| Draw | 0 | Neutral outcome. |

The depth bonus is a subtle but important detail. Without it, the AI might delay winning (since all wins score the same). With `+depth`, a win in 2 moves scores higher than a win in 6 moves, so the AI always takes the fastest path to victory.

**Time complexity:** $O(b^d)$ where b is the branching factor (average number of legal moves) and d is the depth. For tic-tac-toe, the full game tree has at most 9! = 362,880 leaf nodes (small enough to solve completely).

**Medium mode** uses depth-limited minimax (depth = 3). This means the AI only looks 3 moves ahead instead of searching to the end. Because it can't see the full future, it can make suboptimal moves, so you can beat it.

**Hard mode** uses full-depth minimax (depth = 9), which searches the entire game tree. This means the AI plays perfectly. It's mathematically impossible to beat and the best you can do is draw.

---

### 2. Alpha-Beta Pruning (Hard)

Alpha-beta pruning is an optimization of minimax that produces the exact same result while exploring far fewer nodes. It works by maintaining two values during search:

- **Alpha (α):** The best score MAX can guarantee so far (a lower bound)
- **Beta (β):** The best score MIN can guarantee so far (an upper bound)

**The key insight:** If at any point β ≤ α, we can stop searching that branch, because one player already has a better option elsewhere, so neither player would allow the game to reach this branch.

**Example of a cutoff:**

```
        MIN node (AI choosing)
       /                \
  child A = -5       child B = ?
  (already found)    (still searching...)
```

Suppose MIN has already found child A with score -5. MIN will pick at most -5. Now suppose child B's first sub-child scores +3. This means child B will score at least +3 (since B's children are MAX nodes that maximize). But MIN would never pick +3 over -5 — so we can prune the rest of child B's subtree without changing the result.

**The pruning algorithm:**

```
alphabeta(state, depth, alpha, beta, is_maximizing):
    if terminal(state) or depth == 0:
        return evaluate(state)
    
    if is_maximizing:
        value = -infinity
        for each child of state:
            value = max(value, alphabeta(child, depth-1, alpha, beta, false))
            alpha = max(alpha, value)
            if beta <= alpha:
                break    ← beta cutoff (prune remaining children)
        return value
    else:
        value = +infinity
        for each child of state:
            value = min(value, alphabeta(child, depth-1, alpha, beta, true))
            beta = min(beta, value)
            if beta <= alpha:
                break    ← alpha cutoff (prune remaining children)
        return value
```

**Efficiency:** In the best case (moves ordered optimally), alpha-beta pruning reduces the number of nodes explored from $O(b^d)$ to $O(b^(d/2))$. This effectively doubles the searchable depth for the same computation. In tic-tac-toe, this typically prunes 30-60% of the game tree.

The game tracks pruning in real time. The "pruned" counter in the stats bar shows how many branches were skipped during each move.

---

### 3. Q-Learning (Easy)

Q-Learning is a model-free, off-policy reinforcement learning algorithm. Unlike minimax, which explicitly searches a game tree, Q-learning learns a policy by playing thousands of games and updating a table of state-action values.

**The Q-table:**

The Q-table maps (state, action) pairs to expected future rewards. For tic-tac-toe:
- State = the current board configuration (e.g., `[X, O, _, _, X, _, O, _, _]`)
- Action = which cell to play (0-8)
- Q(s, a) = the expected total future reward of taking action a in state s

**The update rule:**

After each move, the agent updates its Q-value using the Bellman equation:

$$Q(s, a) ← Q(s, a) + α · [r + γ · max_{a'} Q(s', a') - Q(s, a)]$$


Where:
- $s$ = current state, a = action taken
- $r$ = immediate reward received
- $s'$ = next state after taking action a
- $\alpha$ = learning rate (0.1) $\rightarrow$ how fast the agent updates its beliefs
- $\gamma$ = discount factor (0.95) $\rightarrow$  how much it values future vs immediate rewards
- $\text{max}_{a'} Q(s', a')$ = the best Q-value achievable from the next state

**Intuition:** The term $r + γ · \text{max} Q(s', a')$ is the "target" (what the agent thinks the total future reward should be). The term $Q(s, a)$ is the current estimate. The update nudges the current estimate toward the target by a step of size α.

**Exploration vs exploitation (epsilon-greedy):**

During training, the agent faces a dilemma: should it exploit what it already knows (pick the highest Q-value) or explore unknown actions (try random moves)?

The epsilon-greedy strategy balances this:
- With probability ε: take a random action (explore)
- With probability 1-ε: take the best known action (exploit)

ε starts at 1.0 (pure exploration) and decays to 0.01 over training, so the agent explores widely at first then gradually shifts to exploiting its learned knowledge.

**Reward structure:**

| Outcome | Reward | Rationale |
|:---:|:---:|:---|
| Win | +1.0 | Strong positive signal |
| Loss | -1.0 | Strong negative signal |
| Draw | +0.3 | Slight positive — drawing against a strong opponent is good |
| Non-terminal | 0.0 | No immediate reward for intermediate moves |

The +0.3 draw reward is important. In tic-tac-toe, a perfect player can never lose — only win or draw. Giving draws a small positive reward encourages the agent to play conservatively against strong opponents rather than taking risky moves that might lead to losses.

**Self-play training:**

The agent trains by playing against itself. It plays as both Player 1 and Player 2, so it learns to play well from either position. Over 100,000+ games, it builds a comprehensive Q-table covering thousands of board states.

After training, the Q-table is exported as a JSON file (~300KB) and loaded by the browser. When the RL agent needs to make a move, it looks up the current board state and picks the action with the highest Q-value — no server needed.

**Training results:**

| Opponent | Win Rate | Draw Rate | Loss Rate |
|:---:|:---:|:---:|:---:|
| Random | ~72% | ~22% | ~6% |
| Minimax | 0% | ~95% | ~5% |

The agent can't beat a perfect minimax player (that's mathematically impossible in tic-tac-toe), but it plays near-optimally, drawing most games.

---

## Game Features

### Visual Decision Tree

On Hard mode, clicking "Show AI Thinking" opens an interactive visualization of the minimax game tree. Each node shows:
- A mini board preview of that game state
- The minimax score (pink = favorable for you, purple = favorable for AI)
- The chosen path highlighted in pink
- Pruned branches marked with ✂

You can expand the tree level by level using the +/- controls to explore deeper into the AI's decision process.

### AI Explainer

After each AI move, a text explanation appears describing what the AI did. For example: "Alpha-beta pruning explored 356 states, pruned 116 branches, and chose top-left — the provably optimal move."

### Stats Bar

The stats bar tracks three metrics in real time:
- **Nodes Explored** — how many game states the AI evaluated
- **Pruned** — how many branches alpha-beta pruning skipped
- **Depth** — how many moves ahead the AI searched

Click the **?** button next to the stats bar for detailed explanations of each metric.

---

## Tech Stack

| Layer | Technology |
|:---|:---|
| Frontend | HTML, CSS, JavaScript (no frameworks) |
| AI (search) | Minimax + Alpha-Beta Pruning (JavaScript) |
| AI (learning) | Q-Learning self-play training (Python) |
| Styling | CSS animations, glassmorphism, pixel art sprites |
| Fonts | Press Start 2P, Silkscreen (Google Fonts) |
| Deployment | GitHub Pages (static, no server needed) |

---

## Project Structure

```
petalpunch/
├── training/                    ← Python RL pipeline
│   ├── train_agent.py           ← Q-learning training script
│   ├── requirements.txt
│   ├── README.md
│   ├── q_table.json             ← trained policy (exported)
│   └── training_history.json    ← training metrics
├── assets/
│   ├── q_table.json             ← trained policy (loaded by browser)
│   ├── background.jpg           
│   ├── bunny.png                
│   ├── flower.png               ← AI player piece
│   ├── heart-brooch.png         
│   └── heart-gem.png            ← human player piece
├── css/
│   └── styles.css
├── js/
│   ├── game.js                  ← game logic + AI algorithms
│   └── tree-visualizer.js       ← decision tree rendering
├── index.html
└── README.md
```

---

## Run Locally

**Play the game:**

```bash
cd petalpunch
python3 -m http.server 8080
# open http://localhost:8080
```

**Retrain the RL agent:**

```bash
cd training
pip install -r requirements.txt
python train_agent.py
# copy the new q_table.json to assets/
cp q_table.json ../assets/
```

---

## Concepts Index

For quick reference, these are the AI/ML concepts that appear in the codebase:

| Concept | File | Function/Section |
|:---|:---|:---|
| Minimax algorithm | `js/game.js` | `minimax()` |
| Alpha-beta pruning | `js/game.js` | `minimax()` (alpha/beta params) |
| Game tree construction | `js/game.js` | `minimax()` (buildTree param) |
| Utility evaluation | `js/game.js` | `evaluate()` |
| Q-learning update rule | `training/train_agent.py` | `QLearningAgent.update()` |
| Epsilon-greedy exploration | `training/train_agent.py` | `QLearningAgent.get_action()` |
| Self-play training | `training/train_agent.py` | `train_agent()` |
| Q-table policy export | `training/train_agent.py` | `export_q_table()` |
| Q-table policy loading | `js/game.js` | `loadQTable()` |
| RL agent move selection | `js/game.js` | `getQLearningMove()` |
| Tree visualization | `js/tree-visualizer.js` | `renderGameTreeSVG()` |

---

## Context

This project uses the following context and topics:

- Adversarial search, minimax, alpha-beta pruning, reinforcement learning, Q-learning
- Evaluation methodology, precision/recall analysis
- Decision trees, model evaluation, cross-validation

---

## License

MIT
