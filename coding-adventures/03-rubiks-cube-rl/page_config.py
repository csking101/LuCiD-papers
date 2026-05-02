"""Page configuration for Coding Adventure 03 — Solve a Rubik's Cube with RL."""

PAGE_DATA = {
    "num": "03",
    "title": "Solve a Rubik's Cube with RL",
    "status": "Done",
    "subtitle": "PPO + curriculum learning on a 2x2 Pocket Cube — 4 source files, 1,647 lines, 162 tests",
    "github_url": "https://github.com/csking101/LuCiD-papers/tree/main/coding-adventures/03-rubiks-cube-rl",
    "last_updated": "May 2026",

    "tags": ["PyTorch", "Rich CLI", "PPO", "Curriculum Learning", "GPU Optional"],

    "tldr": r"""<p>Train a neural network to solve a <strong>2&times;2 Pocket Cube</strong> from scratch using <strong>PPO with curriculum learning</strong>. The agent starts at depth-1 scrambles and advances when solve rate exceeds 80%, progressing through depths 1&ndash;7. Built with PyTorch + Rich. GPU optional (uses CUDA if available). Trains in ~2.5 minutes.</p>""",

    # ─── Table of Contents (sidebar) ───
    "toc": [
        {"id": "what-happens",       "label": "What Happens"},
        {"id": "screenshot-01",      "label": "Cube Environment",       "is_screenshot": True},
        {"id": "screenshot-02",      "label": "Curriculum Training",     "is_screenshot": True},
        {"id": "screenshot-03",      "label": "Live Solving",           "is_screenshot": True},
        {"id": "screenshot-04",      "label": "Stress Test",            "is_screenshot": True},
        {"id": "screenshot-05",      "label": "Comparison + Summary",   "is_screenshot": True},
        {"id": "playing-around",     "label": "Playing Around"},
        {"id": "llm-parallel",       "label": "LLM Parallel Mapping"},
        {"id": "architecture",       "label": "Architecture"},
        {"id": "project-structure",  "label": "Project Structure"},
        {"id": "key-concepts",       "label": "Key Concepts"},
        {"id": "quick-start",        "label": "Quick Start"},
        {"id": "related-papers",     "label": "Related Papers"},
    ],

    # ─── Nav bar (mobile) ───
    "nav": [
        {"id": "what-happens",       "label": "What Happens"},
        {"id": "playing-around",     "label": "Playing Around"},
        {"id": "llm-parallel",       "label": "LLM Parallel"},
        {"id": "architecture",       "label": "Architecture"},
        {"id": "quick-start",        "label": "Quick Start"},
    ],

    # ─── Content blocks ───
    "content": [

        # ── What Happens ──
        {"type": "note", "id": "what-happens", "heading": "What Happens", "html": r"""
<p>The demo walks you through five phases that train and evaluate an RL agent on the 2&times;2 Pocket Cube.</p>

<h3>Phase 1: Cube World</h3>
<p>Explore the cube environment: solved state, moves, scramble/solution demonstrations, and the <strong>mapping to LLM alignment</strong>. The 2&times;2 Pocket Cube has <strong>3,674,160 reachable states</strong>, 6 quarter-turn moves (U, U&prime;, R, R&prime;, F, F&prime;), and a God&rsquo;s number of 14.</p>
<ul>
    <li><strong>State encoding:</strong> 24 stickers &times; 6 colours = 144-dimensional one-hot vector</li>
    <li><strong>Fixed corner:</strong> DLB corner is locked to remove rotational equivalence</li>
    <li><strong>Reward:</strong> +1.0 on solve, &minus;0.02 per step</li>
</ul>

<h3>Phase 2: PPO Training with Curriculum</h3>
<p>The agent trains using PPO with <strong>curriculum learning</strong>: start at depth-1 scrambles, advance to the next depth when solve rate reaches &ge;80%. Training continues through depths 1&ndash;7. This is <em>analogous to progressive difficulty in LLM RLHF</em> &mdash; easy tasks first, harder ones later.</p>
<p><strong>You&rsquo;ll see:</strong></p>
<ul>
    <li>Live training dashboard (solve rate, episode length, loss, entropy)</li>
    <li>Sparkline showing solve rate trend across all episodes</li>
    <li>Curriculum summary table showing advancement through depths</li>
</ul>

<h3>Phase 3: Live Solving</h3>
<p>Watch the trained agent solve cubes at depths 1, 3, and 5. For each attempt, see the scrambled state, the sequence of moves, and the solved result.</p>

<h3>Phase 4: Stress Test</h3>
<p>Test the agent across scramble depths 1&ndash;10 with 100 cubes each. A bar chart shows the <strong>capability frontier</strong> &mdash; the deepest depth where solve rate exceeds 50%.</p>

<h3>Phase 5: Random vs Trained Agent</h3>
<p>Compare the PPO agent against a uniformly random agent across depths 1&ndash;7. The comparison demonstrates how dramatically RL improves over random search, and where the agent&rsquo;s capability falls off.</p>
"""},

        # Screenshot 1
        {"type": "screenshot", "num": 1, "title": "2x2 Pocket Cube Environment",
         "desc": "The solved cube in unfolded cross layout, environment specifications (3.6M states, 144-dim one-hot encoding, 6 moves, God's number 14), and a scrambled cube with its solution. The DLB corner is fixed to eliminate rotational equivalence.",
         "src": "01_cube_world.svg"},

        {"type": "note", "id": "phase2-note", "heading": "Phase 2: Curriculum PPO Training", "html": r"""
<p>The curriculum trainer starts at depth 1 (one random move from solved) and advances when solve rate &ge;80%. At each depth, PPO collects rollouts of 128 episodes, computes GAE advantages, and runs clipped surrogate updates. The maximum budget per depth is 3,072 episodes.</p>
<p>Typical results:</p>
<ul>
    <li><strong>Depths 1&ndash;4:</strong> Advance quickly (80%+ solve rate within 768&ndash;2,304 episodes)</li>
    <li><strong>Depth 5:</strong> Reaches ~57% (harder scrambles, more episodes needed)</li>
    <li><strong>Depths 6&ndash;7:</strong> Plateau at 34&ndash;47% &mdash; the <strong>alignment tax</strong> analogy</li>
</ul>
"""},

        # Screenshot 2
        {"type": "screenshot", "num": 2, "title": "Curriculum PPO Training",
         "desc": "Training summary across all curriculum depths. Depths 1-4 advance with 80%+ solve rate. Deeper scrambles (5-7) hit the episode budget before reaching the threshold, demonstrating diminishing returns — analogous to the alignment tax in LLM RLHF.",
         "src": "02_training.svg"},

        {"type": "note", "id": "phase3-note", "heading": "Phase 3: Live Solving", "html": r"""
<p>The trained agent solves cubes at depths 1, 3, and 5. Each attempt shows:</p>
<ul>
    <li>The scrambled cube state (unfolded cross)</li>
    <li>The move sequence the agent chose</li>
    <li>Whether it solved successfully, and in how many steps</li>
</ul>
<p>At depth 1, the agent consistently finds the one-move solution. At depth 5, it often finds solutions in 5&ndash;7 moves.</p>
"""},

        # Screenshot 3
        {"type": "screenshot", "num": 3, "title": "Agent Solving Cubes",
         "desc": "Three solve demonstrations at depths 1, 3, and 5. Each shows the scrambled state, the agent's chosen moves, and the solved result. The depth-5 cube is solved in 5 moves — near-optimal given the scramble depth.",
         "src": "03_solve_demos.svg"},

        {"type": "note", "id": "phase4-note", "heading": "Phase 4: Stress Test", "html": r"""
<p>100 random cubes are tested at each depth from 1 to 10. The bar chart reveals the <strong>capability frontier</strong> &mdash; typically depth 5 at &ge;50% solve rate. Beyond this, performance degrades gracefully.</p>
"""},

        # Screenshot 4
        {"type": "screenshot", "num": 4, "title": "Stress Test — Solve Rate by Depth",
         "desc": "Bar chart showing solve rates from depth 1 (100%) to depth 10 (15%). The capability frontier (≥50% solve rate) is at depth 5. Performance degrades gradually beyond the training curriculum (depths 1-7).",
         "src": "04_stress_test.svg"},

        {"type": "note", "id": "phase5-note", "heading": "Phase 5: Comparison", "html": r"""
<p>Side-by-side comparison of the trained PPO agent vs a uniformly random agent. The random agent can occasionally solve depth-1 (1/6 chance per move), but is effectively useless beyond depth 2. The trained agent maintains meaningful solve rates through depth 6.</p>
"""},

        # Screenshot 5
        {"type": "screenshot", "num": 5, "title": "Random vs Trained Agent + Summary",
         "desc": "Comparison table (random vs trained solve rates and average steps), final summary (depths passed, total episodes, training time), stress test results, and the LLM parallel mapping table connecting cube RL concepts to LLM alignment.",
         "src": "05_comparison.svg"},

        {"type": "break"},

        # ── Playing Around ──
        {"type": "note", "id": "playing-around", "heading": "Playing Around", "html": r"""
<h3>Tuning hyperparameters</h3>
<p>Edit the <code>TrainConfig</code> defaults in <code>train.py</code>:</p>
<pre><code>max_depth = 7               # Curriculum goes up to depth 7
advance_threshold = 0.80    # Advance when solve rate ≥ 80%
max_episodes_per_depth = 3000  # Budget per depth
episodes_per_rollout = 128  # PPO batch size
lr = 3e-4                   # Learning rate
clip_epsilon = 0.2          # PPO clip range
entropy_coef = 0.01         # Entropy bonus coefficient</code></pre>

<h3>Key experiments</h3>
<ul>
    <li><strong>Increase max_depth to 10:</strong> See how far the agent can learn with more training budget</li>
    <li><strong>Set advance_threshold = 0.95:</strong> Require near-perfect mastery before advancing — slower but potentially stronger at deep depths</li>
    <li><strong>Set entropy_coef = 0.1:</strong> More exploration — useful if the agent gets stuck at a particular depth</li>
    <li><strong>Set max_episodes_per_depth = 10000:</strong> Much larger budget — can the agent eventually master depth 7?</li>
    <li><strong>Remove curriculum (start at depth 7):</strong> Train directly on hard scrambles to see why curriculum matters</li>
</ul>

<h3>Modifying the network</h3>
<p>Edit <code>policy.py</code> to change the architecture:</p>
<ul>
    <li><strong>Wider:</strong> Increase hidden dims from 256/128 to 512/256 for more capacity</li>
    <li><strong>Deeper:</strong> Add a third hidden layer for more representational power</li>
    <li><strong>Residual connections:</strong> Add skip connections for better gradient flow</li>
</ul>

<h3>Observing specific phenomena</h3>
<table>
    <tr><th>Phenomenon</th><th>How to trigger</th><th>What to look for</th></tr>
    <tr><td><strong>Curriculum benefit</strong></td><td>Compare curriculum vs direct depth-7 training</td><td>Curriculum reaches higher solve rates faster</td></tr>
    <tr><td><strong>Diminishing returns</strong></td><td>Watch depths 5-7 during training</td><td>Solve rate plateaus despite more episodes</td></tr>
    <tr><td><strong>Exploration collapse</strong></td><td>Set <code>entropy_coef = 0.0</code></td><td>Agent converges to a single action regardless of state</td></tr>
    <tr><td><strong>Overfitting to depth</strong></td><td>Train only at depth 1 (max_depth=1)</td><td>Agent solves depth 1 perfectly but fails at depth 2+</td></tr>
</table>
"""},

        {"type": "break"},

        # ── LLM Parallel Mapping ──
        {"type": "note", "id": "llm-parallel", "heading": "LLM Parallel Mapping", "html": r"""
<table>
    <tr><th>Cube RL</th><th>LLM Alignment</th></tr>
    <tr><td>24 stickers → 144-dim one-hot</td><td>Tokens → embedding vectors</td></tr>
    <tr><td>6 moves (U, U', R, R', F, F')</td><td>Vocabulary of next-token choices</td></tr>
    <tr><td>One-hot state encoding</td><td>Tokenisation + positional encoding</td></tr>
    <tr><td>Curriculum depth 1→7</td><td>Progressive RLHF: easy → hard preferences</td></tr>
    <tr><td>PPO clipped objective</td><td>Same optimizer used in ChatGPT/Claude RLHF</td></tr>
    <tr><td>Dense reward shaping</td><td>Reward model signal guiding token choices</td></tr>
    <tr><td>Solve rate plateau at deep scrambles</td><td>Alignment tax — harder alignment = lower capability</td></tr>
    <tr><td>God's number = 14</td><td>Optimal solution length ≈ ideal response quality</td></tr>
</table>
"""},

        {"type": "break"},

        # ── Architecture ──
        {"type": "note", "id": "architecture", "heading": "Architecture", "html": r"""
<h3>Cube Environment (<code>cube.py</code>)</h3>
<ul>
    <li>2&times;2 Pocket Cube with 24 stickers (4 per face, 6 faces)</li>
    <li>State: list of 24 colour indices (0&ndash;5)</li>
    <li>Moves: U, U&prime;, R, R&prime;, F, F&prime; (quarter-turns of top, right, front faces)</li>
    <li>DLB corner fixed &mdash; removes rotational equivalence</li>
    <li>One-hot encoding: 24 positions &times; 6 colours = 144 dimensions</li>
    <li>Reward: +1.0 on solve, &minus;0.02 per step</li>
    <li>Scramble generation with configurable depth</li>
</ul>

<h3>Policy Network (<code>policy.py</code>)</h3>
<ul>
    <li>Input: 144-dim one-hot state vector</li>
    <li>Architecture: <code>Linear(144, 256) &rarr; ReLU &rarr; Linear(256, 128) &rarr; ReLU</code></li>
    <li>Actor head: <code>Linear(128, 6)</code> &rarr; action logits</li>
    <li>Critic head: <code>Linear(128, 1)</code> &rarr; state value</li>
    <li>Orthogonal initialisation for stable training</li>
</ul>

<h3>Curriculum Trainer (<code>train.py</code>)</h3>
<ul>
    <li>Curriculum: depth 1 &rarr; max_depth, advance when solve rate &ge; threshold</li>
    <li>PPO with clipped surrogate objective and GAE(&lambda;=0.95)</li>
    <li><code>collect_rollouts()</code>: gather episodes at current depth</li>
    <li><code>ppo_update()</code>: compute advantages, update policy/value networks</li>
    <li><code>evaluate_policy()</code>: test solve rate at specific depths</li>
</ul>

<h3>Visualisation (<code>viz.py</code>)</h3>
<ul>
    <li>All rendering via Rich (no external display needed)</li>
    <li>Unfolded cube cross layout with coloured stickers</li>
    <li>Curriculum summary table, training progress panels</li>
    <li>Solve attempt display, stress test bar charts</li>
    <li>Random vs trained comparison table</li>
</ul>
"""},

        {"type": "break"},

        # ── Project Structure ──
        {"type": "note", "id": "project-structure", "heading": "Project Structure", "html": r"""
<pre><code>03-rubiks-cube-rl/
├── app.py              # Main 5-phase terminal demo (355 lines)
├── cube.py             # 2x2 Pocket Cube environment (272 lines)
├── policy.py           # Actor-critic MLP with PPO helpers (114 lines)
├── train.py            # Curriculum trainer + PPO + dataclasses (483 lines)
├── viz.py              # Rich terminal rendering (423 lines)
├── requirements.txt    # torch, numpy, rich
└── tests/
    ├── test_cube.py       # 89 tests — moves, scramble, tensor, step, DLB
    ├── test_policy.py     # 20 tests — forward pass, action selection, gradients
    ├── test_train.py      # 29 tests — GAE, rollouts, PPO update, curriculum
    └── test_viz.py        # 24 tests — all Rich renderable components
                           # ─────────
                           # 162 tests total</code></pre>
"""},

        # ── Key Concepts ──
        {"type": "note", "id": "key-concepts", "heading": "Key Concepts Demonstrated", "html": r"""
<ul>
    <li><strong>PPO clipped surrogate</strong> &mdash; the same RL optimizer used in ChatGPT, Claude, and other RLHF systems</li>
    <li><strong>Curriculum learning</strong> &mdash; progressive difficulty, mirroring how LLMs are trained on easy examples before hard ones</li>
    <li><strong>Generalised Advantage Estimation (GAE)</strong> &mdash; variance reduction for policy gradients</li>
    <li><strong>One-hot state encoding</strong> &mdash; analogous to tokenisation + positional encoding in transformers</li>
    <li><strong>Capability frontier</strong> &mdash; the solve rate drop-off at deeper scrambles mirrors the alignment tax</li>
    <li><strong>Actor-critic architecture</strong> &mdash; shared backbone with policy and value heads, standard in modern RL</li>
    <li><strong>God&rsquo;s number</strong> &mdash; the theoretical optimal solution length (14 for 2&times;2), setting an upper bound on difficulty</li>
</ul>
"""},

        {"type": "break"},

        # ── Quick Start ──
        {"type": "note", "id": "quick-start", "heading": "Quick Start", "html": r"""
<pre><code># From the repo root
cd coding-adventures/03-rubiks-cube-rl

# Install dependencies (if not using the repo venv)
pip install -r requirements.txt

# Run the demo
python app.py</code></pre>
<p>Training takes ~2.5 minutes on GPU or ~5 minutes on CPU. Press Enter to advance between phases.</p>

<h3>Running Tests</h3>
<pre><code># All 162 tests (~2 seconds)
python -m pytest tests/ -v

# Quick smoke test
python -m pytest tests/ -x -q</code></pre>
"""},

    ],  # end content

    # ─── Related Papers ───
    "related_papers": [
        {
            "arxiv_id": "1707.06347",
            "title": "Proximal Policy Optimization Algorithms",
            "desc": "Schulman et al. (2017) — PPO, the RL optimizer used to train the cube-solving agent.",
            "url": "../../papers/1707.06347/",
        },
        {
            "arxiv_id": "1706.03741",
            "title": "Deep Reinforcement Learning from Human Preferences",
            "desc": "Christiano et al. (2017) — The foundational RLHF paper. Curriculum RL concepts apply directly.",
            "url": "../../papers/1706.03741/",
        },
        {
            "arxiv_id": "2009.01325",
            "title": "Learning to Summarize from Human Feedback",
            "desc": "Stiennon et al. (2020) — Applies RLHF to text summarization, same PPO optimizer at scale.",
            "url": "../../papers/2009.01325/",
        },
    ],
}
