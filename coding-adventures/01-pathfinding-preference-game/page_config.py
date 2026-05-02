"""Page configuration for Coding Adventure 01 — Path-Finding Preference Game."""

PAGE_DATA = {
    "num": "01",
    "title": "Path-Finding Preference Game",
    "status": "Done",
    "subtitle": "Full RLHF pipeline in a grid world — 7 source files, 2,843 lines, 201 tests",
    "github_url": "https://github.com/csking101/LuCiD-papers/tree/main/coding-adventures/01-pathfinding-preference-game",
    "last_updated": "May 2026",

    "tags": ["PyTorch", "Rich CLI", "RLHF", "PPO", "Bradley-Terry", "KL Divergence", "No GPU Required"],

    "tldr": r"""<p>An interactive terminal demo of <strong>Reinforcement Learning from Human Feedback (RLHF)</strong> applied to grid-world navigation. Every step mirrors the LLM alignment pipeline &mdash; you are the human annotator. Pre-train an agent, rate trajectory pairs, train a Bradley-Terry reward model, and PPO fine-tune with KL penalty. Built with PyTorch + Rich. No GPU required. Runs in ~2 minutes.</p>""",

    # ─── Table of Contents (sidebar) ───
    "toc": [
        {"id": "what-happens",       "label": "What Happens"},
        {"id": "screenshot-01",      "label": "Grid World",             "is_screenshot": True},
        {"id": "screenshot-02",      "label": "Preference Pair",        "is_screenshot": True},
        {"id": "screenshot-03",      "label": "Reward Model",           "is_screenshot": True},
        {"id": "screenshot-04",      "label": "PPO Training",           "is_screenshot": True},
        {"id": "screenshot-05",      "label": "Policy Comparison",      "is_screenshot": True},
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
<p>The game walks you through the four-phase RLHF pipeline, with full visibility into what's happening at every step.</p>

<h3>Phase 1: Pre-training</h3>
<p>The agent learns basic navigation on an <strong>8&times;8 grid</strong> with a three-corridor obstacle layout using a simple hand-coded reward (+10 for reaching the goal, &minus;0.01 per step). This is <em>analogous to LLM pre-training</em> &mdash; the model learns basic competence before being aligned.</p>
<p>The grid contains <strong>collectible pickups</strong> scattered along different routes:</p>
<ul>
    <li><strong>Coins</strong> (<code>+0.5</code> reward each) &mdash; placed on common paths</li>
    <li><strong>Gems</strong> (<code>+2.0</code> reward each) &mdash; placed on detour routes, creating exploration tradeoffs</li>
</ul>
<p><strong>You'll see:</strong></p>
<ul>
    <li>The grid with the agent's evolving path, showing walls, pickups, and collection stats</li>
    <li>Training metrics with sparkline trends (reward, goal rate, steps, losses, entropy)</li>
    <li>After training: a policy arrow map (greedy action at every cell), a value heatmap V(s), and a neural network forward pass visualization</li>
</ul>
"""},

        # Screenshot 1
        {"type": "screenshot", "num": 1, "title": "Grid World Environment",
         "desc": "The 8×8 grid with a demo trajectory after pre-training. Cyan dots show the agent's path from start (S) to goal (G), collecting coins and gems along the way. Stats show steps, turns, unique cells visited, and pickups collected.",
         "src": "01_grid_world.svg"},

        {"type": "note", "id": "phase2-note", "heading": "Phase 2: Human Preference Collection", "html": r"""
<p>Two trajectories are shown side-by-side. You pick which path you prefer (or skip). Repeat 30 times. This is <em>exactly what human annotators do</em> when rating LLM responses.</p>
<p><strong>You'll see:</strong></p>
<ul>
    <li>Side-by-side grids with path statistics (steps, turns, unique cells, pickups collected)</li>
    <li>Your evolving preference patterns (e.g. "you tend to prefer shorter paths, fewer turns")</li>
    <li>A progress bar and running tallies</li>
</ul>
<div class="callout callout-key">
<p><strong>Your choices matter:</strong> The reward model in Phase 3 learns entirely from your preferences. If you prefer scenic gem-collecting routes, the agent will learn to take detours. If you prefer efficiency, it will learn the shortest path.</p>
</div>
"""},

        # Screenshot 2
        {"type": "screenshot", "num": 2, "title": "Human Preference Collection",
         "desc": "A preference pair from Phase 2. Two trajectories are displayed side-by-side with full path statistics. The annotator (you) picks which path is better — this is the same interface used to collect human feedback for LLM alignment.",
         "src": "02_preferences.svg"},

        {"type": "note", "id": "phase3-note", "heading": "Phase 3: Reward Model Training", "html": r"""
<p>A neural network (the reward model) learns to predict your preferences using the <strong>Bradley-Terry model</strong>: <code>P(A &succ; B) = &sigma;(R(A) &minus; R(B))</code>. This is the <em>same math used in RLHF reward models</em> for LLMs.</p>
<p><strong>You'll see:</strong></p>
<ul>
    <li>RM architecture diagram</li>
    <li>Loss/accuracy curves as it learns</li>
    <li>A learned reward heatmap showing what the RM thinks is valuable at each grid cell</li>
    <li>Spot-checks of RM predictions against your actual labels</li>
</ul>
"""},

        # Screenshot 3
        {"type": "screenshot", "num": 3, "title": "Reward Model Training",
         "desc": "The RM architecture, training metrics (loss and accuracy curves with sparklines), a learned reward heatmap r(s) showing which grid cells the RM values most, and a spot-check comparing the RM's preference prediction against the human label.",
         "src": "03_reward_model.svg"},

        {"type": "note", "id": "phase4-note", "heading": "Phase 4: RLHF / PPO Fine-tuning", "html": r"""
<p>PPO optimises the agent's policy against the learned reward model, with a <strong>KL penalty</strong> to stay close to the pre-trained behaviour. This is <em>identical to the RLHF step</em> in LLM training (InstructGPT, ChatGPT, Claude, etc.).</p>
<p><strong>You'll see:</strong></p>
<ul>
    <li>Pre-trained vs RLHF paths side-by-side, updating in real time</li>
    <li>RM score and KL divergence trends</li>
    <li>After training: neural network forward pass with reference comparison, and a policy diff showing exactly which cells changed</li>
</ul>
"""},

        # Screenshot 4
        {"type": "screenshot", "num": 4, "title": "PPO Fine-Tuning with KL Penalty",
         "desc": "The PPO fine-tuning dashboard. Training metrics (RM score, goal rate, KL divergence, entropy) with sparkline trends. A value heatmap V(s) from the RLHF policy, and the greedy policy arrow map showing the learned navigation strategy.",
         "src": "04_ppo_training.svg"},

        {"type": "note", "id": "conclusion-note", "heading": "Conclusion", "html": r"""
<p>Final evaluation comparing pre-trained and RLHF policies across 50 episodes, the <strong>alignment tax</strong> (KL divergence cost), and a table mapping every step to its LLM equivalent.</p>
"""},

        # Screenshot 5
        {"type": "screenshot", "num": 5, "title": "Pre-training vs RLHF Policy",
         "desc": "Three-column policy comparison (pre-trained arrows, RLHF arrows, diff), a results summary table with metrics and percent change, and the full LLM parallel mapping table showing how every game step corresponds to real LLM alignment.",
         "src": "05_comparison.svg"},

        {"type": "break"},

        # ── Playing Around ──
        {"type": "note", "id": "playing-around", "heading": "Playing Around", "html": r"""
<p>Here are ways to experiment and build deeper intuition:</p>

<h3>Changing your preference strategy</h3>
<p>Run <code>python app.py</code> multiple times with different annotation strategies in Phase 2:</p>
<ul>
    <li><strong>Efficiency-only:</strong> Always pick the shorter path. The agent will converge to the BFS-optimal route.</li>
    <li><strong>Gem hunter:</strong> Always pick whichever path collects more gems. Watch the agent learn to take detours to high-value pickups.</li>
    <li><strong>Scenic routes:</strong> Pick paths that visit more unique cells. The agent will learn to explore.</li>
    <li><strong>Random/ties:</strong> Pick randomly or skip everything. The RM will have low accuracy and the RLHF phase will show noisy, unstable behaviour &mdash; a great demonstration of why annotation quality matters.</li>
</ul>

<h3>Tuning hyperparameters</h3>
<p>Edit the config block at the top of <code>app.py</code>:</p>
<pre><code>PRETRAIN_EPISODES = 150      # More episodes = stronger pre-trained baseline
NUM_PREFERENCE_PAIRS = 30    # More pairs = better RM accuracy
RM_EPOCHS = 100              # Reward model training epochs
RLHF_EPISODES = 100          # RLHF fine-tuning episodes
KL_COEFF = 0.2               # KL penalty coefficient (beta)</code></pre>

<h3>Key experiments</h3>
<ul>
    <li>Set <code>KL_COEFF = 0.0</code> &mdash; removes the KL penalty entirely. Watch for <strong>reward hacking</strong>: the agent exploits the RM's blind spots.</li>
    <li>Set <code>KL_COEFF = 2.0</code> &mdash; very strong KL constraint. The agent barely moves from pre-trained behaviour, demonstrating the alignment tax tradeoff.</li>
    <li>Set <code>NUM_PREFERENCE_PAIRS = 5</code> &mdash; very few preferences. The RM trains on minimal data, leading to poor generalisation.</li>
</ul>

<h3>Observing specific phenomena</h3>
<table>
    <tr><th>Phenomenon</th><th>How to trigger</th><th>What to look for</th></tr>
    <tr><td><strong>Reward hacking</strong></td><td>Set <code>KL_COEFF = 0.0</code></td><td>RM score climbs but paths look wrong</td></tr>
    <tr><td><strong>Alignment tax</strong></td><td>Compare final KL at different <code>KL_COEFF</code> values</td><td>Higher beta = lower KL but less adaptation</td></tr>
    <tr><td><strong>Mode collapse</strong></td><td><code>KL_COEFF = 0.0</code> + few preferences</td><td>Agent loops or oscillates</td></tr>
    <tr><td><strong>Exploration failure</strong></td><td><code>PRETRAIN_ENTROPY_COEFF = 0.001</code></td><td>Agent only uses one corridor</td></tr>
    <tr><td><strong>RM inaccuracy</strong></td><td><code>NUM_PREFERENCE_PAIRS = 3</code></td><td>Low RM accuracy, noisy RLHF</td></tr>
</table>
"""},

        {"type": "break"},

        # ── LLM Parallel Mapping ──
        {"type": "note", "id": "llm-parallel", "heading": "LLM Parallel Mapping", "html": r"""
<table>
    <tr><th>What you do in the game</th><th>What happens in LLM RLHF</th></tr>
    <tr><td>Pre-train agent on grid rewards</td><td>Pre-train LLM on internet text (next-token prediction)</td></tr>
    <tr><td>Rate pairs of paths</td><td>Annotators rate pairs of model responses</td></tr>
    <tr><td>Train reward model on preferences</td><td>Train RM on comparison data (Bradley-Terry)</td></tr>
    <tr><td>PPO with KL penalty (&beta;)</td><td>PPO fine-tune LLM with KL penalty against SFT policy</td></tr>
    <tr><td>Agent adopts YOUR path style</td><td>LLM adopts human-preferred response style</td></tr>
    <tr><td>Reward hacking when &beta; = 0</td><td>LLM gaming metrics when KL constraint removed</td></tr>
    <tr><td>Gems on detour routes</td><td>High-quality but costly responses (longer, more detailed)</td></tr>
    <tr><td>Alignment tax (KL cost)</td><td>Quality-diversity tradeoff in aligned models</td></tr>
</table>
"""},

        {"type": "break"},

        # ── Architecture ──
        {"type": "note", "id": "architecture", "heading": "Architecture", "html": r"""
<h3>Grid World (<code>env.py</code>)</h3>
<ul>
    <li>8&times;8 grid with start at <code>(0,0)</code> and goal at <code>(7,7)</code></li>
    <li>Three-corridor wall layout forcing path choices</li>
    <li>Collectible pickups: 7 coins (+0.5 each) and 3 gems (+2.0 each)</li>
    <li>Pickups disappear on collection and reset each episode</li>
    <li>BFS shortest-path computation for reachability checks</li>
</ul>

<h3>Policy Network (<code>policy.py</code>)</h3>
<ul>
    <li>Input: normalised <code>(row/size, col/size)</code> position</li>
    <li>Architecture: <code>Linear(2, 64) &rarr; ReLU &rarr; Linear(64, 64) &rarr; ReLU &rarr; Linear(64, 5)</code></li>
    <li>Output: 4 action logits (N/S/E/W) + 1 state value</li>
    <li>PPO with clipped surrogate, GAE (&lambda;=0.95), entropy bonus</li>
</ul>

<h3>Reward Model (<code>reward_model.py</code>)</h3>
<ul>
    <li>Same input/architecture as policy, but outputs scalar reward per state</li>
    <li>Bradley-Terry loss: <code>P(A &succ; B) = sigmoid(R_A &minus; R_B)</code></li>
    <li>Supports tie labels (label = 0.5)</li>
</ul>

<h3>Preference Database (<code>preferences.py</code>)</h3>
<ul>
    <li>Stores <code>(trajectory_A, trajectory_B, label)</code> tuples</li>
    <li>Analytics: count by preference, pattern detection, summary stats</li>
</ul>

<h3>Training Orchestrator (<code>train.py</code>)</h3>
<ul>
    <li><code>pretrain()</code>: PPO against hand-coded env rewards</li>
    <li><code>train_rm()</code>: Bradley-Terry RM training loop</li>
    <li><code>rlhf_train()</code>: PPO against RM scores + KL penalty</li>
</ul>

<h3>Visualisation (<code>viz.py</code>)</h3>
<ul>
    <li>All rendering via Rich (no external display needed)</li>
    <li>Grid rendering, policy arrows, value/reward heatmaps, sparklines, neural network forward pass diagrams, preference pair display</li>
</ul>
"""},

        {"type": "break"},

        # ── Project Structure ──
        {"type": "note", "id": "project-structure", "heading": "Project Structure", "html": r"""
<pre><code>01-pathfinding-preference-game/
├── app.py              # Main interactive terminal application (544 lines)
├── env.py              # 8×8 grid world with pickups (385 lines)
├── policy.py           # MLP policy + value network with PPO (470 lines)
├── reward_model.py     # Bradley-Terry neural reward model (275 lines)
├── preferences.py      # Preference database with analytics (194 lines)
├── train.py            # Training orchestrator — all 4 phases (386 lines)
├── viz.py              # Rich terminal rendering (589 lines)
├── requirements.txt    # torch, numpy, rich
└── tests/
    ├── test_env.py            # 69 tests
    ├── test_policy.py         # 33 tests
    ├── test_preferences.py    # 27 tests
    ├── test_reward_model.py   # 27 tests
    ├── test_viz.py            # 37 tests
    └── test_integration.py    #  8 tests
                               # 201 tests total</code></pre>
"""},

        # ── Key Concepts ──
        {"type": "note", "id": "key-concepts", "heading": "Key Concepts Demonstrated", "html": r"""
<ul>
    <li><strong>Bradley-Terry preference model</strong> &mdash; the mathematical foundation of RLHF</li>
    <li><strong>PPO clipped surrogate</strong> &mdash; the optimisation algorithm used in ChatGPT, Claude, etc.</li>
    <li><strong>KL divergence penalty</strong> &mdash; prevents reward hacking / mode collapse</li>
    <li><strong>Reward overoptimisation</strong> &mdash; observable when beta is set to zero</li>
    <li><strong>Generalised Advantage Estimation (GAE)</strong> &mdash; variance reduction in policy gradients</li>
    <li><strong>Alignment tax</strong> &mdash; the KL divergence cost of adapting to human preferences</li>
    <li><strong>Exploration vs exploitation</strong> &mdash; entropy bonus encourages corridor discovery</li>
</ul>
"""},

        {"type": "break"},

        # ── Quick Start ──
        {"type": "note", "id": "quick-start", "heading": "Quick Start", "html": r"""
<pre><code># From the repo root
cd coding-adventures/01-pathfinding-preference-game

# Install dependencies (if not using the repo venv)
pip install -r requirements.txt

# Run the game
python app.py</code></pre>
<p>You'll be guided through all four phases interactively. Press Enter to advance between sections.</p>

<h3>Running Tests</h3>
<pre><code># All 201 tests (~9 seconds on CPU)
python -m pytest tests/ -v

# Quick smoke test
python -m pytest tests/ -x -q</code></pre>
"""},

    ],  # end content

    # ─── Related Papers ───
    "related_papers": [
        {
            "arxiv_id": "1706.03741",
            "title": "Deep Reinforcement Learning from Human Preferences",
            "desc": "Christiano et al. (2017) — The foundational RLHF paper. Train RL agents using human preference comparisons.",
            "url": "../../papers/1706.03741/",
        },
        {
            "arxiv_id": "1707.06347",
            "title": "Proximal Policy Optimization Algorithms",
            "desc": "Schulman et al. (2017) — PPO, the RL optimizer used inside RLHF.",
            "url": "../../papers/1707.06347/",
        },
        {
            "arxiv_id": "2009.01325",
            "title": "Learning to Summarize from Human Feedback",
            "desc": "Stiennon et al. (2020) — Applies RLHF to text summarization at scale.",
            "url": "../../papers/2009.01325/",
        },
    ],
}
