"""Page configuration for 1707.06347 — Proximal Policy Optimization Algorithms."""

PAGE_DATA = {
    "title": "Proximal Policy Optimization Algorithms",
    "arxiv_id": "1707.06347",
    "authors": "Schulman, Wolski, Dhariwal, Radford, Klimov",
    "year": 2017,
    "last_updated": "May 2026",

    "tldr": """<p>PPO proposes a family of policy gradient methods that use a <strong>clipped surrogate objective</strong> to constrain policy updates, achieving the stability of trust-region methods (TRPO) with the simplicity of first-order optimization. Instead of solving a constrained optimization problem with second-order methods, PPO clips the importance sampling ratio to $[1-\\epsilon, 1+\\epsilon]$, preventing destructively large updates. Combined with Generalized Advantage Estimation and multiple epochs of minibatch updates, PPO delivers strong performance on continuous control (MuJoCo) and discrete (Atari) tasks. Published by OpenAI in 2017, PPO became the default RL optimizer for LLM alignment via RLHF &mdash; powering InstructGPT and ChatGPT.</p>""",

    # ─── Table of Contents (sidebar) ───
    "toc": [
        {"id": "abstract",          "label": "Abstract",              "is_viz": False},
        {"id": "introduction",      "label": "Introduction",          "is_viz": False},
        {"id": "policy-gradients",  "label": "Policy Gradient Methods","is_viz": False},
        {"id": "viz-01",            "label": "Statistics Refresher",  "is_viz": True},
        {"id": "viz-02",            "label": "Policy Gradient Intuition","is_viz": True},
        {"id": "derivation",        "label": "PG Derivation",        "is_viz": False},
        {"id": "viz-03",            "label": "PG Derivation Animation","is_viz": True},
        {"id": "advantage",         "label": "Advantage Function",    "is_viz": False},
        {"id": "viz-04",            "label": "Advantage vs Raw Reward","is_viz": True},
        {"id": "gae",               "label": "GAE",                   "is_viz": False},
        {"id": "viz-05",            "label": "GAE Lambda Tradeoff",   "is_viz": True},
        {"id": "trust-regions",     "label": "Trust Region Methods",  "is_viz": False},
        {"id": "viz-06",            "label": "Trust Region Motivation","is_viz": True},
        {"id": "kl-divergence",     "label": "KL & TRPO",            "is_viz": False},
        {"id": "viz-07",            "label": "KL Trust Region",       "is_viz": True},
        {"id": "clipped-objective", "label": "Clipped Surrogate",     "is_viz": False},
        {"id": "viz-08",            "label": "Clipped Surrogate Viz", "is_viz": True},
        {"id": "algorithm",         "label": "Algorithm",             "is_viz": False},
        {"id": "viz-09",            "label": "PPO vs TRPO vs PG",    "is_viz": True},
        {"id": "experiments",       "label": "Experiments",           "is_viz": False},
        {"id": "llm-alignment",     "label": "PPO for LLMs",         "is_viz": False},
        {"id": "viz-10",            "label": "PPO LLM Alignment",    "is_viz": True},
        {"id": "viz-11",            "label": "CartPole Demo",         "is_viz": True},
        {"id": "conclusion",        "label": "Conclusion",            "is_viz": False},
    ],

    # ─── Nav bar (mobile, abbreviated) ───
    "nav": [
        {"id": "abstract",          "label": "Abstract"},
        {"id": "introduction",      "label": "Intro"},
        {"id": "policy-gradients",  "label": "Policy Gradients"},
        {"id": "derivation",        "label": "Derivation"},
        {"id": "advantage",         "label": "Advantage"},
        {"id": "gae",               "label": "GAE"},
        {"id": "trust-regions",     "label": "Trust Regions"},
        {"id": "clipped-objective", "label": "Clipping"},
        {"id": "algorithm",         "label": "Algorithm"},
        {"id": "experiments",       "label": "Experiments"},
        {"id": "llm-alignment",     "label": "LLMs"},
        {"id": "viz-11",            "label": "Demo"},
    ],

    # ─── Viz gallery cards ───
    "viz_gallery": [
        {"num": 1,  "title": "Statistics Refresher",       "tool": "manim", "tag_label": "Manim"},
        {"num": 2,  "title": "Policy Gradient Intuition",  "tool": "manim", "tag_label": "Manim"},
        {"num": 3,  "title": "PG Derivation",              "tool": "manim", "tag_label": "Manim"},
        {"num": 4,  "title": "Advantage vs Raw Reward",    "tool": "multi", "tag_label": "MPL+Plotly"},
        {"num": 5,  "title": "GAE Lambda Tradeoff",        "tool": "multi", "tag_label": "MPL+Plotly"},
        {"num": 6,  "title": "Trust Region Motivation",    "tool": "manim", "tag_label": "Manim"},
        {"num": 7,  "title": "KL Trust Region",            "tool": "multi", "tag_label": "MPL+Plotly"},
        {"num": 8,  "title": "Clipped Surrogate",          "tool": "multi", "tag_label": "MPL+Plotly"},
        {"num": 9,  "title": "PPO vs TRPO vs PG",          "tool": "manim", "tag_label": "Manim"},
        {"num": 10, "title": "PPO for LLM Alignment",      "tool": "multi", "tag_label": "MPL+Plotly"},
        {"num": 11, "title": "CartPole Demo",              "tool": "plotly", "tag_label": "Plotly"},
    ],

    # ─── Content blocks (ordered) ───
    "content": [

        # ══════════════════════════════════════════
        # ABSTRACT
        # ══════════════════════════════════════════
        {"type": "note", "id": "abstract", "heading": "Abstract", "html": """
            <p>The paper proposes a new family of policy gradient methods that alternate between <em>sampling data through interaction with the environment</em> and <em>optimizing a "surrogate" objective function using stochastic gradient ascent</em>.</p>
            <ul>
                <li>Gradient <strong>ascent</strong> is used since we want to maximize the reward (as opposed to gradient descent, which minimizes a loss).</li>
                <li>In RL, we try to maximize the expected cumulative reward.</li>
                <li>It's <strong>stochastic</strong> because you can't calculate the cumulative reward over all possible trajectories &mdash; only the sampled ones from the mini-batch.</li>
            </ul>
        """},

        # ══════════════════════════════════════════
        # INTRODUCTION
        # ══════════════════════════════════════════
        {"type": "note", "id": "introduction", "heading": "Introduction", "html": """
            <p>We want a method that is:</p>
            <ul>
                <li><strong>Scalable</strong> &mdash; to large models and parallel implementations</li>
                <li><strong>Data Efficient</strong></li>
                <li><strong>Robust</strong> &mdash; works on many problems without hyperparameter tuning</li>
            </ul>
            <p>The landscape of existing approaches:</p>
            <ul>
                <li><strong>Q-learning</strong> with function approximation fails on many simple problems and is poorly understood.</li>
                <li><strong>TRPO</strong> is complicated &mdash; requires second-order optimization, conjugate gradients, and line search.</li>
                <li><strong>Vanilla policy gradient</strong> methods are not data efficient or robust.</li>
            </ul>

            <div class="callout callout-key">
                <p>PPO achieves the stability of trust-region methods with the simplicity of first-order optimization. The proposed algorithm performs better than existing approaches on continuous control tasks.</p>
            </div>

            <h3>Two Approaches to Learning a Policy</h3>
            <ol>
                <li><strong>Value-function based</strong>: Learn a value function that estimates "how good is each action in this state" and choose the action with the highest value. Only works well with discrete actions. Examples: Q-learning, DQN.</li>
                <li><strong>Policy gradient</strong>: Parametrize the policy as a neural network that takes the state as input and outputs a probability distribution over actions. Use gradient ascent to adjust parameters so the policy produces actions that lead to higher rewards. Examples: REINFORCE, PPO, TRPO.</li>
            </ol>
            <p>The distinction: <em>"Which action is the best action?"</em> (value function) vs <em>"For this state, what should I do?"</em> (policy gradient). However, policy gradient methods have <strong>high variance</strong> &mdash; you might get lucky, or not.</p>
        """},

        {"type": "break"},

        # ══════════════════════════════════════════
        # POLICY GRADIENT METHODS
        # ══════════════════════════════════════════
        {"type": "note", "id": "policy-gradients", "heading": "Background: Policy Gradient Methods", "html": """
            <p>Policy gradient methods work by estimating the gradient of expected reward with respect to the policy parameters, then using stochastic gradient ascent to improve.</p>

            <p>The most commonly used gradient estimator has the form:</p>
            <div class="math-block">
                $$ \\widehat{g} = \\widehat{\\mathbb{E}}_{t} \\left[\\nabla_{\\theta}\\log\\pi_{\\theta}(a_t|s_t)\\,\\widehat{A}_t\\right]$$
            </div>
            <p>where $\\pi_\\theta$ is a stochastic policy, $\\widehat{A}_t$ is the estimator of the advantage function at timestep $t$, and $\\widehat{\\mathbb{E}}_t$ denotes an empirical average over a finite batch of samples.</p>

            <p>In practice, automatic differentiation software constructs an objective function whose gradient <em>is</em> the policy gradient estimator:</p>
            <div class="math-block">
                $$ L^{PG}(\\theta) = \\widehat{\\mathbb{E}}_t \\left[\\log\\pi_{\\theta}(a_t|s_t)\\,\\widehat{A}_t\\right]$$
            </div>

            <div class="callout callout-key">
                <p>Performing multiple steps of optimization on $L^{PG}$ using the same trajectory can lead to <strong>destructively large policy updates</strong>. This is the core problem PPO solves.</p>
            </div>

            <h3>Intuition for the Method</h3>
            <p>You have a policy parametrized by a neural network ($\\theta$). The policy maps states to action probabilities. You want to adjust $\\theta$ so the policy collects more rewards.</p>
            <p>The tricky part: the objective has an expectation over trajectories that depend on $\\theta$ in a complex way:</p>
            <div class="math-block">
                $$ \\theta \\rightarrow \\text{policy} \\rightarrow \\text{states visited} \\rightarrow \\text{rewards} $$
            </div>
            <p>How do you differentiate through this? Using the <strong>log-derivative trick</strong>:</p>
            <div class="math-block">
                $$\\nabla P(x) = P(x) \\cdot \\nabla \\log P(x)$$
            </div>
            <p>This lets us rewrite the gradient of an expectation as an expectation we can sample:</p>
            <div class="math-block">
                $$\\nabla\\bigl(\\mathbb{E}[\\text{reward}]\\bigr) = \\mathbb{E}\\bigl[\\text{reward} \\cdot \\nabla \\log \\pi(a|s)\\bigr]$$
            </div>
            <p>The process becomes: run policy &rarr; collect trajectories &rarr; for each (state, action) pair compute reward &times; gradient &rarr; average across samples. This is the <strong>REINFORCE</strong> algorithm.</p>
        """},

        # Viz 1: Statistics Refresher
        {"type": "viz", "id": "viz-01", "num": 1, "title": "Statistics Refresher",
         "tool": "manim",
         "desc": "Animated review of key statistical concepts underpinning policy gradients: expectation, variance, log-derivative trick, and importance sampling.",
         "video": "StatsRefresher.mp4"},

        # Viz 2: Policy Gradient Intuition
        {"type": "viz", "id": "viz-02", "num": 2, "title": "Policy Gradient Intuition",
         "tool": "manim",
         "desc": "Visual walkthrough of the REINFORCE algorithm: how actions with positive advantage get reinforced (probability increased) while actions with negative advantage get suppressed.",
         "video": "PolicyGradientIntuition.mp4"},

        {"type": "break"},

        # ══════════════════════════════════════════
        # DERIVATION
        # ══════════════════════════════════════════
        {"type": "note", "id": "derivation", "heading": "Derivation of the Policy Gradient Estimator", "html": """
            <h3>Setup</h3>
            <p>We have a parameterized policy $\\pi_{\\theta}(a_t|s_t)$ and want to maximize the expected return:</p>
            <div class="math-block">
                $$J(\\theta) = \\mathbb{E}_{\\tau \\sim \\pi_{\\theta}} \\left[ \\sum_{t=0}^{T} r_t \\right]$$
            </div>
            <p>where $\\tau = (s_0, a_0, s_1, a_1, \\ldots)$ is a trajectory sampled under policy $\\pi_{\\theta}$.</p>

            <h4>Step 1: Probability of a trajectory</h4>
            <div class="math-block">
                $$P(\\tau | \\theta) = \\rho(s_0) \\prod_{t=0}^{T} \\pi_{\\theta}(a_t|s_t) \\cdot P(s_{t+1}|s_t, a_t)$$
            </div>
            <p>where $\\rho(s_0)$ is the initial state distribution and $P(s_{t+1}|s_t,a_t)$ is the environment dynamics.</p>

            <h4>Step 2: Gradient of the objective</h4>
            <div class="math-block">
                $$\\nabla_{\\theta} J(\\theta) = \\nabla_{\\theta} \\int P(\\tau|\\theta) \\, R(\\tau) \\, d\\tau$$
            </div>

            <h4>Step 3: The log-derivative trick</h4>
            <p>Using $\\nabla_{\\theta} P(\\tau|\\theta) = P(\\tau|\\theta) \\cdot \\nabla_{\\theta} \\log P(\\tau|\\theta)$:</p>
            <div class="math-block">
                $$\\nabla_{\\theta} J(\\theta) = \\mathbb{E}_{\\tau \\sim \\pi_{\\theta}} \\left[ \\nabla_{\\theta} \\log P(\\tau|\\theta) \\cdot R(\\tau) \\right]$$
            </div>

            <h4>Step 4: Simplify the log-trajectory probability</h4>
            <div class="math-block">
                $$\\log P(\\tau|\\theta) = \\log \\rho(s_0) + \\sum_{t=0}^{T} \\log \\pi_{\\theta}(a_t|s_t) + \\sum_{t=0}^{T} \\log P(s_{t+1}|s_t,a_t)$$
            </div>
            <p>Taking the gradient w.r.t. $\\theta$:</p>
            <div class="math-block">
                $$\\nabla_{\\theta} \\log P(\\tau|\\theta) = \\sum_{t=0}^{T} \\nabla_{\\theta} \\log \\pi_{\\theta}(a_t|s_t)$$
            </div>

            <div class="callout callout-insight">
                <p>The initial state distribution and environment dynamics terms vanish &mdash; they don't depend on $\\theta$. This is why policy gradients are <strong>model-free</strong>: we never need to know or differentiate through the environment dynamics.</p>
            </div>

            <h4>Step 5: The REINFORCE gradient</h4>
            <div class="math-block">
                $$\\nabla_{\\theta} J(\\theta) = \\mathbb{E}_{\\tau \\sim \\pi_{\\theta}} \\left[ \\sum_{t=0}^{T} \\nabla_{\\theta} \\log \\pi_{\\theta}(a_t|s_t) \\cdot R(\\tau) \\right]$$
            </div>

            <h4>Step 6: Introduce the advantage function</h4>
            <p>Using $R(\\tau)$ produces high variance. Replace it with $\\widehat{A}_t = Q(s_t, a_t) - V(s_t)$ without changing the expected gradient:</p>
            <div class="math-block">
                $$\\nabla_{\\theta} J(\\theta) = \\mathbb{E}_{t} \\left[ \\nabla_{\\theta} \\log \\pi_{\\theta}(a_t|s_t) \\cdot \\widehat{A}_t \\right]$$
            </div>

            <h4>Step 7: The policy gradient estimator</h4>
            <p>Estimate this expectation by sampling and averaging:</p>
            <div class="formal-def">
                <div class="math-block">
                    $$\\boxed{\\widehat{g} = \\widehat{\\mathbb{E}}_{t} \\left[ \\nabla_{\\theta} \\log \\pi_{\\theta}(a_t|s_t) \\, \\widehat{A}_t \\right]}$$
                </div>
                <p>where $\\widehat{\\mathbb{E}}_t$ denotes the empirical average over a finite batch of timesteps, and $\\widehat{A}_t$ is an estimator of the advantage function (computed using GAE in PPO's case).</p>
            </div>
        """},

        # Viz 3: Policy Gradient Derivation
        {"type": "viz", "id": "viz-03", "num": 3, "title": "Policy Gradient Derivation",
         "tool": "manim",
         "desc": "Step-by-step animated derivation of the policy gradient estimator, from trajectory probability through the log-derivative trick to the final REINFORCE formula.",
         "video": "PolicyGradientDerivation.mp4"},

        {"type": "break"},

        # ══════════════════════════════════════════
        # ADVANTAGE FUNCTION
        # ══════════════════════════════════════════
        {"type": "note", "id": "advantage", "heading": "The Advantage Function", "html": """
            <h3>Building Blocks</h3>
            <p><strong>Value function</strong> $V(s_t)$ &mdash; "If I'm in state $s_t$ and follow my policy from here, what's my expected total future reward?"</p>
            <div class="math-block">
                $$V(s_t) = \\mathbb{E}_{\\pi} \\left[ \\sum_{t'=t}^{T} r_{t'} \\;\\middle|\\; s_t \\right]$$
            </div>

            <p><strong>Action-value function</strong> $Q(s_t, a_t)$ &mdash; "If I take <em>specific</em> action $a_t$ in state $s_t$, then follow my policy after that?"</p>
            <div class="math-block">
                $$Q(s_t, a_t) = \\mathbb{E}_{\\pi} \\left[ \\sum_{t'=t}^{T} r_{t'} \\;\\middle|\\; s_t, a_t \\right]$$
            </div>

            <p>Note that $V$ is just $Q$ averaged over all actions weighted by the policy: $V(s_t) = \\sum_{a} \\pi(a|s_t) \\cdot Q(s_t, a)$.</p>

            <h3>Definition</h3>
            <div class="formal-def">
                <div class="math-block">
                    $$A(s_t, a_t) = Q(s_t, a_t) - V(s_t)$$
                </div>
                <p>"How much better (or worse) is this specific action compared to what I'd get on average from this state?"</p>
                <ul>
                    <li>$A > 0$: this action is better than the average action from this state</li>
                    <li>$A < 0$: this action is worse than average</li>
                    <li>$A = 0$: this action is exactly as good as average</li>
                </ul>
            </div>

            <h3>Why This Definition? &mdash; Variance Reduction</h3>
            <p>The "something" multiplying $\\nabla_\\theta \\log \\pi_\\theta(a_t|s_t)$ in the policy gradient can be:</p>
            <ol>
                <li><strong>Total trajectory reward</strong> $R(\\tau)$ &mdash; valid but high variance</li>
                <li><strong>Future reward from $t$ onward</strong> $\\sum_{t' \\geq t} r_{t'}$ &mdash; valid, lower variance</li>
                <li><strong>$Q(s_t, a_t)$</strong> &mdash; same expected gradient, focuses on the action's value</li>
                <li><strong>$Q(s_t, a_t) - V(s_t)$</strong> &mdash; same expected gradient, <em>lowest variance</em></li>
            </ol>

            <div class="callout callout-insight">
                <p>Rewards from the past can't influence future rewards, so past rewards just add variance. Looking at only the "rewards to go" ($Q$ from time $t$) removes that noise. Adding a state-dependent baseline ($V$) reduces variance even further &mdash; without introducing any bias.</p>
            </div>

            <h4>Proof that subtracting $V(s_t)$ adds zero in expectation</h4>
            <div class="math-block">
                $$\\mathbb{E}_{a \\sim \\pi} \\left[ \\nabla_\\theta \\log \\pi_\\theta(a|s) \\cdot V(s) \\right]
                = V(s) \\cdot \\sum_{a} \\nabla_\\theta \\pi_\\theta(a|s)
                = V(s) \\cdot \\nabla_\\theta \\underbrace{\\sum_{a} \\pi_\\theta(a|s)}_{= 1}
                = 0$$
            </div>
            <p>Therefore subtracting $V(s_t)$ adds zero in expectation but dramatically reduces variance. <strong>The advantage is the optimal baseline for variance reduction in policy gradients.</strong></p>

            <h3>Summary</h3>
            <p>By using gradient ascent with the advantage, we push up actions that have higher-than-average returns and push down those with lower-than-average returns. The advantage acts as both a <em>filter</em> and a <em>multiplier</em>.</p>
        """},

        # Viz 4: Advantage vs Raw Reward
        {"type": "viz", "id": "viz-04", "num": 4, "title": "Advantage vs Raw Reward",
         "tool": "multi",
         "desc": "Side-by-side comparison of policy gradient updates using raw trajectory rewards vs advantage-based estimation, showing how variance is reduced.",
         "static_img": "04_advantage_vs_raw_reward.png",
         "interactive": "04_advantage_vs_raw_reward.html"},

        {"type": "break"},

        # ══════════════════════════════════════════
        # GAE
        # ══════════════════════════════════════════
        {"type": "note", "id": "gae", "heading": "Generalized Advantage Estimation (GAE)", "html": """
            <p>The $\\widehat{A}_t$ in the PPO equation means "we don't know the true $A$, so we estimate it." PPO uses <strong>GAE</strong>, which introduces a parameter $\\lambda$ that trades off bias and variance.</p>

            <h3>TD Residual</h3>
            <p>First, define the <strong>temporal difference (TD) residual</strong> at timestep $t$:</p>
            <div class="formal-def">
                <div class="math-block">
                    $$\\delta_t^V = r_t + \\gamma V(s_{t+1}) - V(s_t)$$
                </div>
                <p>A one-step estimate of the advantage: "I got reward $r_t$, ended up in a state worth $V(s_{t+1})$, so the discounted value of what actually happened is $r_t + \\gamma V(s_{t+1})$. Subtract what I expected, $V(s_t)$, to get the surprise."</p>
            </div>

            <h3>The GAE Formula</h3>
            <div class="math-block">
                $$\\widehat{A}_t^{\\text{GAE}(\\gamma, \\lambda)} = \\sum_{l=0}^{\\infty} (\\gamma \\lambda)^l \\, \\delta_{t+l}^V = \\delta_t^V + (\\gamma \\lambda) \\delta_{t+1}^V + (\\gamma \\lambda)^2 \\delta_{t+2}^V + \\cdots$$
            </div>
            <p>where $\\gamma \\in [0, 1]$ is the discount factor and $\\lambda \\in [0, 1]$ controls the bias-variance tradeoff.</p>

            <h3>Special Cases</h3>
            <h4>When $\\lambda = 0$</h4>
            <div class="math-block">
                $$\\widehat{A}_t = \\delta_t^V = r_t + \\gamma V(s_{t+1}) - V(s_t)$$
            </div>
            <p>Just the one-step TD residual. <strong>Low variance</strong> (only depends on one step of randomness) but <strong>high bias</strong> (relies entirely on $V$ being accurate).</p>

            <h4>When $\\lambda = 1$</h4>
            <div class="math-block">
                $$\\widehat{A}_t = -V(s_t) + \\sum_{l=0}^{\\infty} \\gamma^l r_{t+l}$$
            </div>
            <p>The full Monte Carlo return minus the baseline. <strong>No bias</strong> (uses actual rewards) but <strong>high variance</strong> (depends on the entire future trajectory).</p>

            <div class="callout callout-key">
                <p>$\\lambda$ interpolates smoothly between these extremes. Instead of choosing between "trust your value function estimate" ($\\lambda=0$) and "trust the raw Monte Carlo returns" ($\\lambda=1$), GAE blends them. The $(\\gamma\\lambda)^l$ weighting means nearby TD residuals matter more and distant ones are exponentially downweighted. In practice, PPO typically uses <strong>$\\lambda = 0.95$</strong>.</p>
            </div>

            <div class="callout callout-note">
                <p><strong>LLM connection:</strong> When PPO is used for LLM fine-tuning, each token generation is an action. Tokens with positive advantage (better than average continuations) get reinforced &mdash; made more likely &mdash; while tokens with negative advantage get suppressed.</p>
            </div>
        """},

        # Viz 5: GAE Lambda Tradeoff
        {"type": "viz", "id": "viz-05", "num": 5, "title": "GAE Lambda Tradeoff",
         "tool": "multi",
         "desc": "Interactive visualization of how the GAE lambda parameter controls the bias-variance tradeoff in advantage estimation, with TD weight decay curves for different lambda values.",
         "static_img": "05_gae_lambda_tradeoff.png",
         "interactive": "05_gae_lambda_tradeoff.html"},

        {"type": "break"},

        # ══════════════════════════════════════════
        # TRUST REGION METHODS
        # ══════════════════════════════════════════
        {"type": "note", "id": "trust-regions", "heading": "Trust Region Methods", "html": """
            <p>In TRPO, the objective function ("surrogate objective") is maximized subject to a constraint on the size of the policy update.</p>

            <h3>The Fundamental Problem with Policy Gradients</h3>
            <p>The policy gradient gives us a <em>direction</em> to move $\\theta$, but says nothing about <em>how far</em> to move:</p>
            <div class="math-block">
                $$\\theta_{\\text{new}} = \\theta_{\\text{old}} + \\alpha \\, \\widehat{g}$$
            </div>
            <p>The learning rate $\\alpha$ is the problem. This is much worse than in supervised learning, for a reason specific to RL.</p>

            <h3>Why RL Is Different From Supervised Learning</h3>
            <p>In supervised learning, your dataset is <strong>fixed</strong>. A bad gradient step makes the loss go up, but next step you compute a gradient on the same data and recover.</p>
            <p>In RL, your dataset is <strong>generated by your policy</strong>. A bad gradient step changes your policy, which changes what trajectories you collect, which changes the quality of your next gradient estimate.</p>

            <div class="callout callout-key">
                <p>This creates a <strong>feedback loop</strong>: bad update &rarr; bad policy &rarr; bad data &rarr; bad gradient &rarr; worse update &rarr; &hellip; A single overly large update can be <strong>unrecoverable</strong>. The policy enters a region of parameter space where it collects useless data, and no amount of subsequent gradient steps can fix it.</p>
            </div>

            <h3>The Step Size Dilemma</h3>
            <ul>
                <li><strong>Small $\\alpha$</strong>: safe but slow &mdash; you might need millions of updates.</li>
                <li><strong>Large $\\alpha$</strong>: fast but dangerous &mdash; risk the death spiral above.</li>
                <li><strong>Adaptive methods</strong> (Adam, RMSProp) don't solve this. They adapt step sizes per-parameter, but have no notion of "how much did the <em>policy</em> change?"</li>
            </ul>

            <div class="callout callout-insight">
                <p><strong>The core insight:</strong> we don't care how much $\\theta$ changes in Euclidean space. We care how much the <em>policy distribution</em> changes. A step of size 0.01 in $\\theta$ might barely change the policy, or it might completely flip the action probabilities &mdash; depending on the local geometry.</p>
            </div>
        """},

        # Viz 6: Trust Region Motivation
        {"type": "viz", "id": "viz-06", "num": 6, "title": "Trust Region Motivation",
         "tool": "manim",
         "desc": "Animated comparison of parameter-space vs policy-space distances, showing how a small parameter change can cause a large policy shift and why KL-based trust regions are needed.",
         "video": "TrustRegionMotivation.mp4"},

        {"type": "break"},

        # ══════════════════════════════════════════
        # KL DIVERGENCE & TRPO
        # ══════════════════════════════════════════
        {"type": "note", "id": "kl-divergence", "heading": "From Parameter Space to Distribution Space", "html": """
            <p>We need a measure of distance between <em>policies</em>, not between parameter vectors. This is where <strong>KL divergence</strong> comes in:</p>
            <div class="formal-def">
                <div class="math-block">
                    $$\\text{KL}\\left[\\pi_{\\theta_{\\text{old}}}(\\cdot|s) \\;\\|\\; \\pi_\\theta(\\cdot|s)\\right] = \\sum_a \\pi_{\\theta_{\\text{old}}}(a|s) \\log \\frac{\\pi_{\\theta_{\\text{old}}}(a|s)}{\\pi_\\theta(a|s)}$$
                </div>
                <p>KL divergence measures how different two probability distributions are. Always $\\geq 0$, equals $0$ only when the distributions are identical. It is <strong>invariant to parametrization</strong> &mdash; even if the policy's parameters $\\theta$ have changed a lot, it doesn't matter if the output distribution is similar.</p>
            </div>

            <h3>The Local Approximation (Surrogate Objective)</h3>
            <p>If $\\pi_\\theta$ is close to $\\pi_{\\theta_{\\text{old}}}$, we can approximate the true objective by using importance sampling:</p>
            <div class="math-block">
                $$L^{CPI}(\\theta) = \\widehat{\\mathbb{E}}_t \\left[ \\frac{\\pi_\\theta(a_t|s_t)}{\\pi_{\\theta_{\\text{old}}}(a_t|s_t)} \\widehat{A}_t \\right]$$
            </div>
            <p>This surrogate is a good approximation <strong>only locally</strong>. Far from $\\theta_{\\text{old}}$, it breaks down because the state distribution diverges and the importance sampling ratio becomes unreliable.</p>

            <h3>TRPO: Constrained Optimization</h3>
            <p>TRPO formalizes "stay close" as:</p>
            <div class="formal-def">
                <div class="math-block">
                    $$\\underset{\\theta}{\\text{maximize}} \\quad \\widehat{\\mathbb{E}}_t \\left[ \\frac{\\pi_\\theta(a_t|s_t)}{\\pi_{\\theta_{\\text{old}}}(a_t|s_t)} \\widehat{A}_t \\right]$$
                    $$\\text{subject to} \\quad \\widehat{\\mathbb{E}}_t \\left[ \\text{KL}\\left[\\pi_{\\theta_{\\text{old}}}(\\cdot|s_t) \\;\\|\\; \\pi_\\theta(\\cdot|s_t)\\right] \\right] \\leq \\delta$$
                </div>
                <p>The constraint defines a <strong>trust region</strong> &mdash; the set of all $\\theta$ values where the KL divergence from the old policy is at most $\\delta$. With a <strong>monotonic improvement guarantee</strong>.</p>
            </div>

            <h3>The Cost of TRPO</h3>
            <p>Solving this constrained problem requires:</p>
            <ol>
                <li>Computing the <strong>Fisher information matrix</strong> $F$ (second-order derivative of KL w.r.t. $\\theta$)</li>
                <li>Using <strong>conjugate gradient</strong> to approximately solve $F^{-1} g$</li>
                <li>A <strong>line search</strong> to find the largest step satisfying the KL constraint</li>
            </ol>
            <p>This is doable but expensive and complex. This complexity is precisely what motivates PPO.</p>
        """},

        # Viz 7: KL Trust Region
        {"type": "viz", "id": "viz-07", "num": 7, "title": "KL Trust Region",
         "tool": "multi",
         "desc": "Visualization of the KL divergence constraint defining the trust region in policy space, showing how different constraint radii affect the allowed policy updates.",
         "static_img": "07_kl_trust_region.png",
         "interactive": "07_kl_trust_region.html"},

        {"type": "break"},

        # ══════════════════════════════════════════
        # CLIPPED SURROGATE OBJECTIVE
        # ══════════════════════════════════════════
        {"type": "note", "id": "clipped-objective", "heading": "Clipped Surrogate Objective", "html": """
            <p>Define the probability ratio:</p>
            <div class="math-block">
                $$r_t(\\theta) = \\frac{\\pi_\\theta(a_t|s_t)}{\\pi_{\\theta_{\\text{old}}}(a_t|s_t)} \\qquad \\text{so } r_t(\\theta_{\\text{old}}) = 1$$
            </div>

            <p>Without a constraint, maximizing $L^{CPI}(\\theta) = \\widehat{\\mathbb{E}}_t [r_t(\\theta)\\widehat{A}_t]$ could lead to excessively large policy updates. PPO modifies the objective to penalize changes that move $r_t(\\theta)$ away from 1:</p>

            <div class="formal-def">
                <div class="math-block">
                    $$ L^{\\text{CLIP}}(\\theta) = \\widehat{\\mathbb{E}}_t\\left[\\min\\bigl(r_t(\\theta)\\widehat{A}_t,\\;\\text{clip}(r_t(\\theta),\\,1-\\epsilon,\\,1+\\epsilon)\\,\\widehat{A}_t\\bigr)\\right]$$
                </div>
                <p>where $\\epsilon$ is a hyperparameter (typically $0.2$). By taking the <strong>minimum</strong> of the clipped and unclipped objective, the final objective is a pessimistic (lower) bound on the unclipped objective.</p>
            </div>

            <p>The motivation:</p>
            <ul>
                <li>The first term is the same as TRPO's surrogate.</li>
                <li>The second term clips the ratio, removing the incentive to move $r_t(\\theta)$ outside $[1-\\epsilon, 1+\\epsilon]$.</li>
                <li>Where $r = 1$, both terms are equal. They diverge as $\\theta$ moves away from $\\theta_{\\text{old}}$.</li>
                <li>The $\\min$ handles both positive and negative advantages correctly.</li>
            </ul>

            <div class="callout callout-key">
                <p>When the advantage is <strong>positive</strong> ($A > 0$): the objective increases as $r$ increases (reinforcing the action), but clips at $1+\\epsilon$ &mdash; preventing excessive reinforcement. When the advantage is <strong>negative</strong> ($A < 0$): the objective increases as $r$ decreases (suppressing the action), but clips at $1-\\epsilon$ &mdash; preventing excessive suppression.</p>
            </div>

            <h3>Adaptive KL Penalty (Alternative)</h3>
            <p>As an alternative to clipping, PPO can use an adaptive KL penalty:</p>
            <div class="math-block">
                $$L^{KLPEN}(\\theta) = \\widehat{\\mathbb{E}}_t \\left[ \\frac{\\pi_\\theta(a_t|s_t)}{\\pi_{\\theta_{\\text{old}}}(a_t|s_t)} \\widehat{A}_t - \\beta \\, \\text{KL}\\left[\\pi_{\\theta_{\\text{old}}}(\\cdot|s_t) \\;\\|\\; \\pi_\\theta(\\cdot|s_t)\\right] \\right]$$
            </div>
            <p>The coefficient $\\beta$ is adjusted dynamically:</p>
            <ul>
                <li>If $d < d_{\\text{targ}} / 1.5$: update was too conservative, decrease $\\beta \\leftarrow \\beta / 2$</li>
                <li>If $d > d_{\\text{targ}} \\times 1.5$: update was too aggressive, increase $\\beta \\leftarrow \\beta \\times 2$</li>
            </ul>
            <p>In practice, this worked worse than the clipped surrogate objective.</p>

            <h3>The Progression</h3>
            <table class="progression-table">
                <thead>
                    <tr><th>Method</th><th>How it controls step size</th><th>Complexity</th></tr>
                </thead>
                <tbody>
                    <tr><td>Vanilla PG</td><td>Fixed learning rate $\\alpha$</td><td>Simple, unstable</td></tr>
                    <tr><td>TRPO</td><td>Hard KL constraint, second-order optimization</td><td>Stable, complex</td></tr>
                    <tr><td>PPO (clip)</td><td>Clip importance ratio to $[1-\\epsilon, 1+\\epsilon]$</td><td>Stable, simple</td></tr>
                    <tr><td>PPO (penalty)</td><td>Adaptive KL penalty in objective</td><td>Stable, simple</td></tr>
                </tbody>
            </table>

            <div class="callout callout-insight">
                <p>The narrative arc: gradient direction is easy, step size is hard &rarr; step size in parameter space is the wrong thing to control &rarr; control distance in policy space instead (KL) &rarr; TRPO does this exactly but is complex &rarr; <strong>PPO approximates it simply</strong>.</p>
            </div>
        """},

        # Viz 8: Clipped Surrogate
        {"type": "viz", "id": "viz-08", "num": 8, "title": "Clipped Surrogate Objective",
         "tool": "multi",
         "desc": "Visualization of the PPO clipping mechanism: how the surrogate objective is clipped for both positive and negative advantages, with the effective loss landscape.",
         "static_img": "08_clipped_surrogate.png",
         "interactive": "08_clipped_surrogate.html"},

        {"type": "break"},

        # ══════════════════════════════════════════
        # ALGORITHM
        # ══════════════════════════════════════════
        {"type": "note", "id": "algorithm", "heading": "Algorithm", "html": """
            <h3>Combined Loss</h3>
            <p>If the policy and value function share parameters (common in practice), the loss function combines the policy surrogate, a value function error term, and an entropy bonus:</p>
            <div class="formal-def">
                <div class="math-block">
                    $$L_t^{CLIP+VF+S}(\\theta) = \\widehat{\\mathbb{E}}_t \\left[ L_t^{CLIP}(\\theta) - c_1 L_t^{VF}(\\theta) + c_2 S[\\pi_\\theta](s_t) \\right]$$
                </div>
                <p>where $c_1, c_2$ are coefficients, $S[\\pi_\\theta](s_t)$ is an entropy bonus to encourage exploration, and $L_t^{VF}(\\theta) = ( V_\\theta(s_t) - V_t^{\\text{targ}} )^2$ is the squared-error value function loss.</p>
            </div>

            <h3>Truncated GAE</h3>
            <p>The implementation runs the policy for $T$ timesteps (per trajectory segment, not per episode) and uses a truncated version of GAE:</p>
            <div class="math-block">
                $$\\widehat{A}_t = \\delta_t + (\\gamma \\lambda)\\delta_{t+1} + \\cdots + (\\gamma \\lambda)^{T-t+1}\\delta_{T-1}$$
                $$\\text{where } \\delta_t = r_t + \\gamma V(s_{t+1}) - V(s_t)$$
            </div>

            <h3>PPO, Actor-Critic Style</h3>
            <div class="formal-def">
                <p><strong>Algorithm 1:</strong></p>
                <p><strong>for</strong> iteration $= 1, 2, \\ldots$ <strong>do</strong></p>
                <p style="padding-left: 20px;"><strong>for</strong> actor $= 1, 2, \\ldots, N$ <strong>do</strong></p>
                <p style="padding-left: 40px;">Run policy $\\pi_{\\theta_{\\text{old}}}$ in environment for $T$ timesteps</p>
                <p style="padding-left: 40px;">Compute advantage estimates $\\widehat{A}_1, \\ldots, \\widehat{A}_T$</p>
                <p style="padding-left: 20px;"><strong>end for</strong></p>
                <p style="padding-left: 20px;">Optimize surrogate $L$ w.r.t. $\\theta$, with $K$ epochs and minibatch size $M \\leq NT$</p>
                <p style="padding-left: 20px;">$\\theta_{\\text{old}} \\leftarrow \\theta$</p>
                <p><strong>end for</strong></p>
            </div>

            <div class="callout callout-insight">
                <p>The algorithm is simple: collect a batch of experience with the current policy across $N$ parallel workers ($NT$ total timesteps). Compute advantages using GAE. Do $K$ passes of minibatch gradient ascent on the clipped surrogate. Snapshot the updated policy, throw away the batch, collect fresh data, and repeat. <strong>The clipping is what makes the $K$ epochs of reuse safe</strong> &mdash; without it, multiple passes over the same data would push the policy too far from where the data was collected.</p>
            </div>
        """},

        # Viz 9: PPO vs TRPO vs PG
        {"type": "viz", "id": "viz-09", "num": 9, "title": "PPO vs TRPO vs PG",
         "tool": "manim",
         "desc": "Animated comparison of vanilla policy gradient, TRPO, and PPO: showing how each method handles the step-size problem differently, with PPO achieving TRPO's stability via simple clipping.",
         "video": "PPOvsTRPOvsPG.mp4"},

        {"type": "break"},

        # ══════════════════════════════════════════
        # EXPERIMENTS
        # ══════════════════════════════════════════
        {"type": "note", "id": "experiments", "heading": "Experiments", "html": """
            <h3>Comparison of Surrogate Objectives</h3>
            <ul>
                <li>Tested clipping in log space &mdash; not much improvement.</li>
                <li>Used MuJoCo tasks with 1 million timesteps.</li>
                <li>Hyperparameters: $\\epsilon$ (clip), $\\beta$ (KL penalty), $d_{\\text{targ}}$ (target KL).</li>
                <li>A 2-layer MLP with tanh activations was used. Policy and value function parameters were <strong>not shared</strong>, so the entropy bonus was not used.</li>
            </ul>

            <h3>Continuous Control (MuJoCo)</h3>
            <p><strong>PPO outperformed on almost every task.</strong> To showcase performance on high-dimensional problems, the authors trained on 3D humanoid tasks: running, steering, getting up off the ground, and being pelted by cubes.</p>

            <h3>Atari</h3>
            <p><strong>PPO achieved better results</strong> across Atari games as well. Some practitioners have used the adaptive KL variant for robotics applications.</p>

            <div class="callout callout-note">
                <p>The experiments confirmed that the clipped surrogate objective ($L^{CLIP}$) with $\\epsilon = 0.2$ consistently matched or exceeded the performance of TRPO's constrained optimization &mdash; while being far simpler to implement and tune.</p>
            </div>
        """},

        {"type": "break"},

        # ══════════════════════════════════════════
        # PPO FOR LLM ALIGNMENT
        # ══════════════════════════════════════════
        {"type": "note", "id": "llm-alignment", "heading": "PPO for LLM Alignment", "html": """
            <p>PPO became the standard RL algorithm for aligning language models via RLHF (Reinforcement Learning from Human Feedback). The connection:</p>
            <ol>
                <li><strong>Policy</strong> = the language model. Given a prompt (state), it generates tokens (actions).</li>
                <li><strong>Reward model</strong> = trained from human preferences (as in the <a href="../1706.03741/">RLHF paper</a>).</li>
                <li><strong>PPO optimization</strong> = fine-tune the LM to maximize the learned reward while staying close to the original model (via KL penalty).</li>
            </ol>

            <div class="callout callout-key">
                <p>The KL penalty in LLM alignment serves the same purpose as PPO's clipping: prevent the model from diverging too far from the base policy. Without it, the model would "hack" the reward model by generating degenerate text that scores highly but is nonsensical. This is the same trust-region principle, applied at the token level.</p>
            </div>

            <p>This pipeline &mdash; pretrain LM, train reward model from preferences, fine-tune with PPO &mdash; powered <strong>InstructGPT</strong> (Ouyang et al., 2022) and <strong>ChatGPT</strong>. PPO's simplicity and stability made it the practical choice over TRPO for this application.</p>
        """},

        # Viz 10: PPO for LLM Alignment
        {"type": "viz", "id": "viz-10", "num": 10, "title": "PPO for LLM Alignment",
         "tool": "multi",
         "desc": "Visualization of how PPO is applied in the RLHF pipeline for language model alignment: the three-stage process from pretraining through reward modeling to PPO fine-tuning.",
         "static_img": "10_ppo_llm_alignment.png",
         "interactive": "10_ppo_llm_alignment.html"},

        {"type": "break"},

        # Viz 11: CartPole Demo
        {"type": "viz", "id": "viz-11", "num": 11, "title": "Interactive CartPole Demo",
         "tool": "plotly",
         "desc": "Self-contained PPO implementation running in your browser: watch a simulated CartPole agent learn to balance a pole through clipped surrogate optimization. Adjust hyperparameters ($\\epsilon$, $\\gamma$, $\\lambda$) and observe their effects in real time.",
         "interactive": "11_ppo_cartpole_demo.html",
         "iframe_height": 750},

        {"type": "break"},

        # ══════════════════════════════════════════
        # CONCLUSION
        # ══════════════════════════════════════════
        {"type": "note", "id": "conclusion", "heading": "Conclusion", "html": """
            <p>Proximal Policy Optimization introduces a set of <strong>policy optimization methods</strong> that use <strong>stochastic gradient ascent</strong> to perform each policy update. These methods are as stable and <strong>reliable</strong> as trust-region methods, but <strong>simpler</strong> to implement, requiring only first-order optimization. Better overall performance across both continuous control and discrete action domains.</p>

            <div class="callout callout-key">
                <p>PPO's lasting impact goes beyond its original benchmarks. By making trust-region-quality RL optimization accessible to any practitioner with a standard deep learning toolkit, PPO became the backbone of LLM alignment &mdash; the algorithm that fine-tuned the models behind ChatGPT, Claude, and other instruction-following AI systems.</p>
            </div>
        """},

    ],  # end content

    # ─── Coding Adventures ───
    "adventures": [
        {
            "num": "01",
            "title": "Path-Finding Preference Game",
            "status": "Done",
            "desc": "Full RLHF pipeline in a grid world using PPO as the RL optimizer \u2014 you are the human annotator. Implements the clipped surrogate objective and GAE from this paper. 201 tests.",
            "url": "../../adventures/01/",
            "tags": ["PyTorch", "Rich CLI", "PPO", "GAE"],
        },
        {
            "num": "03",
            "title": "Solve a Rubik\u2019s Cube with RL",
            "status": "Done",
            "desc": "PPO + curriculum learning on a 2\u00d72 Pocket Cube. Implements clipped surrogate objective and GAE in a pure RL setting without human feedback. 162 tests.",
            "url": "../../adventures/03/",
            "tags": ["PyTorch", "Rich CLI", "PPO", "Curriculum"],
        },
    ],
}
