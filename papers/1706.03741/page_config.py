"""Page configuration for 1706.03741 — Deep Reinforcement Learning from Human Preferences."""

PAGE_DATA = {
    "title": "Deep Reinforcement Learning from Human Preferences",
    "arxiv_id": "1706.03741",
    "authors": "Christiano, Leike, Brown, Martic, Legg, Amodei",
    "year": 2017,
    "last_updated": "May 2026",

    "extra_meta_html": '<a href="https://github.com/csking101/LuCiD-papers/tree/main/papers/1706.03741/implementation">Implementation</a>',

    "extra_footer_links": [
        {"label": "Implementation (Cat Grid-World)", "url": "https://github.com/csking101/LuCiD-papers/tree/main/papers/1706.03741/implementation"},
    ],

    "tldr": r"""<p>This paper shows you can train RL agents using human preferences instead of hand-designed reward functions. Show a human two short clips of agent behavior, ask which is better, and use those comparisons to learn a reward function. This works on MuJoCo robotics tasks (700 comparisons) and Atari games (5,500 comparisons), requiring feedback on less than 1% of the agent's interactions with the environment. It can even teach novel behaviors like backflips &mdash; in about an hour of human time. Published in 2017 by researchers from OpenAI and DeepMind, this is the foundational RLHF paper whose approach later powered InstructGPT and ChatGPT.</p>""",

    # ─── Table of Contents (sidebar) ───
    "toc": [
        {"id": "abstract",              "label": "Abstract",               "is_viz": False},
        {"id": "introduction",          "label": "Introduction",           "is_viz": False},
        {"id": "viz-02",                "label": "RL vs RLHF Pipeline",    "is_viz": True},
        {"id": "method",                "label": "Method",                 "is_viz": False},
        {"id": "viz-01",                "label": "System Architecture",    "is_viz": True},
        {"id": "preference-elicitation","label": "Preference Elicitation", "is_viz": False},
        {"id": "viz-03",                "label": "Elicitation Flow",       "is_viz": True},
        {"id": "reward-function",       "label": "Reward Function",        "is_viz": False},
        {"id": "viz-04",                "label": "Bradley-Terry Model",    "is_viz": True},
        {"id": "viz-05",                "label": "Cross-Entropy Loss",     "is_viz": True},
        {"id": "selecting-queries",     "label": "Selecting Queries",      "is_viz": False},
        {"id": "viz-06",                "label": "Reward Convergence",     "is_viz": True},
        {"id": "results",               "label": "Experimental Results",   "is_viz": False},
        {"id": "viz-07",                "label": "MuJoCo Results",         "is_viz": True},
        {"id": "viz-08",                "label": "Atari Results",          "is_viz": True},
        {"id": "ablations",             "label": "Ablation Studies",       "is_viz": False},
        {"id": "viz-09",                "label": "Ablation Heatmap",       "is_viz": True},
        {"id": "conclusion",            "label": "Conclusion",             "is_viz": False},
        {"id": "compute-cost",          "label": "Compute vs. Human Cost", "is_viz": True},
        {"id": "limitations",           "label": "Limitations",            "is_viz": True},
        {"id": "legacy",                "label": "Historical Significance","is_viz": True},
        {"id": "rlhf-demo",            "label": "Implementation Demo",    "is_viz": False},
        {"id": "demo-viz",             "label": "RLHF Demo",              "is_viz": True},
        {"id": "viz-10",                "label": "Interactive Demo",       "is_viz": True},
    ],

    # ─── Nav bar (mobile, abbreviated) ───
    "nav": [
        {"id": "abstract",              "label": "Abstract"},
        {"id": "introduction",          "label": "Introduction"},
        {"id": "method",                "label": "Method"},
        {"id": "preference-elicitation","label": "Preferences"},
        {"id": "reward-function",       "label": "Reward Function"},
        {"id": "results",               "label": "Results"},
        {"id": "ablations",             "label": "Ablations"},
        {"id": "conclusion",            "label": "Conclusion"},
        {"id": "rlhf-demo",            "label": "Implementation"},
        {"id": "viz-10",                "label": "Interactive Demo"},
    ],

    # ─── Viz gallery cards ───
    "viz_gallery": [
        {"num": 1,  "title": "System Architecture",       "tool": "manim",  "tag_label": "Manim"},
        {"num": 2,  "title": "RL vs RLHF Pipeline",       "tool": "manim",  "tag_label": "Manim"},
        {"num": 3,  "title": "Preference Elicitation",     "tool": "manim",  "tag_label": "Manim"},
        {"num": 4,  "title": "Bradley-Terry Model",        "tool": "multi",  "tag_label": "MPL+Plotly"},
        {"num": 5,  "title": "Cross-Entropy Loss",         "tool": "multi",  "tag_label": "MPL+Plotly"},
        {"num": 6,  "title": "Reward Convergence",         "tool": "manim",  "tag_label": "Manim"},
        {"num": 7,  "title": "MuJoCo Results",             "tool": "mpl",    "tag_label": "Matplotlib"},
        {"num": 8,  "title": "Atari Results",              "tool": "mpl",    "tag_label": "Matplotlib"},
        {"num": 9,  "title": "Ablation Heatmap",           "tool": "multi",  "tag_label": "MPL+Plotly"},
        {"num": 10, "title": "Preference Simulation",      "tool": "plotly", "tag_label": "Plotly"},
    ],

    # ─── Content blocks (ordered) ───
    "content": [

        # ══════════════════════════════════════════
        # ABSTRACT
        # ══════════════════════════════════════════
        {"type": "note", "id": "abstract", "heading": "Abstract", "html": r"""
            <ul>
                <li>If an RL system needs to be useful, we need to communicate complex goals to these systems.</li>
                <li>In this paper, goals are defined in terms of human preferences between pairs of trajectory segments.</li>
                <li>Therefore, without access to the reward function, the RL task can be solved.</li>
                <li>This works on Atari games and simulated robot locomotion, while providing feedback on less than 1% of the agent's interactions with the environment.</li>
                <li>Novel complex behaviors can be trained with about an hour of human time &mdash; considerably more complex than any previously learned from human feedback.</li>
            </ul>
        """},

        # ══════════════════════════════════════════
        # INTRODUCTION
        # ══════════════════════════════════════════
        {"type": "note", "id": "introduction", "heading": "Introduction", "html": r"""
            <ul>
                <li>At the time, success was there in scaling RL with well defined reward functions.</li>
                <li>That is not the usual case.</li>
                <li>You can try coming up with a simple function, however, the RL will end up satisfying that function and not solving the overall task.</li>
                <li>For example, suppose you wanted a robot to clean a table or scramble an egg &mdash; it's not clear how to write a reward function over the robot's sensors. A simple approximation will often result in behavior that games the reward without actually satisfying your preferences. This is a core concern in AI alignment.</li>
                <li>It'd be better to try to convey objectives to the agents.</li>
                <li>If you have demonstrations of the task, you can extract a reward function by using inverse RL.</li>
                <li>You can use Imitation Learning, to clone the demonstrated behaviour as well.</li>
                <li>Another alternative is to allow a human to provide feedback on the system's current behaviour and use this feedback to define the task, however this is expensive. For deep RL systems, you need a lot of human feedback.</li>
            </ul>

            <div class="callout callout-key">
                <p>What the paper tries to do is &mdash; <em>learn a reward function from human feedback and then optimize that reward function.</em></p>
            </div>

            <ul>
                <li>Therefore, we want a well-specified reward function that:
                    <ul>
                        <li>enables us to solve tasks for which we can only recognize the desired behaviour, but not necessarily demonstrate it</li>
                        <li>allows agents to be taught by non-expert users</li>
                        <li>scales to large problems</li>
                        <li>is economic with user feedback</li>
                    </ul>
                </li>
                <li>The algorithm does two things at once:
                    <ol>
                        <li>Fits a reward function to the human's preferences</li>
                        <li>Simultaneously training a policy to optimize the current predicted reward function</li>
                    </ol>
                </li>
                <li>The paper experiments with two domains &mdash; Atari games and robotics tasks in MuJoCo.</li>
                <li>We look for &mdash; small amount of feedback, leading to good results.</li>
                <li>Then we see if the algorithm can learn novel behaviours.</li>
            </ul>

            <h3>Related Work</h3>
            <ul>
                <li>Lot of work with RL from human ratings/rankings.</li>
                <li>Some use preferences instead of absolute reward values.</li>
                <li>Old work considers continuous domains with four degrees of freedom and small discrete domains. This work considers physics tasks that have dozens of degrees of freedom &amp; Atari tasks have no hand engineered features.</li>
                <li>Old work with feedback was to have a target policy and fit the reward function to that using Bayesian inference. Synthetic human feedback was drawn from the Bayesian model and used, instead of RL.</li>
            </ul>
        """},

        # Viz 2: RL vs RLHF Pipeline
        {"type": "viz", "id": "viz-02", "num": 2, "title": "RL vs RLHF Pipeline",
         "tool": "manim",
         "desc": "Side-by-side comparison of the traditional RL pipeline (hand-designed reward) vs the RLHF pipeline (learned reward from human preferences).",
         "video": "RLvsRLHF.mp4"},

        {"type": "break"},

        # ══════════════════════════════════════════
        # METHOD
        # ══════════════════════════════════════════
        {"type": "note", "id": "method", "heading": "Preliminaries and Method", "html": r"""
            <h3>Setting and Goal</h3>
            <ul>
                <li>There is an agent interacting with an environment over a sequence of steps; at each time $t$, the agent receives an observation $o_t \in O$ from the environment, and then sends an action $a_t \in A$ to the environment.</li>
                <li>In traditional RL, the environment would give a reward and the agent would like to maximize the discounted sum of rewards.</li>
                <li>However, instead of assuming that the environment produces a reward signal, we assume that there is a human overseer who can express preferences between <strong>trajectory segments</strong>.</li>
                <li>A trajectory segment &mdash; sequence of observations and actions: $$\sigma = ((o_0,a_0), (o_1,a_1),...,(o_{k-1},a_{k-1})) \in (O \times A)^k$$</li>
                <li>Human preferred trajectory segment: $\sigma^1 \succ \sigma^2$</li>
                <li>The goal of the agent, is to produce <strong>trajectories that are preferred by human, while making as few queries to the human.</strong></li>
            </ul>

            <div class="formal-def">
                <p>We say that preferences $\succ$ are generated by a reward function $r : \mathcal{O} \times \mathcal{A} \rightarrow \mathbb{R}$ if</p>
                <div class="math-block">
                    $$ ((o_0^1,a_0^1),...,(o_{k-1}^1,a_{k-1}^1)) \succ ((o_0^2,a_0^2),...,(o_{k-1}^2,a_{k-1}^2)) $$
                    whenever
                    $$ r(o_{0}^1,a_0^1) + ... + r(o_{k-1}^1,a_{k-1}^1) > r(o_{0}^2,a_0^2) + ... + r(o_{k-1}^2,a_{k-1}^2) $$
                </div>
                <p>This means that preferences imply a higher total reward overall. If we know the reward function $r$, we can evaluate the agent quantitatively. However, sometimes there is no reward function by which we can quantitatively evaluate the same.</p>
            </div>

            <h3>Our Method</h3>
            <p>At each point in time, our method maintains a:</p>
            <ul>
                <li>Policy $\pi : \mathcal{O} \rightarrow \mathcal{A}$</li>
                <li>Reward Function Estimate $\widehat{r} : \mathcal{O} \times \mathcal{A} \rightarrow \mathbb{R}$</li>
            </ul>
            <p>Both of which are parametrized by deep neural networks.</p>
            <p>These networks are updated by 3 processes:</p>
            <ol>
                <li>Policy $\pi$ interacts with the environment to produce a set of trajectories $\{\tau^1,...,\tau^i\}$. The parameters of $\pi$ are updated by a traditional RL algorithm, in order to maximize the sum of predicted rewards $r_t = \widehat{r}(o_t,a_t)$.</li>
                <li>We select pairs of segments $(\sigma^1, \sigma^2)$ from the trajectories produced in step 1, and then send them to a human for comparison.</li>
                <li>The parameters of the mapping $\widehat{r}$ are optimized via supervised learning to fit the comparisons collected from the human so far.</li>
            </ol>

            <div class="callout callout-insight">
                <p>These 3 processes run asynchronously. Trajectories flow from process 1 &rarr; 2, human comparisons from 2 &rarr; 3, and parameters for $\widehat{r}$ from 3 &rarr; 1.</p>
            </div>

            <ul>
                <li>Process 1 &rarr; Does the stuff and tries things out.</li>
                <li>Process 2 &rarr; See's what stuff is done and asks a human to see what was good and what wasn't good.</li>
                <li>Process 3 &rarr; Update the rewards based on what the human said.</li>
            </ul>
        """},

        # Viz 1: System Architecture
        {"type": "viz", "id": "viz-01", "num": 1, "title": "System Architecture",
         "tool": "manim",
         "desc": "Step-by-step animation of the three asynchronous processes: policy training, human preference collection, and reward model learning (Figure 1 from the paper).",
         "video": "SystemArchitecture.mp4"},

        # Optimizing the Policy
        {"type": "note", "id": "optimizing-policy", "heading": "Optimizing the Policy", "html": r"""
            <p>After using $\widehat{r}$ to compute the rewards, it becomes a normal RL problem.</p>
            <p>We can use any algorithm to solve the problem. However, if $\widehat{r}$ is non-stationary, we would like to choose methods that are robust to changes in the reward function. This is why policy gradient methods are useful.</p>
            <p>For this paper:</p>
            <ul>
                <li>Advantage Actor-Critic (A2C) &rarr; Atari</li>
                <li>Trust Region Policy Optimization (TRPO) &rarr; Simulated Robotics Tasks</li>
            </ul>

            <div class="callout callout-note">
                <p>Parameter settings were reused from traditional RL tasks. Only entropy bonus was adjusted for TRPO because TRPO relies on the trust region to ensure adequate exploration. If the reward function is changing, there might be inadequate exploration.</p>
            </div>

            <p>Rewards produced by $\widehat{r}$ were normalized to have zero mean and constant standard deviation. This is preprocessing, since the position of the rewards haven't been determined yet.</p>
        """},

        {"type": "break"},

        # ══════════════════════════════════════════
        # PREFERENCE ELICITATION
        # ══════════════════════════════════════════
        {"type": "note", "id": "preference-elicitation", "heading": "Preference Elicitation", "html": r"""
            <p>The human is given a visualization of two segments, in the form of short movie clips, around 1-2 seconds long. The human says which is good, both good, or unable to compare.</p>
            <p>A database $\mathcal{D}$ of triples $(\sigma^1,\sigma^2,\mu)$, where $\mu$ is the distribution over $\{1,2\}$, indicating which segment the human preferred.</p>
            <ul>
                <li>If the human thinks 1 is better, then all the mass is on 1.</li>
                <li>If equal, then the distribution is uniform.</li>
                <li>If uncomparable, then it's not included in the database.</li>
            </ul>

            <div class="callout callout-insight">
                <p><strong>Why comparisons instead of scores?</strong> The authors found it much easier for humans to provide consistent comparisons than consistent absolute scores. On continuous control tasks, predicting comparisons worked much better than predicting scores &mdash; likely because reward scales vary substantially, complicating the regression problem. Comparisons smooth this out: you only need to predict which is better, not by how much. This mirrors the Elo system in chess &mdash; you don't need to know how "good" a player is on an absolute scale, just who beats whom.</p>
            </div>
        """},

        # Viz 3: Preference Elicitation Flow
        {"type": "viz", "id": "viz-03", "num": 3, "title": "Preference Elicitation Flow",
         "tool": "manim",
         "desc": "Walkthrough of the 5-step preference collection process: segment sampling, pair presentation, human comparison, label recording, and reward model update.",
         "video": "PreferenceElicitation.mp4"},

        {"type": "break"},

        # ══════════════════════════════════════════
        # FITTING THE REWARD FUNCTION
        # ══════════════════════════════════════════
        {"type": "note", "id": "reward-function", "heading": "Fitting the Reward Function", "html": r"""
            <p>$\widehat{r}$ is interpreted as a preference-predictor, if we view it as a latent factor explaining the human's judgements and assume that the human's probability of preferring a segment $\sigma^i$ depends exponentially on the value of the latent reward summed over the length of the clip (here discounting isn't used):</p>
            <div class="math-block">
                $$\widehat{P} [\sigma^1 \succ \sigma^2] = \frac{\exp \sum \widehat{r}(o_t^1,a_t^1)}{\exp \sum \widehat{r}(o_t^1,a_t^1) + \exp \sum \widehat{r}(o_t^2,a_t^2)}$$
            </div>
            <p>This is basically converting the preference into probabilities and normalising it.</p>
            <p>We choose $\widehat{r}$ to minimize the cross-entropy loss between these predictions and the actual human labels:</p>
            <div class="math-block">
                $$ \text{loss}(\widehat{r}) = - \sum_{(\sigma^1,\sigma^2,\mu)\in \mathcal{D}} \mu(1) \log\widehat{P} [\sigma^1 \succ \sigma^2] + \mu(2) \log\widehat{P} [\sigma^2 \succ \sigma^1] $$
            </div>

            <div class="callout callout-insight">
                <p>This follows from the <strong>Bradley-Terry model</strong> &mdash; estimate score functions from pairwise-preferences. It is also the specialization of the <strong>Luce-Shephard choice rule</strong>, to preferences over trajectory segments.</p>
                <p>It's similar to how difference in Elo points for chess players is calculated pairwise, and estimates the probability of one player defeating another in the game &mdash; the difference in the predicted reward of two trajectory segments estimates the probability that one is chosen over the other by the human.</p>
            </div>

            <p>There were some modifications done in the approach:</p>
            <ul>
                <li>An ensemble of predictors were trained on $|\mathcal{D}|$ triples sampled from $\mathcal{D}$, with replacement. The estimate of $\widehat{r}$ is defined by independently normalizing each of the predictors and averaging the results.</li>
                <li>A fraction of $\frac{1}{e}$ is held out as a validation set for each predictor. L2 regularization is used, and val loss is between 1.1 to 1.5 times the training loss. In some domains, dropout was used too.</li>
                <li>Rather than using softmax directly, we assume that there is a 10% chance that the human responds uniformly at random. This is needed because human raters have a constant probability of making an error, which doesn't decay to 0 as the reward difference becomes extreme.</li>
            </ul>
        """},

        # Viz 4: Bradley-Terry Model
        {"type": "viz", "id": "viz-04", "num": 4, "title": "Bradley-Terry Model",
         "tool": "multi",
         "desc": "The Bradley-Terry sigmoid that converts reward differences into preference probabilities, with human error rate curves showing the effect of the epsilon parameter.",
         "static_img": "04_bradley_terry.png",
         "interactive": "04_bradley_terry.html"},

        # Viz 5: Cross-Entropy Loss
        {"type": "viz", "id": "viz-05", "num": 5, "title": "Cross-Entropy Loss",
         "tool": "multi",
         "desc": "Visualization of the cross-entropy loss used to train the reward model on human preference comparisons, including a rotatable 3D loss surface and contour plot.",
         "static_img": "05_cross_entropy_loss.png",
         "interactive": "05_cross_entropy_loss.html"},

        # Selecting Queries
        {"type": "note", "id": "selecting-queries", "heading": "Selecting Queries", "html": r"""
            <p>Preferences are queries based on an approximation to uncertainty in the reward function estimator.</p>
            <ol>
                <li>Sample a large number of pairs of trajectory segments of length $k$.</li>
                <li>Use each reward predictor in our ensemble to predict which segment will be preferred from each pair.</li>
                <li>Select those trajectories for which the predictions have the highest variance across the ensemble members.</li>
            </ol>

            <div class="callout callout-note">
                <p>This is a crude approximation &mdash; in some tasks it impairs performance. Ideally, we would want to query based on the expected value of the information of the query.</p>
            </div>
        """},

        # Viz 6: Reward Convergence
        {"type": "viz", "id": "viz-06", "num": 6, "title": "Reward Convergence",
         "tool": "manim",
         "desc": "Animation showing how the reward model's predictions improve and converge as more human comparison data is collected over time.",
         "video": "RewardConvergence.mp4"},

        {"type": "break"},

        # ══════════════════════════════════════════
        # EXPERIMENTAL RESULTS
        # ══════════════════════════════════════════
        {"type": "note", "id": "results", "heading": "Experimental Results", "html": r"""
            <p>In TensorFlow, MuJoCo and Arcade Learning Environment interfaced with OpenAI Gym.</p>

            <h3>RL Tasks with Unobserved Rewards</h3>
            <p>Solve a range of benchmark tasks for deep RL without observing the true rewards. The agent learns about the goal by asking a human which segment is better.</p>
            <p>Feedback is given by contractors. Each trajectory segment is 1-2 seconds. Contractors responded in 3-5 seconds, total time was 0.5-5 hours.</p>
            <p>For MuJoCo, <strong>700 queries</strong> were sent to human raters. For Atari, <strong>5,500 queries</strong>. A synthetic oracle (whose preferences exactly reflect the true reward) was also tested at 350, 700, and 1,400 labels for comparison. The baseline is standard RL with access to the true reward function.</p>
            <p>You can use a synthetic oracle whose preferences over trajectories exactly reflect reward in the underlying task. Aim is to do well without access to reward information, and rely on the scarce feedback. Human feedback can do better however.</p>
            <p>Variable-length episodes were removed to avoid encoding task information in the termination conditions &mdash; for example, ending the episode when the robot falls over implicitly tells the agent that falling is bad, even without a reward signal.</p>

            <h4>Simulated Robotics</h4>
            <p><strong>Tasks</strong>: Hopper, Walker, HalfCheetah, Reacher, Ant, Swimmer, Pendulum, Double Pendulum &mdash; 8 tasks total.</p>
            <p>Reward functions are linear functions of distances, positions and velocities, and all are quadratic functions of the features.</p>
            <p>With 700 labels, the learned reward nearly matches standard RL on all tasks. At 1,400 labels, the RLHF agent actually <strong>slightly outperforms</strong> standard RL &mdash; likely because the learned reward is better shaped (it assigns positive rewards to all behaviors that are typically followed by high reward).</p>
            <p>Real human feedback is only slightly less effective than synthetic. Depending on the task, human feedback ranged from half as efficient as ground truth to equally efficient.</p>
            <p>On the Ant task, human feedback significantly <strong>outperformed</strong> synthetic because humans were asked to prefer trajectories where the robot was "standing upright," which provided useful reward shaping. The RL reward function had a similar bonus, but the simple hand-crafted version was not as useful.</p>
        """},

        # Viz 7: MuJoCo Results
        {"type": "viz", "id": "viz-07", "num": 7, "title": "MuJoCo Results",
         "tool": "mpl",
         "desc": "8-panel reproduction of Figure 2 from the paper: learning curves across MuJoCo continuous control tasks comparing RLHF agents against RL baselines.",
         "static_img": "07_mujoco_results.png"},

        # Atari
        {"type": "note", "id": "atari", "heading": "Atari Games", "html": r"""
            <p><strong>Games tested</strong>: BeamRider, Breakout, Enduro, Pong, Qbert, Seaquest, SpaceInvaders &mdash; the same 7 games from Mnih et al. (2013).</p>
            <p>On most games, real human feedback is similar or slightly worse than synthetic, even if synthetic labels are 40% lesser. This may be due to human error in labelling, inconsistency between different contractors labelling the same run, or uneven rate of labelling by contractors. This can make some labels concentrated in the state space.</p>
            <p>Specific results:</p>
            <ul>
                <li><strong>BeamRider and Pong</strong>: synthetic labels match or approach RL with only 3,300 labels.</li>
                <li><strong>Seaquest and Qbert</strong>: eventually reach RL-level performance but learn more slowly.</li>
                <li><strong>SpaceInvaders and Breakout</strong>: never fully match RL, but substantial improvement &mdash; passing the first level in SpaceInvaders, reaching a score of 20-50 on Breakout.</li>
                <li><strong>Qbert</strong>: fails with real human feedback &mdash; short clips are confusing and difficult to evaluate.</li>
                <li><strong>Enduro</strong>: human feedback <strong>outperforms</strong> RL &mdash; humans reward any progress towards passing cars, essentially providing reward shaping that A2C can't discover through random exploration alone.</li>
            </ul>
        """},

        # Viz 8: Atari Results
        {"type": "viz", "id": "viz-08", "num": 8, "title": "Atari Results",
         "tool": "mpl",
         "desc": "7-panel reproduction of Figure 3 from the paper: learning curves across Atari games comparing RLHF agents against RL baselines.",
         "static_img": "08_atari_results.png"},

        # Novel Behaviours
        {"type": "note", "id": "novel-behaviours", "heading": "Novel Behaviours", "html": r"""
            <p>The ultimate purpose of human interaction is to solve tasks where no reward function is available. Using the same parameters as the benchmark experiments, the authors demonstrate:</p>
            <ul>
                <li><strong>Hopper backflip</strong>: The robot learns to perform a sequence of backflips, landing upright each time, and repeat. Trained with <strong>900 queries</strong> in less than an hour. This is a behavior you can recognize but would be very hard to specify as a mathematical reward function.</li>
                <li><strong>HalfCheetah one-legged running</strong>: The robot moves forward while balancing on one leg. Trained with <strong>800 queries</strong> in under an hour.</li>
                <li><strong>Enduro driving with traffic</strong>: Rather than passing other cars, the agent learns to keep alongside them. Trained with <strong>~1,300 queries</strong> over 4 million frames. The agent stays even with moving cars for a substantial fraction of the episode, though it gets confused by background changes.</li>
            </ul>
            <div class="callout callout-key">
                <p>These demonstrate the core promise of the approach: teaching behaviors you can <strong>recognize</strong> but can't easily <strong>demonstrate</strong> or <strong>specify</strong> as a reward function. A backflip is easy to judge ("did it land upright?") but nearly impossible to write as a mathematical function of joint angles and velocities.</p>
            </div>
        """},

        {"type": "break"},

        # ══════════════════════════════════════════
        # ABLATION STUDIES
        # ══════════════════════════════════════════
        {"type": "note", "id": "ablations", "heading": "Ablation Studies", "html": r"""
            <p>Changes tested: random queries, no ensemble, no online queries (only those gathered at the beginning of training, not throughout), no regularization, no segments, target (true rewards).</p>

            <div class="callout callout-key">
                <p>In offline training, its performance was very poor. The nonstationarity of the occupancy distribution leads to the predictor capturing only a part of the true rewards, and maximising this partial reward can lead to bizarre behaviour that is undesirable as measured by the true reward. Thus, general human feedback is required.</p>
            </div>

            <p>For instance, on Pong, offline training sometimes leads the agent to avoid losing points but not to score points &mdash; resulting in extremely long volleys that repeat the same sequence of events ad infinitum. This demonstrates that human feedback needs to be <strong>intertwined with RL learning</strong>, not provided statically.</p>

            <p>On the continuous control tasks, humans gave better scores/feedback. Prediction comparisons here was better than predicting scores. This is likely because the scale of rewards and scores makes things complicated.</p>
            <p>There were large performance differences in using single frames rather than clips. Asking humans to compare longer clips was more helpful per clip and significantly less helpful per frame. Short clips took time to understand the situation for humans, but for longer ones, it was a linear function of the clip length. In Atari, easier to compare longer clips since there is more context.</p>
            <p>The authors tried to choose the shortest clip length for which evaluation time was linear &mdash; rather than dominated by the human just trying to figure out what's happening in the scene.</p>
        """},

        # Viz 9: Ablation Heatmap
        {"type": "viz", "id": "viz-09", "num": 9, "title": "Ablation Heatmap",
         "tool": "multi",
         "desc": "Heatmap of the ablation study results showing the impact of different design decisions on final performance.",
         "static_img": "09_ablation_heatmap.png",
         "interactive": "09_ablation_heatmap.html"},

        {"type": "break"},

        # ══════════════════════════════════════════
        # DISCUSSION AND CONCLUSIONS
        # ══════════════════════════════════════════
        {"type": "note", "id": "conclusion", "heading": "Discussion and Conclusions", "html": r"""
            <div class="callout callout-key">
                <p>Agent-environment interactions are radically cheaper than human interaction. Learning a separate reward model using supervised learning reduces the interaction complexity by 3 orders of magnitude.</p>
            </div>
            <p>We can train deep RL agents from human preferences. Also, we are hitting diminishing returns on further improvements in non-expert feedback.</p>

            <h3 id="compute-cost">Compute vs. Human Cost</h3>
            <p>For the Atari experiments, compute cost was roughly <strong>$25</strong> (a VM with 16 CPUs and one K80 GPU for about a day). Training with 5,000 labels corresponds to roughly 5 hours of human labor, totaling about <strong>$36</strong> at US minimum wage. The fact that compute and human costs are already comparable means we're hitting diminishing returns on further sample-efficiency improvements.</p>

            <h3 id="limitations">Limitations</h3>
            <ul>
                <li>The approach depends on the quality and consistency of human feedback. Contractors can be inconsistent, labels can cluster in narrow parts of state space, and some tasks (like Qbert) produce clips that are genuinely confusing to evaluate.</li>
                <li>The reward model can be exploited &mdash; the policy may find behaviors that score highly under the learned reward without actually being desirable. This is why online learning (continually gathering new feedback) is critical.</li>
                <li>The method is demonstrated on relatively simple environments. Scaling to more complex real-world tasks with higher-dimensional observations remains an open challenge.</li>
            </ul>

            <h3 id="legacy">Historical Significance</h3>
            <p>This paper, authored by researchers from OpenAI (Christiano, Brown, Amodei) and DeepMind (Leike, Martic, Legg), established the RLHF framework that would reshape AI development. The same approach was later scaled to language models in <em>Learning to Summarize from Human Feedback</em> (Stiennon et al., 2020), then to <em>InstructGPT</em> (Ouyang et al., 2022), which directly led to ChatGPT. Dario Amodei went on to co-found Anthropic, where RLHF remains central to Claude's training. Jan Leike later led alignment research at OpenAI. This paper is where that entire trajectory began.</p>
        """},

        {"type": "break"},

        # ══════════════════════════════════════════
        # RLHF IMPLEMENTATION DEMO
        # ══════════════════════════════════════════
        {"type": "note", "id": "rlhf-demo", "heading": "Implementation: Cat Grid-World RLHF", "html": r"""
            <p>We implemented the paper's RLHF framework on a minimal 8&times;8 grid world: a cat at (0,0) must reach the goal at (7,7), avoiding a trap at (1,0) and a wall at (1,1). A synthetic oracle provides preference comparisons using distance-shaped rewards, and the agent learns both a reward model and policy from these preferences alone &mdash; exactly as described in the paper.</p>
            <div class="callout callout-note">
                <p>All visualizations below render from a single training run (100 iterations, ~40s on CPU). Data is loaded from <code>rlhf_results.json</code> and rendered client-side with Canvas and Plotly.</p>
            </div>
        """},

        # RLHF Demo iframe
        {"type": "viz", "id": "demo-viz", "num": 11, "title": "Interactive RLHF Demo",
         "tool": "plotly",
         "desc": "Full interactive RLHF implementation: policy grids, reward heatmap, training curves, trajectory animation, and preference pair replay.",
         "interactive": "rlhf_demo.html",
         "iframe_height": 1600},

        {"type": "break"},

        # Viz 10: Interactive Preference Simulation
        {"type": "viz", "id": "viz-10", "num": 10, "title": "Interactive Preference Simulation",
         "tool": "plotly",
         "desc": "Full interactive RLHF preference simulation: generate trajectory pairs, make preference comparisons, and watch the reward model learn in real time.",
         "interactive": "10_preference_demo.html",
         "iframe_height": 700},

    ],  # end content

    # ─── Coding Adventures ───
    "adventures": [
        {
            "num": "01",
            "title": "Path-Finding Preference Game",
            "status": "Done",
            "desc": "Full RLHF pipeline in a grid world \u2014 you are the human annotator. Pre-train an agent, rate trajectory pairs, train a Bradley-Terry reward model, and PPO fine-tune with KL penalty. 201 tests.",
            "url": "https://github.com/csking101/LuCiD-papers/tree/main/coding-adventures/01-pathfinding-preference-game",
            "tags": ["PyTorch", "Rich CLI", "RLHF", "PPO"],
            "svg_thumb": "../../adventures/01/01_grid_world.svg",
        },
    ],
}
