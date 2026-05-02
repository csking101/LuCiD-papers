"""Page configuration for 2009.01325 — Learning to Summarize from Human Feedback."""

PAGE_DATA = {
    "title": "Learning to Summarize from Human Feedback",
    "arxiv_id": "2009.01325",
    "authors": "Stiennon, Ouyang, Wu, Ziegler, Lowe, Voss, Radford, Amodei, Christiano",
    "year": 2020,
    "last_updated": "May 2026",

    "tldr": """<p>This paper demonstrates that optimizing language models for <strong>human preferences via reinforcement learning</strong> produces significantly better summaries than supervised fine-tuning alone. The pipeline: (1) SFT a GPT-3-style model on Reddit TL;DR, (2) train a <strong>reward model</strong> from 65K human comparisons using a Bradley-Terry loss, (3) fine-tune with <strong>PPO</strong> using the learned reward plus a KL penalty against the SFT policy. The resulting 6.7B model generates summaries preferred by humans over the reference TL;DRs and even over much larger supervised models. The approach transfers zero-shot to CNN/DM news summarization. A critical finding: <strong>reward overoptimization</strong> &mdash; past a certain KL budget, the reward model score keeps climbing while true quality degrades.</p>""",

    # ─── Table of Contents (sidebar) ───
    "toc": [
        {"id": "abstract",            "label": "Abstract",               "is_viz": False},
        {"id": "introduction",        "label": "Introduction",           "is_viz": False},
        {"id": "viz-01",              "label": "RLHF Text Pipeline",     "is_viz": True},
        {"id": "method",              "label": "Method",                 "is_viz": False},
        {"id": "reward-model",        "label": "Reward Model",           "is_viz": False},
        {"id": "viz-02",              "label": "Bradley-Terry RM",       "is_viz": True},
        {"id": "human-feedback",      "label": "Human Feedback Policies","is_viz": False},
        {"id": "viz-03",              "label": "KL-Penalized Reward",    "is_viz": True},
        {"id": "results-reddit",      "label": "Results: Reddit",        "is_viz": False},
        {"id": "viz-04",              "label": "SFT vs RL Quality",      "is_viz": True},
        {"id": "results-news",        "label": "Results: News",          "is_viz": False},
        {"id": "viz-05",              "label": "RM Accuracy Scaling",    "is_viz": True},
        {"id": "reward-understanding","label": "Understanding the RM",   "is_viz": False},
        {"id": "viz-06",              "label": "Reward Overoptimization", "is_viz": True},
        {"id": "viz-07",              "label": "KL Coefficient Effect",  "is_viz": True},
        {"id": "automatic-metrics",   "label": "Automatic Metrics",      "is_viz": False},
        {"id": "viz-08",              "label": "ROUGE vs Learned RM",    "is_viz": True},
        {"id": "viz-09",              "label": "Transfer: CNN/DM",       "is_viz": True},
        {"id": "discussion",          "label": "Discussion",             "is_viz": False},
        {"id": "viz-10",              "label": "Alignment Timeline",     "is_viz": True},
    ],

    # ─── Nav bar (mobile, abbreviated) ───
    "nav": [
        {"id": "abstract",            "label": "Abstract"},
        {"id": "introduction",        "label": "Intro"},
        {"id": "method",              "label": "Method"},
        {"id": "reward-model",        "label": "Reward Model"},
        {"id": "human-feedback",      "label": "RL Policy"},
        {"id": "results-reddit",      "label": "Results"},
        {"id": "reward-understanding","label": "Understanding RM"},
        {"id": "automatic-metrics",   "label": "Metrics"},
        {"id": "discussion",          "label": "Discussion"},
    ],

    # ─── Viz gallery cards ───
    "viz_gallery": [
        {"num": 1,  "title": "RLHF Text Pipeline",      "tool": "manim", "tag_label": "Manim"},
        {"num": 2,  "title": "Bradley-Terry RM",         "tool": "manim", "tag_label": "Manim"},
        {"num": 3,  "title": "KL-Penalized Reward",      "tool": "manim", "tag_label": "Manim"},
        {"num": 4,  "title": "SFT vs RL Quality",        "tool": "multi", "tag_label": "MPL+Plotly"},
        {"num": 5,  "title": "RM Accuracy Scaling",      "tool": "multi", "tag_label": "MPL+Plotly"},
        {"num": 6,  "title": "Reward Overoptimization",   "tool": "multi", "tag_label": "MPL+Plotly"},
        {"num": 7,  "title": "KL Coefficient Effect",    "tool": "multi", "tag_label": "MPL+Plotly"},
        {"num": 8,  "title": "ROUGE vs Learned RM",      "tool": "multi", "tag_label": "MPL+Plotly"},
        {"num": 9,  "title": "Transfer: CNN/DM",         "tool": "mpl",   "tag_label": "Matplotlib"},
        {"num": 10, "title": "Alignment Timeline",       "tool": "manim", "tag_label": "Manim"},
    ],

    # ─── Content blocks (ordered) ───
    "content": [

        # ══════════════════════════════════════════
        # ABSTRACT
        # ══════════════════════════════════════════
        {"type": "note", "id": "abstract", "heading": "Abstract", "html": """
            <p>As language models become more powerful, training and evaluation are increasingly bottlenecked by the data and metrics used for a particular task. In this work, the authors show that it's possible to significantly improve summary quality &mdash; far beyond what metrics like ROUGE capture &mdash; by training a model to <strong>optimize for human preferences</strong>.</p>
            <ul>
                <li>A large, high-quality dataset of human preferences is created (65K comparisons).</li>
                <li>A <strong>reward model</strong> is trained to predict which summary a human would prefer.</li>
                <li>That reward model is used to fine-tune a summarization policy via <strong>reinforcement learning</strong> (PPO).</li>
            </ul>
            <div class="callout callout-key">
                <p>TL;DR: this paper applies the full RLHF pipeline &mdash; previously demonstrated on Atari and simulated robotics &mdash; to a real NLP task (text summarization) for the first time at scale.</p>
            </div>
        """},

        # ══════════════════════════════════════════
        # INTRODUCTION
        # ══════════════════════════════════════════
        {"type": "note", "id": "introduction", "heading": "Introduction", "html": """
            <p>Most NLP tasks train models to maximize the log probability of human demonstrations. But there is a fundamental <strong>misalignment</strong>: we want <em>"generating high-quality outputs as determined by humans"</em>, while the model objective is <em>"maximize the likelihood of human-written text"</em>.</p>

            <p>This misalignment has concrete consequences:</p>
            <ul>
                <li>The model doesn't distinguish important errors (making up facts) from unimportant ones (choosing a slightly different synonym).</li>
                <li>Models are incentivized to place probability mass on all human demonstrations, even low-quality ones.</li>
                <li>Distributional shift during sampling can degrade performance.</li>
            </ul>

            <p>Metrics like ROUGE, the standard automatic metric for evaluating summary quality, have received criticism for poor correlation with human judgement. The paper's four main contributions:</p>
            <ol>
                <li>Training with human feedback <strong>significantly outperforms</strong> very strong baselines on English summarization.</li>
                <li>Human feedback models <strong>generalize much better</strong> to new domains than supervised models.</li>
                <li>Extensive empirical analyses of the policy and reward model behavior.</li>
                <li>Public release of the human feedback dataset for further research.</li>
            </ol>

            <div class="callout callout-insight">
                <p>The key realization: we can make models generate text (they know how to output something), and then use RLHF to ensure the policy outputs what we <em>actually want</em>. This captures the essence of human evaluation &mdash; abstract qualities like accuracy, coherence, and coverage &mdash; that log-likelihood training misses.</p>
            </div>
        """},

        # Viz 1
        {"type": "viz", "id": "viz-01", "num": 1, "title": "RLHF Text Pipeline",
         "tool": "manim",
         "desc": "Animated walkthrough of the full three-stage pipeline: supervised fine-tuning on TL;DR, reward model training from human comparisons, and PPO optimization with KL penalty.",
         "video": "RLHFTextPipeline.mp4"},

        {"type": "break"},

        # ══════════════════════════════════════════
        # METHOD
        # ══════════════════════════════════════════
        {"type": "note", "id": "method", "heading": "Method", "html": """
            <h3>High-Level Methodology</h3>
            <p>Start with an initial policy fine-tuned via SFT on the Reddit TL;DR summarization dataset. Then iterate through three stages:</p>
            <ol>
                <li><strong>Collect comparisons</strong> &mdash; For each Reddit post, sample summaries from the current policy, initial policy, original reference summary, and various baselines. Human evaluators choose the best summary from each batch.</li>
                <li><strong>Learn a reward model</strong> &mdash; Given a post and candidate summary, train a reward model to predict the log odds that this summary is the better one as judged by labelers.</li>
                <li><strong>Optimize the policy</strong> &mdash; Use the reward model's output as a reward signal, optimized via PPO.</li>
            </ol>

            <h3>Datasets and Task</h3>
            <ul>
                <li>The <strong>TL;DR</strong> summarization dataset has ~3 million posts from Reddit, filtered for quality to 123,169 posts (5% validation).</li>
                <li>Summaries must be 24&ndash;48 tokens.</li>
                <li><strong>CNN/DM</strong> is used as a transfer evaluation dataset &mdash; an "easy" dataset where good performance can be achieved readily.</li>
            </ul>

            <h3>Models</h3>
            <ul>
                <li>GPT-3-style decoders with <strong>1.3B</strong> and <strong>6.7B</strong> parameters.</li>
                <li><strong>Pretrained models</strong>: autoregressive next-token predictors used as zero-shot baselines.</li>
                <li><strong>Supervised baselines</strong>: fine-tuned via SFT with temperature 0 at evaluation time.</li>
            </ul>
        """},

        {"type": "break"},

        # ══════════════════════════════════════════
        # REWARD MODEL
        # ══════════════════════════════════════════
        {"type": "note", "id": "reward-model", "heading": "Reward Model Training", "html": """
            <p>Starting from the supervised baseline, a randomly initialized linear head is added that outputs a scalar value. The model is trained to predict which summary $y \\in \\{y_0, y_1\\}$ is better as judged by a human, given a post $x$.</p>

            <div class="formal-def">
                <p><strong>Bradley-Terry Loss</strong></p>
                <div class="math-block">
                    $$\\text{loss}(r_\\theta) = -\\mathbb{E}_{(x,y_0,y_1,i) \\sim \\mathcal{D}}\\left[\\log\\sigma\\!\\left(r_\\theta(x, y_i) - r_\\theta(x, y_{1-i})\\right)\\right]$$
                </div>
                <p>where $r_\\theta(x, y)$ is the scalar output of the reward model for post $x$ and summary $y$, $y_i$ is the human-preferred summary, and $\\sigma$ is the sigmoid function.</p>
            </div>

            <p>This is the <strong>Bradley-Terry model</strong>, as encountered in the original RLHF paper. It converts differences in reward scores into a probability that one summary is preferred over another. At the end of training, reward model outputs are normalized to have mean 0.</p>

            <div class="callout callout-insight">
                <p>The reward model doesn't need to produce calibrated scores &mdash; only the <em>relative ordering</em> matters. The sigmoid on the difference $r(x, y_i) - r(x, y_{1-i})$ makes this a binary classification problem: "which summary is better?" The model learns to assign higher scalar rewards to summaries humans prefer.</p>
            </div>
        """},

        # Viz 2
        {"type": "viz", "id": "viz-02", "num": 2, "title": "Bradley-Terry Reward Model",
         "tool": "manim",
         "desc": "Animated derivation of the Bradley-Terry loss: from pairwise human comparisons to the sigmoid-based training objective that turns scalar reward differences into preference probabilities.",
         "video": "BradleyTerryRM.mp4"},

        {"type": "break"},

        # ══════════════════════════════════════════
        # HUMAN FEEDBACK POLICIES
        # ══════════════════════════════════════════
        {"type": "note", "id": "human-feedback", "heading": "Human Feedback Policies", "html": """
            <p>The trained reward model is used to fine-tune a policy that generates higher-quality summaries. This is done via RL, treating the reward model output as a reward signal maximized with <strong>PPO</strong>. Each timestep corresponds to a BPE token.</p>

            <h3>KL-Penalized Reward</h3>
            <p>The full reward includes a KL divergence penalty between the RL policy $\\pi_\\phi^{\\text{RL}}$ and the original SFT model $\\pi^{\\text{SFT}}$:</p>

            <div class="formal-def">
                <div class="math-block">
                    $$R(x, y) = r_\\theta(x, y) - \\beta \\log\\!\\left[\\frac{\\pi_\\phi^{\\text{RL}}(y|x)}{\\pi^{\\text{SFT}}(y|x)}\\right]$$
                </div>
                <p>where $\\beta$ controls the strength of the KL penalty.</p>
            </div>

            <p>The KL penalty term serves two purposes:</p>
            <ul>
                <li><strong>Entropy bonus</strong> &mdash; encourages the policy to explore and prevents it from collapsing to a single mode.</li>
                <li><strong>Distribution anchoring</strong> &mdash; ensures the RL policy doesn't produce outputs too far from what the reward model saw during training, preventing reward hacking.</li>
            </ul>

            <h3>Implementation Details</h3>
            <ul>
                <li>The PPO <strong>value function</strong> uses a Transformer with <em>completely separate parameters</em> from the policy. This prevents value function updates from partially destroying the pretrained policy early in training.</li>
                <li>The value function is initialized from the reward model parameters.</li>
                <li>The reward model, policy, and value function are all the same size in the experiments.</li>
            </ul>

            <div class="callout callout-key">
                <p>The separate value function is a crucial design choice. In standard RL, sharing parameters between policy and value networks is common and efficient. But here, the policy starts as a carefully pretrained language model &mdash; value function gradients flowing into the policy network would corrupt the pretrained representations before the RL signal has time to steer them properly.</p>
            </div>
        """},

        # Viz 3
        {"type": "viz", "id": "viz-03", "num": 3, "title": "KL-Penalized Reward",
         "tool": "manim",
         "desc": "Animated visualization of the KL-penalized reward objective: how the reward model signal is balanced against the KL divergence penalty to prevent the policy from straying too far from the SFT baseline.",
         "video": "KLPenalizedReward.mp4"},

        {"type": "break"},

        # ══════════════════════════════════════════
        # RESULTS: REDDIT
        # ══════════════════════════════════════════
        {"type": "note", "id": "results-reddit", "heading": "Results: Summarizing Reddit Posts", "html": """
            <p>Policies trained with human feedback are preferred to much larger supervised policies. The 6.7B human feedback model produces summaries that humans prefer over the reference TL;DRs written by the original post authors.</p>

            <ul>
                <li>Slightly longer summaries tend to perform better &mdash; length is an implicit objective. When controlling for length explicitly, human feedback model performance drops by ~5% but still reaches 65% preference.</li>
                <li>Labelers assessed summary quality across four dimensions using a <strong>7-point Likert scale</strong>:
                    <ul>
                        <li><strong>Coverage</strong> &mdash; how much important information from the original post is covered</li>
                        <li><strong>Accuracy</strong> &mdash; to what degree the statements in the summary are supported by the post</li>
                        <li><strong>Coherence</strong> &mdash; how easy the summary is to read on its own</li>
                        <li><strong>Overall quality</strong></li>
                    </ul>
                </li>
            </ul>

            <div class="callout callout-key">
                <p>The RL-trained models improve on <em>all four dimensions</em> compared to supervised baselines, with the largest gains on coverage and overall quality. This confirms that the reward model captures more than just surface-level text features.</p>
            </div>
        """},

        # Viz 4
        {"type": "viz", "id": "viz-04", "num": 4, "title": "SFT vs RL Quality",
         "tool": "multi",
         "desc": "Comparison of supervised fine-tuning vs RL-trained model quality across evaluation dimensions, showing the consistent advantage of optimizing for human preferences.",
         "static_img": "04_sft_vs_rl_quality.png",
         "interactive": "04_sft_vs_rl_quality.html"},

        {"type": "break"},

        # ══════════════════════════════════════════
        # RESULTS: NEWS
        # ══════════════════════════════════════════
        {"type": "note", "id": "results-news", "heading": "Results: Summarizing News Articles", "html": """
            <p>Without any further training, the TL;DR-trained model generates excellent summaries of <strong>CNN/DM news articles</strong>. The summaries are consistently fluent and reasonable representations of the articles.</p>

            <div class="callout callout-insight">
                <p>This zero-shot transfer is remarkable: the model was trained exclusively on informal Reddit posts, yet it generalizes to formal news articles. Human feedback models transfer much better than supervised models, suggesting that learning <em>what humans value in a summary</em> generalizes across domains, while learning <em>the surface statistics of Reddit text</em> does not.</p>
            </div>
        """},

        # Viz 5
        {"type": "viz", "id": "viz-05", "num": 5, "title": "RM Accuracy Scaling",
         "tool": "multi",
         "desc": "How reward model accuracy scales with the amount of comparison data and model size, showing diminishing but consistent returns.",
         "static_img": "05_rm_accuracy_scaling.png",
         "interactive": "05_rm_accuracy_scaling.html"},

        {"type": "break"},

        # ══════════════════════════════════════════
        # UNDERSTANDING THE RM
        # ══════════════════════════════════════════
        {"type": "note", "id": "reward-understanding", "heading": "Understanding the Reward Model", "html": """
            <p>A reward model isn't a perfect representation of labeler preferences &mdash; it has limited capacity and only sees a small amount of comparison data from a relatively narrow distribution. The critical question: <em>how much can you optimize against the reward model before it starts giving useless evaluations?</em></p>

            <h3>Reward Overoptimization</h3>
            <p>The authors created a range of policies optimized against an earlier version of the reward model with varying degrees of optimization strength, then asked labelers to compare them.</p>
            <ul>
                <li>Under <strong>light optimization</strong>, the models genuinely improve.</li>
                <li>Under <strong>heavy optimization</strong>, the predicted reward keeps climbing but true human preferences fall off &mdash; eventually the reward model becomes <em>anti-correlated</em> with human preferences.</li>
            </ul>

            <div class="callout callout-key">
                <p>This is <strong>reward overoptimization</strong> (also called "reward hacking" or "Goodhart's Law in RL"). The policy learns to exploit quirks of the reward model rather than producing genuinely better summaries. This finding became foundational for alignment research &mdash; it demonstrates that proxy objectives break under heavy optimization, even when the proxy was trained on real human judgments.</p>
            </div>

            <h3>Reward Model Performance</h3>
            <ul>
                <li>The reward model agrees with labeler preferences around <strong>62&ndash;66%</strong> of the time.</li>
                <li>Inter-labeler agreement is <strong>66.9%</strong>, meaning the RM performs close to human-level consistency.</li>
                <li>The RM is sensitive to small but semantically important details in summaries.</li>
                <li>More data and larger model size yield marginal but consistent returns.</li>
            </ul>
        """},

        # Viz 6
        {"type": "viz", "id": "viz-06", "num": 6, "title": "Reward Overoptimization",
         "tool": "multi",
         "desc": "The critical reward overoptimization phenomenon: as KL divergence from the SFT policy increases, the reward model score keeps climbing but true human preference quality peaks and then degrades.",
         "static_img": "06_reward_overoptimization.png",
         "interactive": "06_reward_overoptimization.html"},

        # Viz 7
        {"type": "viz", "id": "viz-07", "num": 7, "title": "KL Coefficient Effect",
         "tool": "multi",
         "desc": "How the KL penalty coefficient beta controls the tradeoff between reward maximization and staying close to the SFT policy, with different beta values producing different quality profiles.",
         "static_img": "07_kl_coefficient_effect.png",
         "interactive": "07_kl_coefficient_effect.html"},

        {"type": "break"},

        # ══════════════════════════════════════════
        # AUTOMATIC METRICS
        # ══════════════════════════════════════════
        {"type": "note", "id": "automatic-metrics", "heading": "Analyzing Automatic Metrics", "html": """
            <p>The paper provides a thorough comparison of automatic metrics against human judgment:</p>

            <ul>
                <li><strong>Learned reward models</strong> consistently outperform other metrics, even on the CNN/DM dataset on which they were never trained.</li>
                <li><strong>ROUGE</strong> can't track sample quality as the model improves &mdash; it has low agreement with human evaluators.</li>
                <li>Optimizing ROUGE directly using a simple scheme doesn't actually increase summary quality as judged by humans.</li>
            </ul>

            <div class="callout callout-insight">
                <p>This is a strong indictment of ROUGE as an evaluation metric. The learned reward model, despite being trained on a relatively small dataset from one domain (Reddit), provides a better signal of summary quality than ROUGE even on out-of-distribution data (CNN/DM). The implication: <em>learning what humans value</em> gives you a more robust evaluation signal than hand-designed metrics.</p>
            </div>
        """},

        # Viz 8
        {"type": "viz", "id": "viz-08", "num": 8, "title": "ROUGE vs Learned RM",
         "tool": "multi",
         "desc": "Head-to-head comparison of ROUGE scores and learned reward model scores as predictors of human preference, demonstrating the superiority of learned evaluation.",
         "static_img": "08_rouge_vs_learned_rm.png",
         "interactive": "08_rouge_vs_learned_rm.html"},

        # Viz 9
        {"type": "viz", "id": "viz-09", "num": 9, "title": "Transfer: CNN/DM",
         "tool": "mpl",
         "desc": "Zero-shot transfer performance from the Reddit TL;DR-trained model to CNN/DM news summarization, showing that human feedback models generalize better than supervised baselines.",
         "static_img": "09_transfer_cnn_dm.png"},

        {"type": "break"},

        # ══════════════════════════════════════════
        # DISCUSSION
        # ══════════════════════════════════════════
        {"type": "note", "id": "discussion", "heading": "Discussion", "html": """
            <h3>Limitations</h3>
            <ul>
                <li>The time and cost required to produce the final models was substantial.</li>
                <li>Collecting human feedback is an expensive, long process that's hard to scale.</li>
                <li>The reward model is an imperfect proxy &mdash; reward overoptimization is a real concern.</li>
            </ul>

            <h3>Future Directions</h3>
            <ul>
                <li>The methods can be applied to <strong>any task where humans can compare samples</strong>: dialogue, machine translation, question answering, speech synthesis, music generation.</li>
                <li>Particularly important for <strong>long-form generation</strong> where distributional shift and degeneracy of maximum-likelihood samples are most problematic.</li>
                <li>Scaling human feedback to tasks where humans <em>can't easily evaluate</em> the quality of model outputs remains an open challenge.</li>
            </ul>

            <h3>Broader Impacts</h3>
            <ul>
                <li>The implications for <strong>aligning ML algorithms with designer preferences</strong> are broad.</li>
                <li>As models become more capable, it will be harder to spot mistakes and the consequences of errors become more severe.</li>
                <li>Large-scale models trained with human feedback could have significant societal impact, including job automation and potential for misuse.</li>
            </ul>

            <div class="callout callout-key">
                <p>This paper sits at a pivotal point in the alignment timeline: it proved that RLHF works for real NLP tasks, directly enabling <strong>InstructGPT</strong> (2022) and <strong>ChatGPT</strong>. The same pipeline &mdash; SFT &rarr; reward model &rarr; PPO &mdash; would become the standard recipe for aligning large language models. The reward overoptimization finding also seeded a major research direction: how to build robust reward signals that don't break under optimization pressure.</p>
            </div>
        """},

        # Viz 10
        {"type": "viz", "id": "viz-10", "num": 10, "title": "Alignment Timeline",
         "tool": "manim",
         "desc": "Animated timeline of the alignment research lineage: from the original RLHF paper (2017) through PPO (2017) to this paper (2020), and forward to InstructGPT (2022) and DPO (2023).",
         "video": "AlignmentTimeline.mp4"},

    ],  # end content

    # ─── Coding Adventures ───
    "adventures": [
        {
            "num": "02",
            "title": "KL Divergence: Implication on LLM Outputs",
            "status": "Done",
            "desc": "Load Qwen2.5-1.5B base + instruct side-by-side and see how KL divergence manifests in real token distributions. Per-token KL analysis, category comparison, and KL-constrained generation. 98 tests.",
            "url": "../../adventures/02/",
            "tags": ["Qwen2.5-1.5B", "Rich CLI", "KL Divergence", "Transformers"],
        },
    ],
}
