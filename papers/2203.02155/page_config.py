"""Page configuration for 2203.02155 — Training language models to follow instructions with human feedback (InstructGPT)."""

PAGE_DATA = {
    "title": "Training Language Models to Follow Instructions with Human Feedback",
    "arxiv_id": "2203.02155",
    "authors": "Ouyang, Wu, Jiang, Almeida, Wainwright, Mishkin, Zhang, Agarwal, Slama, Ray, Schulman, Hilton, Kelton, Miller, Simens, Askell, Welinder, Christiano, Leike, Lowe",
    "year": 2022,
    "last_updated": "May 2026",

    "tldr": """<p>InstructGPT applies the <strong>RLHF pipeline</strong> from the summarization paper (Stiennon et al. 2020) to the much broader problem of <strong>following arbitrary human instructions</strong>. The same three-step recipe &mdash; SFT on labeler demonstrations, reward model from K-way rankings, PPO fine-tuning &mdash; but with two key innovations: <strong>PPO-ptx</strong> (mixing pretraining gradients to prevent alignment tax) and <strong>batched K-way ranking</strong> (grouping C(K,2) pairs per prompt as a single gradient step to prevent overfitting). The headline result: a <strong>1.3B InstructGPT</strong> model is preferred by humans over the <strong>175B GPT-3</strong> baseline. InstructGPT also hallucinates less (21% vs 41%) and is less toxic under respectful prompts &mdash; though it becomes <em>more</em> toxic when explicitly prompted to be biased.</p>""",

    # ─── Table of Contents (sidebar) ───
    "toc": [
        {"id": "abstract",            "label": "Abstract",               "is_viz": False},
        {"id": "introduction",        "label": "Introduction",           "is_viz": False},
        {"id": "viz-01",              "label": "InstructGPT Pipeline",   "is_viz": True},
        {"id": "method",              "label": "Method",                 "is_viz": False},
        {"id": "ppo-ptx",             "label": "PPO-ptx Objective",      "is_viz": False},
        {"id": "viz-02",              "label": "PPO-ptx Objective",      "is_viz": True},
        {"id": "ranking",             "label": "K-way Ranking",          "is_viz": False},
        {"id": "viz-03",              "label": "K-way Batching",         "is_viz": True},
        {"id": "data",                "label": "Dataset & Tasks",        "is_viz": False},
        {"id": "viz-04",              "label": "Task Distribution",      "is_viz": True},
        {"id": "results",             "label": "Results",                "is_viz": False},
        {"id": "viz-05",              "label": "Winrate vs Baselines",   "is_viz": True},
        {"id": "truthfulness",        "label": "Truthfulness",           "is_viz": False},
        {"id": "viz-06",              "label": "TruthfulQA Results",     "is_viz": True},
        {"id": "toxicity",            "label": "Toxicity",               "is_viz": False},
        {"id": "viz-07",              "label": "Toxicity Paradox",       "is_viz": True},
        {"id": "alignment-tax",       "label": "Alignment Tax",          "is_viz": False},
        {"id": "viz-08",              "label": "Alignment Tax Fix",      "is_viz": True},
        {"id": "alignment-hierarchy", "label": "Who Are We Aligning To?","is_viz": False},
        {"id": "viz-09",              "label": "Alignment Hierarchy",    "is_viz": True},
        {"id": "discussion",          "label": "Discussion",             "is_viz": False},
        {"id": "viz-10",              "label": "Alignment Timeline",     "is_viz": True},
    ],

    # ─── Nav bar (mobile, abbreviated) ───
    "nav": [
        {"id": "abstract",            "label": "Abstract"},
        {"id": "introduction",        "label": "Intro"},
        {"id": "method",              "label": "Method"},
        {"id": "data",                "label": "Data"},
        {"id": "results",             "label": "Results"},
        {"id": "truthfulness",        "label": "Truth"},
        {"id": "toxicity",            "label": "Toxicity"},
        {"id": "alignment-tax",       "label": "Tax"},
        {"id": "discussion",          "label": "Discussion"},
    ],

    # ─── Viz gallery cards ───
    "viz_gallery": [
        {"num": 1,  "title": "InstructGPT Pipeline",     "tool": "manim", "tag_label": "Manim"},
        {"num": 2,  "title": "PPO-ptx Objective",        "tool": "manim", "tag_label": "Manim"},
        {"num": 3,  "title": "K-way Ranking Batching",   "tool": "manim", "tag_label": "Manim"},
        {"num": 4,  "title": "Task Distribution",        "tool": "multi", "tag_label": "MPL+Plotly"},
        {"num": 5,  "title": "Winrate vs Baselines",     "tool": "multi", "tag_label": "MPL+Plotly"},
        {"num": 6,  "title": "TruthfulQA Results",       "tool": "multi", "tag_label": "MPL+Plotly"},
        {"num": 7,  "title": "Toxicity Paradox",         "tool": "multi", "tag_label": "MPL+Plotly"},
        {"num": 8,  "title": "Alignment Tax & PPO-ptx",  "tool": "multi", "tag_label": "MPL+Plotly"},
        {"num": 9,  "title": "Alignment Hierarchy",      "tool": "mpl",   "tag_label": "Matplotlib"},
        {"num": 10, "title": "Alignment Timeline",       "tool": "manim", "tag_label": "Manim"},
    ],

    # ─── Content blocks (ordered) ───
    "content": [

        # ══════════════════════════════════════════
        # ABSTRACT
        # ══════════════════════════════════════════
        {"type": "note", "id": "abstract", "heading": "Abstract", "html": """
            <p>Making language models bigger does not inherently make them better at following user intent. Large models can generate outputs that are <strong>untruthful, toxic, or simply not helpful</strong> to the user. The authors show that by fine-tuning with human feedback, it's possible to align language models with user intent on a wide range of tasks.</p>
            <ul>
                <li>Start with a set of labeler-written prompts and prompts submitted through the OpenAI API.</li>
                <li>Collect a dataset of <strong>labeler demonstrations</strong> of desired behavior &rarr; supervised fine-tuning (SFT).</li>
                <li>Collect a dataset of <strong>labeler rankings</strong> of model outputs &rarr; train a reward model (RM).</li>
                <li>Use the reward model as a reward signal to fine-tune the SFT model with <strong>PPO</strong>.</li>
            </ul>
            <div class="callout callout-key">
                <p>The resulting InstructGPT models (1.3B parameters) are preferred to outputs from the 175B GPT-3, despite having 100x fewer parameters. InstructGPT shows improvements in truthfulness and reductions in toxic output generation.</p>
            </div>
        """},

        # ══════════════════════════════════════════
        # INTRODUCTION
        # ══════════════════════════════════════════
        {"type": "note", "id": "introduction", "heading": "Introduction", "html": """
            <p>Large language models can be "prompted" to perform a wide range of NLP tasks. However, these models often express <strong>unintended behaviors</strong>: making up facts, generating biased or toxic text, or simply not following the user's instruction. This is because the language modeling objective &mdash; predicting the next token &mdash; is <em>different from</em> the objective of "follow the user's instructions helpfully and safely."</p>

            <p>InstructGPT addresses this by training GPT-3 to act as an <strong>instruction-following assistant</strong> using the same RLHF pipeline established in Stiennon et al. (2020) for summarization, but applied to a much broader task distribution:</p>
            <ul>
                <li><strong>Generation</strong> (45.6%), <strong>Open QA</strong> (12.4%), <strong>Brainstorming</strong> (11.2%), Chat (8.4%), Rewrite (6.6%), Summarization (4.2%), and more.</li>
                <li>Prompts come from both hired labelers (who write diverse tasks) and real OpenAI API users.</li>
            </ul>

            <div class="callout callout-insight">
                <p>The key scaling insight: the same SFT &rarr; RM &rarr; PPO recipe that worked for summarization also works for the much harder problem of general instruction-following. The innovations are in the details: K-way ranking for more efficient labeling, PPO-ptx for preventing capability regression, and a much broader task distribution.</p>
            </div>
        """},

        # Viz 1
        {"type": "viz", "id": "viz-01", "num": 1, "title": "InstructGPT Pipeline",
         "tool": "manim",
         "desc": "Animated walkthrough of the 3-step InstructGPT pipeline: SFT on 13K labeler demonstrations, reward model training from K-way rankings (33K prompts), and PPO-ptx with pretraining gradient mixing.",
         "video": "InstructGPTPipeline.mp4"},

        {"type": "break"},

        # ══════════════════════════════════════════
        # METHOD
        # ══════════════════════════════════════════
        {"type": "note", "id": "method", "heading": "Method", "html": """
            <h3>Three-Step Pipeline</h3>
            <p>The methodology follows the same structure as the summarization paper, with important refinements:</p>
            <ol>
                <li><strong>Supervised Fine-Tuning (SFT)</strong> &mdash; Fine-tune GPT-3 on ~13K labeler demonstrations of desired assistant behavior. Train for 16 epochs with cosine learning rate decay and dropout of 0.2.</li>
                <li><strong>Reward Model (RM)</strong> &mdash; Train a 6B model to predict human preferences from ~33K prompts with K-way ranked outputs. Only 6B because 175B RM training was unstable.</li>
                <li><strong>PPO-ptx</strong> &mdash; Optimize the SFT model against the reward model using PPO, with an added pretraining gradient to prevent capability regression.</li>
            </ol>

            <h3>Model Sizes</h3>
            <ul>
                <li>SFT, RM, and RL models at <strong>1.3B, 6B, and 175B</strong> parameters.</li>
                <li>The reward model is always <strong>6B</strong> (175B training was unstable).</li>
                <li>SFT model is initialized from GPT-3 pretrained weights.</li>
            </ul>

            <div class="callout callout-key">
                <p>A crucial design choice: the SFT model <strong>overfits on validation loss after 1 epoch</strong> but continues to improve on human preference metrics when trained for 16 epochs. This mirrors a finding in the summarization paper and suggests that pure log-likelihood on demonstrations is a poor proxy for what humans actually want.</p>
            </div>
        """},

        {"type": "break"},

        # ══════════════════════════════════════════
        # PPO-PTX OBJECTIVE
        # ══════════════════════════════════════════
        {"type": "note", "id": "ppo-ptx", "heading": "The PPO-ptx Objective", "html": """
            <p>The standard PPO objective from the summarization paper includes a KL penalty against the SFT policy:</p>

            <div class="formal-def">
                <p><strong>Standard PPO Objective</strong></p>
                <div class="math-block">
                    $$\\text{objective}(\\phi) = \\mathbb{E}_{(x,y) \\sim D_{\\pi_\\phi^{RL}}}\\!\\left[r_\\theta(x,y) - \\beta\\log\\frac{\\pi_\\phi^{RL}(y|x)}{\\pi^{SFT}(y|x)}\\right]$$
                </div>
            </div>

            <p>InstructGPT adds a <strong>pretraining gradient</strong> to prevent the model from forgetting its general capabilities:</p>

            <div class="formal-def">
                <p><strong>PPO-ptx Objective</strong></p>
                <div class="math-block">
                    $$\\text{objective}(\\phi) = \\mathbb{E}\\!\\left[r_\\theta(x,y) - \\beta\\log\\frac{\\pi_\\phi^{RL}(y|x)}{\\pi^{SFT}(y|x)}\\right] + \\gamma\\,\\mathbb{E}_{x \\sim D_{\\text{pretrain}}}\\!\\left[\\log\\pi_\\phi^{RL}(x)\\right]$$
                </div>
                <p>where $\\beta = 0.02$ (per-token KL penalty) and $\\gamma = 27.8$ (pretraining mix coefficient).</p>
            </div>

            <div class="callout callout-insight">
                <p>The pretraining term $\\gamma \\cdot \\mathbb{E}[\\log\\pi(x)]$ is just the standard language modeling loss on pretraining data, mixed into each PPO update. With $\\gamma = 0$ (plain PPO), HellaSwag accuracy drops from 78.6 to 71.4. With $\\gamma = 27.8$ (PPO-ptx), it recovers to 78.8. This elegantly solves the alignment tax at the cost of slightly slower RL convergence.</p>
            </div>
        """},

        # Viz 2
        {"type": "viz", "id": "viz-02", "num": 2, "title": "PPO-ptx Objective",
         "tool": "manim",
         "desc": "Step-by-step animated derivation of the PPO-ptx objective: from standard RL reward to KL-penalized reward (from the summarization paper) to the final pretraining-mixed objective that prevents alignment tax.",
         "video": "PPOptxObjective.mp4"},

        {"type": "break"},

        # ══════════════════════════════════════════
        # K-WAY RANKING
        # ══════════════════════════════════════════
        {"type": "note", "id": "ranking", "heading": "K-way Ranking Strategy", "html": """
            <p>Instead of collecting pairwise comparisons (as in the summarization paper), InstructGPT has labelers <strong>rank K outputs</strong> (K=4 to 9) from best to worst for each prompt. This produces $\\binom{K}{2}$ comparison pairs per prompt.</p>

            <div class="formal-def">
                <p><strong>Batched RM Loss</strong></p>
                <div class="math-block">
                    $$\\text{loss}(\\theta) = -\\frac{1}{\\binom{K}{2}}\\sum_{(w,l)\\,:\\,y_w \\succ y_l}\\log\\sigma\\!\\left(r_\\theta(x,y_w) - r_\\theta(x,y_l)\\right)$$
                </div>
                <p>All $\\binom{K}{2}$ pairs from a single prompt are treated as <strong>one batch element</strong>, with a single forward pass per completion.</p>
            </div>

            <p>Why batching matters:</p>
            <ul>
                <li><strong>Naive approach</strong>: each pair is a separate data point &rarr; the same completion appears in multiple gradient steps &rarr; overfitting.</li>
                <li><strong>Batched approach</strong>: all pairs from one prompt grouped together &rarr; one forward pass per completion &rarr; much less overfitting.</li>
            </ul>

            <div class="callout callout-key">
                <p>The batched approach is both more <strong>computationally efficient</strong> (fewer forward passes) and <strong>statistically better</strong> (less overfitting). With K=9, you get $\\binom{9}{2}=36$ comparison pairs from a single labeler ranking session, making the 33K prompt dataset equivalent to ~660K comparison pairs.</p>
            </div>
        """},

        # Viz 3
        {"type": "viz", "id": "viz-03", "num": 3, "title": "K-way Ranking Batching",
         "tool": "manim",
         "desc": "Animated demonstration of how K outputs per prompt get ranked by labelers, producing C(K,2) comparison pairs. Contrasts the naive (overfitting) vs batched (efficient) training approaches.",
         "video": "KwayRankingBatching.mp4"},

        {"type": "break"},

        # ══════════════════════════════════════════
        # DATASET & TASKS
        # ══════════════════════════════════════════
        {"type": "note", "id": "data", "heading": "Dataset & Task Distribution", "html": """
            <p>The training data comes from two sources:</p>
            <ul>
                <li><strong>Labeler-written prompts</strong>: ~40 contractors write diverse tasks spanning generation, QA, brainstorming, chat, rewriting, summarization, classification, and extraction.</li>
                <li><strong>Customer API prompts</strong>: real prompts from OpenAI API users (with PII filtering).</li>
            </ul>

            <h3>Dataset Sizes (Table 6)</h3>
            <ul>
                <li><strong>SFT</strong>: ~13K demonstrations (mostly labeler-written)</li>
                <li><strong>RM</strong>: ~33K prompts with K-way rankings (mostly customer API)</li>
                <li><strong>PPO</strong>: ~31K prompts used as RL training distribution (all customer API)</li>
            </ul>

            <p>The task distribution (Table 1) is heavily skewed toward <strong>generation</strong> (45.6%), followed by open QA (12.4%), brainstorming (11.2%), and chat (8.4%).</p>

            <div class="callout callout-insight">
                <p>The shift from labeler-written to customer API prompts as you move from SFT to RM to PPO is deliberate: SFT needs high-quality demonstrations (labeler expertise), while RM and PPO benefit from covering the <em>real distribution</em> of user queries. This ensures the final model is aligned to what actual users ask, not just what labelers imagine users might ask.</p>
            </div>
        """},

        # Viz 4
        {"type": "viz", "id": "viz-04", "num": 4, "title": "Task Distribution & Dataset",
         "tool": "multi",
         "desc": "Two-panel visualization: donut chart of API prompt categories (Table 1) and stacked bar chart of dataset sizes by source (Table 6), showing the shift from labeler to customer data across pipeline stages.",
         "static_img": "04_task_distribution_dataset.png",
         "interactive": "04_task_distribution_dataset.html"},

        {"type": "break"},

        # ══════════════════════════════════════════
        # RESULTS
        # ══════════════════════════════════════════
        {"type": "note", "id": "results", "heading": "Results: Human Preference", "html": """
            <p>The core evaluation uses human labelers rating model outputs on the API prompt distribution. The headline findings:</p>

            <ul>
                <li><strong>1.3B InstructGPT is preferred over 175B GPT-3</strong> &mdash; despite having 100x fewer parameters.</li>
                <li>PPO and PPO-ptx models at all sizes significantly outperform their GPT-3 and SFT counterparts.</li>
                <li>Labelers find InstructGPT outputs more <strong>helpful, truthful, and harmless</strong>.</li>
                <li>InstructGPT <strong>hallucinates less</strong>: closed-domain hallucination drops from 41% (GPT-3) to 21% (PPO-ptx).</li>
            </ul>

            <div class="callout callout-key">
                <p>The 1.3B &gt; 175B result is the paper's most striking finding. It demonstrates that <strong>alignment is more important than scale</strong> for producing outputs humans actually prefer. A well-aligned small model beats a much larger but misaligned one.</p>
            </div>
        """},

        # Viz 5
        {"type": "viz", "id": "viz-05", "num": 5, "title": "Winrate vs Baselines",
         "tool": "multi",
         "desc": "Grouped bar chart reproducing Figure 1: winrate against 175B SFT for all model variants. The headline: 1.3B PPO-ptx preferred over 175B GPT-3.",
         "static_img": "05_winrate_vs_baselines.png",
         "interactive": "05_winrate_vs_baselines.html"},

        {"type": "break"},

        # ══════════════════════════════════════════
        # TRUTHFULNESS
        # ══════════════════════════════════════════
        {"type": "note", "id": "truthfulness", "heading": "Truthfulness: TruthfulQA", "html": """
            <p>InstructGPT shows significant improvements on the <strong>TruthfulQA</strong> benchmark, which tests whether models generate truthful answers to questions that humans might answer incorrectly due to common misconceptions.</p>

            <ul>
                <li>PPO-ptx 175B is truthful on <strong>50%</strong> of questions vs 22% for GPT-3 175B (&sim;2.3x improvement).</li>
                <li>For "truthful <em>and</em> informative" (not just declining to answer), PPO-ptx achieves <strong>41%</strong> vs 19% for GPT-3.</li>
                <li>Interestingly, smaller GPT-3 models are <em>more</em> truthful than larger ones &mdash; the "inverse scaling" phenomenon. InstructGPT reverses this trend.</li>
            </ul>

            <div class="callout callout-insight">
                <p>The inverse scaling on truthfulness for GPT-3 suggests that larger models are better at reproducing convincing-sounding but false claims from the training data. RLHF counteracts this by teaching the model that humans prefer honest "I don't know" responses over confident misinformation.</p>
            </div>
        """},

        # Viz 6
        {"type": "viz", "id": "viz-06", "num": 6, "title": "TruthfulQA Results",
         "tool": "multi",
         "desc": "Grouped bar chart of truthful and truthful+informative fractions across GPT-3/SFT/PPO/PPO-ptx at all sizes. Toggle between 175B-only and all-sizes views.",
         "static_img": "06_truthfulqa_results.png",
         "interactive": "06_truthfulqa_results.html"},

        {"type": "break"},

        # ══════════════════════════════════════════
        # TOXICITY
        # ══════════════════════════════════════════
        {"type": "note", "id": "toxicity", "heading": "Toxicity: The Paradox", "html": """
            <p>Toxicity evaluation uses the Perspective API on RealToxicityPrompts. The results reveal a <strong>paradox</strong>:</p>

            <ul>
                <li>With <strong>respectful prompts</strong>: InstructGPT is significantly <em>less</em> toxic than GPT-3.</li>
                <li>With <strong>standard prompts</strong>: InstructGPT is moderately less toxic.</li>
                <li>With <strong>biased prompts</strong> (instructing toxicity): InstructGPT is <em>more</em> toxic than GPT-3.</li>
            </ul>

            <div class="callout callout-key">
                <p>InstructGPT is better at <strong>following instructions</strong> &mdash; including harmful ones. When told to be biased, it's more effective at it than GPT-3. This highlights a fundamental tension in RLHF alignment: you can't just teach a model to follow instructions; you also need to teach it <em>which instructions to refuse</em>. The paper notes this as a significant limitation and area for future work.</p>
            </div>
        """},

        # Viz 7
        {"type": "viz", "id": "viz-07", "num": 7, "title": "Toxicity Paradox",
         "tool": "multi",
         "desc": "Three prompt conditions (respectful/standard/biased) showing the toxicity paradox: InstructGPT less toxic when asked to be respectful, but MORE toxic when prompted to be biased.",
         "static_img": "07_toxicity_paradox.png",
         "interactive": "07_toxicity_paradox.html"},

        {"type": "break"},

        # ══════════════════════════════════════════
        # ALIGNMENT TAX
        # ══════════════════════════════════════════
        {"type": "note", "id": "alignment-tax", "heading": "The Alignment Tax", "html": """
            <p>A major concern with RLHF: does aligning the model to human preferences <strong>degrade its general capabilities</strong>? This "alignment tax" is measured on standard NLP benchmarks:</p>

            <ul>
                <li><strong>PPO (no pretraining mix)</strong>: significant regression on HellaSwag (78.6&rarr;71.4), SQuAD (69.0&rarr;49.2), DROP (36.7&rarr;24.0), WMT FR&rarr;EN (32.6&rarr;18.4).</li>
                <li><strong>PPO-ptx (with pretraining mix)</strong>: near-full recovery on all benchmarks &mdash; HellaSwag 78.8, SQuAD 65.8, DROP 36.5, FR&rarr;EN 33.8.</li>
            </ul>

            <div class="callout callout-key">
                <p>PPO-ptx essentially eliminates the alignment tax. By mixing pretraining gradients ($\\gamma = 27.8$) into each PPO step, the model retains its broad language understanding while still learning to follow instructions. The trade-off is slightly slower RL convergence, but the capability preservation is worth it.</p>
            </div>
        """},

        # Viz 8
        {"type": "viz", "id": "viz-08", "num": 8, "title": "Alignment Tax & PPO-ptx Fix",
         "tool": "multi",
         "desc": "Multi-panel comparison of 4 NLP benchmarks (HellaSwag, SQuAD, DROP, FR→EN) across GPT-3/SFT/PPO/PPO-ptx at 175B. PPO regresses on all; PPO-ptx recovers.",
         "static_img": "08_alignment_tax_ppoptx.png",
         "interactive": "08_alignment_tax_ppoptx.html"},

        {"type": "break"},

        # ══════════════════════════════════════════
        # ALIGNMENT HIERARCHY
        # ══════════════════════════════════════════
        {"type": "note", "id": "alignment-hierarchy", "heading": "Who Are We Aligning To?", "html": """
            <p>The paper is remarkably honest about a fundamental question: <em>whose preferences are we actually training on?</em> There is a four-layer hierarchy:</p>

            <ol>
                <li><strong>Labelers</strong> (~40 contractors) &mdash; provide the actual training signal. English-speaking, screened for agreement with researchers, skewed demographics.</li>
                <li><strong>Researchers</strong> &mdash; define the task, write labeling instructions, choose the labeler pool, design the reward criteria.</li>
                <li><strong>Customers</strong> &mdash; API users (developers) whose prompts shape the RL training distribution. Business objectives may not align with end user interests.</li>
                <li><strong>End Users</strong> &mdash; the people actually affected by model outputs. The most diverse group, but they have <strong>no direct voice</strong> in the training process.</li>
            </ol>

            <div class="callout callout-insight">
                <p>The training signal flows inward: end users &rarr; customers &rarr; researchers &rarr; labelers &rarr; model. At each layer, preferences are filtered and potentially distorted. The model is ultimately aligned to <em>labeler</em> preferences, not end user preferences. This is a structural limitation that no amount of engineering can fully resolve without broader participation in the alignment process.</p>
            </div>
        """},

        # Viz 9
        {"type": "viz", "id": "viz-09", "num": 9, "title": "Alignment Hierarchy",
         "tool": "mpl",
         "desc": "Static infographic showing the four-layer alignment hierarchy (labelers → researchers → customers → end users) with annotations about biases and limitations at each layer.",
         "static_img": "09_who_are_we_aligning_to.png"},

        {"type": "break"},

        # ══════════════════════════════════════════
        # DISCUSSION
        # ══════════════════════════════════════════
        {"type": "note", "id": "discussion", "heading": "Discussion", "html": """
            <h3>Key Contributions</h3>
            <ul>
                <li><strong>PPO-ptx</strong>: mixing pretraining gradients into PPO eliminates the alignment tax.</li>
                <li><strong>K-way batched ranking</strong>: more efficient labeling and less overfitting.</li>
                <li><strong>Scale validation</strong>: the SFT &rarr; RM &rarr; PPO pipeline works for broad instruction-following, not just summarization.</li>
                <li><strong>Honest limitations analysis</strong>: the alignment hierarchy, toxicity paradox, and labeler demographics are discussed openly.</li>
            </ul>

            <h3>Limitations</h3>
            <ul>
                <li>The model follows harmful instructions more effectively than GPT-3 (the toxicity paradox).</li>
                <li>Alignment to ~40 English-speaking labelers, not the global population of users.</li>
                <li>Reward model can still be overoptimized (same concern as Stiennon et al.).</li>
                <li>The SFT model overfits on validation loss but improves on human metrics &mdash; we still don't fully understand this phenomenon.</li>
            </ul>

            <h3>Historical Significance</h3>

            <div class="callout callout-key">
                <p>InstructGPT is the direct ancestor of <strong>ChatGPT</strong>. The same RLHF methodology, applied to a dialogue-optimized model, produced ChatGPT in November 2022 &mdash; arguably the most impactful AI product launch in history. Every major LLM alignment effort since (Claude, Gemini, Llama-2-Chat) uses a pipeline descended from this paper. The lineage: RLHF (2017) &rarr; PPO (2017) &rarr; Summarize from HF (2020) &rarr; <strong>InstructGPT (2022)</strong> &rarr; ChatGPT &rarr; DPO (2023).</p>
            </div>
        """},

        # Viz 10
        {"type": "viz", "id": "viz-10", "num": 10, "title": "Alignment Timeline",
         "tool": "manim",
         "desc": "Animated timeline of the alignment research lineage, centered on InstructGPT's position as the bridge from academic RLHF research to the ChatGPT product launch.",
         "video": "AlignmentTimeline.mp4"},

    ],  # end content

    # ─── Coding Adventures ───
    "adventures": [
        {
            "num": "04",
            "title": "System Prompt Steering",
            "status": "Done",
            "desc": "Measure how system prompts steer token distributions using KL divergence. 7 system prompts x 7 user prompts, first-token analysis, forced-continuation KL profiles, and a full steering matrix. 128 tests.",
            "url": "../../adventures/04/",
            "tags": ["Qwen2.5-1.5B", "Rich CLI", "128 Tests"],
        },
    ],
}
