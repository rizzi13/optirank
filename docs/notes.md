# 📒 OptiRank — Concept Guides & Interview Q&A

> Each section covers:
> 1. 🧠 **The Core Concept** — what it is, how it works, and why it exists.
> 2. 🎯 **The Production Problem It Solves** — why companies use it in real life.
> 3. 🎤 **Interview Questions & Answers** — exact questions interviewers ask on this specific concept + punchy answers to give verbally.

---

## 1. Multi-Stage Ranking Cascade (The Funnel Architecture)

### 💡 Intuitive Analogy: The Audition Funnel
Imagine you are a judge on a TV talent show. 10,000 contestants show up:
- **Round 1 (Eligibility Filter, < 0.2 ms):** A bouncer at the door checks: 'Are you registered? Do you have an ID?' Drops 9,800 people instantly without judging.
- **Round 2 (Stage 1 Pre-Ranker, ~1.5 ms):** An assistant listens to the remaining 200 singers for 5 seconds each. Their only job is to cut the bottom 180 and pass the **Top 20** to the celebrity judges.
- **Round 3 (Stage 2 Deep Ranker, ~5–10 ms):** The celebrity judges listen closely to the 20 finalists and award fine-grained scores.
- **Round 4 (Auction & Calibration):** Combine the judge's score with sponsor bids to pick the winning ad shown on TV.

If the celebrity judges listened to all 10,000 contestants for 10 minutes each, the show would take 3 years to air and bankrupt the TV studio. That is why high-volume recommendation systems use a multi-stage funnel.

### 🧠 The Core Concept
Instead of evaluating one massive AI model on every candidate item, modern recommendation and ad systems use a **multi-stage funnel**:
1. **Eligibility Rules Layer (`< 0.2 ms`):** Drops invalid candidates instantly (e.g., user already saw this ad 10 times today, or ad size doesn't match the banner slot).
2. **Stage 1 — Pre-Ranker (`~1.5 ms`):** Evaluates a lightweight model (such as `LightGBM` or a distilled tiny MLP) over all ~160 eligible candidates. Its job is **not** to pick the single final winner—its job is to discard the bottom 85% of irrelevant ads with high recall and pass the **Top $K$ ($K=20$)** to Stage 2.
3. **Stage 2 — Deep Ranker (`~5–10 ms`):** Runs a deep neural network (Deep & Cross Network, DCN-v2) on GPU FP16 or CPU INT8 over *only* those $K$ candidates to score subtle non-linear and high-order feature interactions.
4. **Calibration & Auction Layer:** Converts raw model scores into true calibrated click probabilities ($\text{pCTR}$) and multiplies by advertiser bids to select the winning ad by expected revenue ($\text{eCPM} = \text{pCTR} \times \text{Bid}$).

### 🎯 The Production Problem It Solves
- **Strict Latency SLA:** Web and mobile apps require ad responses within **~25–50 milliseconds**. Evaluating a deep neural network on 200 items takes 50–100 ms, which violates the SLA and slows down page loading.
- **Compute & Cloud Cost:** Scoring 200 items on GPU for every single pageview costs thousands of dollars a day in server bills. Pre-ranking cuts GPU compute by **over 70%** while preserving 95%+ of the heavy model's top selections.

### 🎤 Interview Questions on This Specific Concept:
> **Q1: "Why use a two-stage cascade instead of just making the deep neural network smaller?"**  
> **A:** *"Even an optimized neural network has significant latency when evaluated across 200 items per query. A tree-based model like LightGBM or a distilled MLP can evaluate 200 items in under 1.5 ms on CPU. By pairing a fast coarse filter with a deep fine ranker, we get the latency and cost of a small model with the ranking quality of a deep model."*

> **Q2: "What evaluation metric do you use to measure whether Stage 1 is doing a good job?"**  
> **A:** *"We use **Consistency Recall@$K$** (or Overlap@$K$). Out of the top 5 ads that the heavy Stage-2 model would choose if it evaluated all 200 candidates, what percentage was successfully included in Stage 1's top-$K$ shortlist? We aim for Recall@20 $\ge 95\%$."*

---

## 2. Chronological (Time-Based) Splitting vs. Random Splitting

### 💡 Intuitive Analogy: Studying for Tomorrow's Exam vs Time-Traveling Cheating
Imagine taking a history exam:
- **Real Life (Chronological):** You study history up to the year 2020. On the exam, the teacher asks you to predict what happens in 2021. You have to genuinely understand the underlying principles of politics and human behavior.
- **Cheating via Time Travel (Random Shuffle):** Someone shreds history books from 1900 to 2024 into random sentences. 80% go into your study deck, 20% go on your exam. If your study deck has: 'On Oct 28 at 2:05 PM, Nike launched a 50% flash sale', and your exam asks: 'Did someone buy Nike on Oct 28 at 2:00 PM?', you don't have to understand economics—you just memorize that Oct 28 was Nike sale day!
When you deploy to production on Nov 1, the sale is over, and your model fails completely. That is why time-series data must NEVER be shuffled randomly.

### 🧠 The Core Concept
When working with tabular data, standard data science tutorials often use `train_test_split(test_size=0.2, shuffle=True)` or 5-fold cross-validation.  
In time-series, financial transactions, and ad-tech, **random splitting is completely forbidden**.  
Instead, data is split **strictly chronologically**:
- **Train Split:** October 21 to October 28 (8 days of history)
- **Validation Split:** October 29 (day 9, used for tuning hyperparameters, calibration, and cascade thresholds)
- **Test Split:** October 30 (day 10, strictly held out to simulate "tomorrow's live incoming traffic")

### 🎯 The Production Problem It Solves
- **Temporal Data Leakage:** In online advertising, user engagement, click trends, and advertiser budgets change continuously across hours and days. If you randomly shuffle, an impression from Oct 28 at 2:05 PM is in the training set while an impression from Oct 28 at 2:00 PM for the exact same user/ad is in the test set. The model simply memorizes that day's active campaign performance, leading to artificially inflated offline test metrics that fail completely when deployed to production.

### 🎤 Interview Questions on This Specific Concept:
> **Q1: "Why did you use a chronological split instead of K-fold cross validation?"**  
> **A:** *"Ad logs have non-stationary distributions—click patterns shift hour-by-hour and campaign-by-campaign. Shuffling data causes temporal leakage where future events leak into the training set. A strict time-based split (Days 1–8 for train, Day 9 for validation, Day 10 for test) accurately mirrors production where the model must predict future unseen hours."*

> **Q2: "What is the difference in role between your validation day (Oct 29) and your test day (Oct 30)?"**  
> **A:** *"We touch the validation set repeatedly during development to tune LightGBM early stopping, fit isotonic calibration curves, and tune the cascade threshold $K$. Because these decisions optimize for Day 9, it develops subtle selection bias. Day 10 is kept in a vault and only touched once at the end to get an unbiased estimate of true generalization on future traffic."*

---

## 3. Negative Downsampling & Prior Shift Correction

### 💡 Intuitive Analogy: The Rare Disease & The Overconfident Doctor
Imagine a rare disease that only 1 in 10,000 people have.
If a medical student looks at 1,000,000 healthy x-rays, they fall asleep and learn nothing because 99.99% are negative.
So the professor creates a special training deck: they keep all 100 sick x-rays, but keep only 900 healthy ones. Now the study deck is 10% sick instead of 0.01%!
The student learns the signs fast! BUT there is a dangerous side-effect:
When the student graduates and enters a real hospital, they think *every 10th patient* has the rare disease! They are terrified and over-diagnosing everyone!
To fix this, the hospital gives the student a correction formula (Bayes' Odds Correction): 'Remember, in the real world, healthy people are 100 times more common than in your study deck. Scale your suspicion back down before writing the prescription!'

### 🧠 The Core Concept
Click prediction is inherently imbalanced: ~83% of impressions are non-clicks (`0`) and ~17% are clicks (`1`).  
To train models quickly within a 16 GB RAM budget, we perform **Negative Downsampling**:
- Keep **100% of all clicks (`click = 1`)**.
- Keep only a fraction $w$ (e.g. $w = 0.25$, or 25%) of non-clicks (`click = 0`).

**The Catch (Prior Shift):** Because 75% of non-clicks were discarded during training, the model is trained on a synthetic dataset with ~45% positive rate instead of 17%. The raw output scores $\hat{p}$ are systematically inflated.

**The Mathematical Fix:** Before using $\hat{p}$ in an auction, we apply Bayes' analytical prior correction formula to recover the true probability $\hat{q}$:
$$\hat{q} = \frac{\hat{p}}{\hat{p} + \frac{1 - \hat{p}}{w}}$$

### 🎯 The Production Problem It Solves
- **Memory & Training Speed:** Discarding redundant non-clicks reduces training volume by over 60%, allowing 8 days of data to fit smoothly in RAM and train in minutes rather than hours.
- **Accurate Ad Pricing:** In an ad auction, an advertiser pays based on expected revenue: $\text{eCPM} = \text{pCTR} \times \text{Bid}$. If predicted CTR is uncalibrated and 2× too high, the platform overprices bids, exhausts advertiser budgets prematurely, and loses business trust.

### 🎤 Interview Questions on This Specific Concept:
> **Q1: "If you downsample non-clicks during training, how do you prevent your model from over-predicting click probabilities in production?"**  
> **A:** *"Downsampling by a factor $w=0.25$ shifts the class prior. We apply the analytical odds correction $\hat{q} = \frac{\hat{p}}{\hat{p} + (1 - \hat{p})/w}$ to transform the raw model score $\hat{p}$ back into the unbiased real-world probability $\hat{q}$. We then verify that our Calibration Ratio ($\frac{\text{mean predicted CTR}}{\text{observed CTR}}$) on the untouched validation set equals $1.00$."*

> **Q2: "Did you downsample the validation and test sets as well?"**  
> **A:** *"No. Validation and test sets must never be downsampled. They must reflect the true, unadulterated real-world distribution (17% CTR) to accurately evaluate calibration and LogLoss."*

---

## 4. Entity Resolution & The Composite User Proxy

### 🧠 The Core Concept
To predict clicks accurately, models need user history features (e.g. *"how many ads did this user see in the last 1 hour?"* or *"did this user already see this Nike ad twice today?"*).  
In the Avazu dataset, over **82% of rows** contain `device_id = "a99f214a"`, which is a placeholder indicating that the user's real device ID was masked or blocked by mobile privacy settings.

**The Fix (Composite User Proxy):**
$$\text{user\_key} = \begin{cases} \text{device\_id} & \text{if } \text{device\_id} \neq \text{"a99f214a"} \\ \text{device\_ip} + \text{"\_"} + \text{device\_model} & \text{otherwise} \end{cases}$$

### 🎯 The Production Problem It Solves
If an engineer blindly groups rows by `device_id`, 82% of all global users get merged into one massive, fake super-user with millions of impressions an hour. Rolling velocity features and frequency caps become completely corrupted. Building a fallback composite key (`IP + Model`) separates individual user sessions cleanly.

### 🎤 Interview Questions on This Specific Concept:
> **Q1: "How did you construct user-level frequency and history features when the dataset had missing user IDs for 80%+ of rows?"**  
> **A:** *"We performed entity resolution via a composite User Proxy: when `device_id` was the placeholder `'a99f214a'`, we fell back to a joint fingerprint of `device_ip` and `device_model`. This segmented the anonymous traffic into realistic device-level session streams, allowing us to compute point-in-time ad fatigue and hourly velocity features without cross-contaminating different users."*

---

## 5. Chunked Streaming, Explicit `dtypes`, and Daily Parquet Partitioning

### 🧠 The Core Concept
Avazu's raw `train.gz` file contains **40,428,967 rows** (~6 GB uncompressed CSV).
Instead of loading the full file in one call, our data pipeline uses three techniques:
1. **Compact `dtypes`:** Overriding pandas' default 64-bit storage (`int64` = 8 bytes) with minimum-size types:
   - `click`, `banner_pos`, `device_type` $\rightarrow$ `int8` (1 byte, values $-128$ to $+127$)
   - `C15` (width = 320), `C16` (height = 50), `C17` $\rightarrow$ `int16` (2 bytes, up to $+32,767$)
   - `hour`, `C14` $\rightarrow$ `int32` (4 bytes, up to $+2$ billion)
2. **Chunked Streaming (`chunksize=1_000_000`):** Reads only 1 million rows into RAM at a time as a Python iterator.
3. **Daily Partitioned Parquet:** Slices rows by day (`hour // 100`) and flushes completed days to disk as compressed `.parquet` files (`day_141021.parquet` .. `day_141030.parquet`).

### 🎯 The Production Problem It Solves
- **Preventing OOM (Out-of-Memory) Crashes:** Loading a 6 GB CSV with default 64-bit types consumes >14 GB of RAM, freezing standard laptops. Compact types cut memory by **~65%**, and chunked streaming caps peak RAM at **under 1.5 GB**.
- **10× Faster Iteration:** CSV is uncompressed plain text and row-oriented; reading even 2 columns requires parsing all 24 columns across 40M rows (~30 seconds). Parquet is binary and columnar; reading a single day's specific columns takes **~1 second**.

### 🎤 Interview Questions on This Specific Concept:
> **Q1: "Why use Parquet instead of CSV for storing intermediate feature data?"**  
> **A:** *"Parquet is a columnar binary format with snappy compression. It has two huge advantages over CSV: first, it preserves exact memory dtypes (like `int8`) upon reload without re-parsing; second, because it is columnar, downstream pipelines can read only the 5 feature columns they need without scanning the other 19 columns, reducing I/O and load times by over 10×."*

> **Q2: "How did you choose the integer data types for each column in your schema?"**  
> **A:** *"We inspected the minimum and maximum values of each attribute: binary targets and small positions ($0..7$) fit in `int8` ($-128$ to $+127$), banner dimensions and campaign IDs ($<32,767$) fit in `int16`, and large IDs and timestamps ($14,000,000$) fit in `int32`. Using the smallest sufficient integer width cut in-memory memory footprint by 65% compared to pandas' default `int64`."*

---

## 6. Reinforcement Learning & Contextual Multi-Armed Bandits (Exploration vs. Exploitation)

### 🧠 The Core Concept
Supervised Machine Learning models are **greedy**: they only want to show ads that already have proven high historical click rates (**Exploitation**).  
However, when an advertiser launches a brand-new ad creative (**Cold Start**), the model has zero historical data and predicts a low default score, so the new ad is never shown and never gets a chance to gather data.

**The Reinforcement Learning Solution (Contextual Multi-Armed Bandits / LinUCB / Thompson Sampling):**
We frame ad selection as a Contextual Bandit:
- **Agent:** The ranking engine.
- **Context ($x$):** The user's device, app, and time of day.
- **Arm / Action ($a$):** Which ad to display from the top-$K$.
- **Reward ($r$):** $+1$ if the user clicks, $0$ if they do not.

Instead of ranking purely by predicted $\text{pCTR}$, the bandit scores ads using **Upper Confidence Bound (UCB)**:
$$\text{Bandit Score} = \Big(\text{pCTR} + \alpha \cdot \sqrt{\text{Uncertainty Variance}}\Big) \times \text{Bid}$$
- For **established ads** seen millions of times, uncertainty is near $0$ $\rightarrow$ ranked strictly by predicted CTR.
- For **brand-new ads**, uncertainty is large $\rightarrow$ they receive an **Exploration Bonus** allowing them to be shown to real users.
- Once feedback arrives (click or no click), the bandit updates its covariance matrix online in real time.

### 🎯 The Production Problem It Solves
- **The Cold-Start Death Spiral:** In a purely greedy supervised system, new ad campaigns starve and cannot ramp up, which frustrates advertisers who spend money on new creatives. Contextual Bandits mathematically solve the exploration vs. exploitation dilemma with sub-millisecond overhead.

### 🎤 Interview Questions on This Specific Concept:
> **Q1: "Why did you use a Contextual Multi-Armed Bandit instead of a full Deep Reinforcement Learning algorithm like DQN or PPO?"**  
> **A:** *"Full Deep RL (DQN, PPO) models multi-step state transitions where an action today changes the environment state tomorrow (like playing chess or robotics). In real-time ad serving, an ad impression is a single-step decision with an immediate reward (click/no click), with a strict 30 ms SLA. A Contextual Bandit (such as LinUCB or Thompson Sampling) is the industry standard because it provides provable sub-linear regret bounds, updates online with closed-form matrix math, and executes in under 0.5 milliseconds."*

> **Q2: "How do you control how aggressively the system explores new ads versus exploiting known top performers?"**  
> **A:** *"We tune the exploration hyperparameter $\alpha$ in the UCB formula $\hat{p} + \alpha \cdot \sigma$. Setting $\alpha = 0$ reduces the policy to purely greedy exploitation. By tuning $\alpha$ (typically $0.1$ to $0.5$), we guarantee that high-uncertainty cold-start ads receive a controlled exploration budget without degrading overall platform CTR and revenue."*

---

## 7. Python Entry Points: `if __name__ == "__main__":`

### 🧠 The Core Concept
Every Python file has a built-in hidden variable named `__name__`. Python sets this variable automatically depending on **how** the file was executed:
- **Direct Terminal Execution (`python myfile.py` or `python -m package.module`):** Python assigns `__name__ = "__main__"`.
- **Imported as a Library (`import myfile` or `from package.module import helper_func`):** Python assigns `__name__ = "package.module"` (the module's actual import path).

The `if __name__ == "__main__":` block is a guard:
```python
def my_reusable_function():
    return 42

if __name__ == "__main__":
    # This code executes ONLY when run directly from the terminal
    print("Running script directly!")
```

### 🎯 The Production Problem It Solves
- **Preventing Unintended Side Effects on Import:** In production ML systems, files contain both reusable utility definitions (e.g. data type schemas, mathematical functions, metric calculators) and heavy executable logic (e.g. converting 40M rows, downloading datasets, running database migrations).
- **Safe Unit Testing:** Test frameworks like `pytest` import modules to test individual functions. Without this guard, importing a schema function in a test file would accidentally trigger a 3-minute, 40-million row data processing job, freezing the CI/CD pipeline.
- **Clean CLI Tooling:** It allows the exact same `.py` file to serve as both a reusable Python package module and a standalone CLI command-line utility.

### 🎤 Interview Questions on This Specific Concept:
> **Q1: "What does `if __name__ == '__main__':` do under the hood, and why is it mandatory in production Python codebases?"**  
> **A:** *"Under the hood, Python sets the special variable `__name__` to the string `'__main__'` only for the top-level script directly invoked by the interpreter. If the file is imported by another module, `__name__` is set to its module path instead. In production, this block is mandatory because it decouples **definition** from **execution**. It prevents heavy execution side effects—like starting a training run, reading a 40M row file, or establishing database connections—from firing when another module or test runner simply imports a helper function."*

> **Q2: "What is the difference between running a script with `python script.py` versus `python -m package.script`?"**  
> **A:** *"Running with `python script.py` adds the directory of `script.py` to `sys.path`, which can break relative imports if the project is organized into sub-packages. Running with `python -m package.script` searches `sys.path` for the module within the current working directory or installed package structure, ensuring that absolute and relative imports across the package (like `optirank.config`) resolve correctly without path manipulation."*

---

## 8. Exploratory Data Analysis (EDA) in Ad-Tech: The 4 Core Realities

### 🧠 The Core Concept
In generic tabular ML, EDA is often treated as simple summary statistics (`df.describe()`).  
In large-scale ad systems, EDA is designed to uncover **structural realities of digital advertising**:

1. **Class Imbalance & Base CTR (~17%):**
   - ~83% of impressions generate no click; ~17% generate a click.
   - This baseline rate dictates auction pricing, evaluation metrics (LogLoss & ROC-AUC), and the necessity of negative downsampling.
2. **Diurnal Cyclicity (Hour-of-Day Pattern):**
   - Click rates are not constant across 24 hours. CTR drops during sleeping hours (2 AM – 5 AM) and peaks during morning commutes and evening relaxation.
   - Feature engineering must extract cyclical features: $\sin\left(\frac{2\pi \cdot \text{hour}}{24}\right)$ and $\cos\left(\frac{2\pi \cdot \text{hour}}{24}\right)$.
3. **Banner Position & Position Bias:**
   - Ad placements on screens are indexed (e.g. `banner_pos` 0, 1, 7). Position 0 (above-the-fold header) typically has radically higher CTR than lower banner slots simply because users see it first, regardless of ad quality.
4. **Traffic Segmentation (In-App vs. Mobile Web):**
   - Avazu has a sentinel placeholder `app_id = "ecad2386"` indicating the impression occurred on a mobile website browser, while distinct `app_id` values represent installed smartphone apps. In-app vs. mobile web traffic display vastly different conversion patterns and latencies.
5. **High Ad Turnover & Cold-Start Dynamics:**
   - Comparing the unique ad campaign IDs (`C14`) between early hours and later hours reveals that **15–25% of active ads are brand new**.
   - This proves that a purely static or memory-based model will fail, justifying dynamic exploration (Contextual Multi-Armed Bandits) and category-level embedding fallbacks.

### 🎯 The Production Problem It Solves
- **Preventing Position Bias Leakage:** If a model learns that "Ad X is amazing" when Ad X was merely placed in Position 0 all day, the model makes poor ranking decisions when that ad is moved elsewhere.
- **Validating Bandits & Exploration:** Quantifying the cold-start rate gives the business team proof that a Reinforcement Learning bandit exploration budget is mandatory rather than optional.

### 🎤 Interview Questions on This Specific Concept:
> **Q1: "What key patterns did you discover during EDA on the Avazu CTR dataset, and how did they directly influence your model architecture?"**  
> **A:** *"Our EDA uncovered four key operational realities: First, an 82% dummy ID rate (`a99f214a`), which forced us to implement a composite user proxy (`IP + device_model`) for frequency features. Second, strong diurnal cyclicity, prompting cyclical sine/cosine hour encodings. Third, severe position bias where position 0 had significantly higher baseline CTR, requiring position to be an explicit conditioning feature. Fourth, a ~20% ad campaign turnover rate across consecutive windows, confirming that an exploration mechanism like LinUCB bandits was required to prevent cold-start starvation."*

> **Q2: "How do you handle position bias in ad ranking models?"**  
> **A:** *"Position bias means users click top positions simply because of screen placement, not ad relevance. During offline training, `banner_pos` is provided as an input feature so the model attributes the position effect to that feature rather than attributing it to the ad itself. During online inference at rank time, if position is not yet known before the auction, we evaluate candidates with a default position or marginalize over possible slots."*
