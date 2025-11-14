p-Value Hacking in AB Testing
================

When I teach my students the pitfalls (and benefits) of "peeking in AB tests" also known as
**early stopping in sequential testing**, I am often asked for clean solutions to this dilemma.
I usually do not have the time in my classroom to present 
the various algorithms designed to deal with peeking, so here is my attempt 
at a quick overview.

There *is* a very large literature on sequential testing; you’re right that many methods exist that let you peek and still keep Type-I error under control. 
Below is a compact, practical recommendation (shortlist), reasons to pick each method, notes about software, and a small, **easy-to-run Python example** (Beta–Binomial *mixture* sequential test) for illustration.

---

## Quick shortlist (recommended order)

1. **Group-sequential / alpha-spending (Lan–DeMets with O’Brien–Fleming or Pocock boundaries)**

   * *Why:* Simple, extremely well understood, commonly used in clinical trials, good when you plan a small number of interim looks (e.g. 3–6).
   * *When to use:* You have a pre-specified schedule of interim analyses and want exact frequentist alpha control.
   * *Software:* R — **`gsDesign`**, **`rpact`** (mature, well-documented).
   * *Complexity:* Low–medium (choose a spending function, number/timing of looks).

2. **Mixture Sequential Probability Ratio Test (mSPRT) / mixture-likelihood tests (aka Beta–Binomial for coins)**

   * *Why:* Very simple to implement for binary data (coin flips), supports continuous monitoring (you can check after every toss), and gives powerful tests that control Type I rate when calibrated correctly. Works well as a teaching tool.
   * *When to use:* You want continuous peeking and a practically simple algorithm with good power.
   * *Software:* Not always packaged as a single mainstream Python lib, but trivial to code yourself for Bernoulli data (see code below). R implementations / code snippets exist in literature and packages for sequential analysis.
   * *Complexity:* Low.

3. **Always-valid p-values / e-processes / betting-based tests (e-values)**

   * *Why:* Modern, conceptually clean framework for continuous monitoring with rigorous Type I guarantees (testing via supermartingale / e-process). Very flexible and extends to composite hypotheses and multiple testing settings.
   * *When to use:* You want principled online inference, possibly multiple comparisons / online FDR control.
   * *Software:* Emerging — some R packages and research code exist; implementations are less “standardized” but becoming popular in online A/B testing communities.
   * *Complexity:* Medium.

---

## Practical recommendation for teaching and classroom use

* If you want **regulatory-style** rigor and only a few interim looks → use **Group-sequential (gsDesign / rpact)**.
* If you want **continuous peeking** and an easy-to-explain method for *coin/bernoulli* experiments → use a **Beta–Binomial mixture sequential test / mSPRT** (simple to demonstrate and code).
* If you want to introduce modern theory and online FDR → present **e-values / always-valid p-values** (a bit more abstract, but very relevant).

For most teaching exercises I prefer *not overly complex* and *stable implementations*, such as:

* **Group-sequential** when the class is about clinical/regulated trials and you want to show classical alpha control, and
* **mSPRT / Beta–Binomial mixture** when you want a simple always-peekable test for coin flips and to run interactive demos/competitions.

---

## Small, runnable Python example — Beta–Binomial mixture sequential test (coin flips)

This is a friendly, intuitive test: integrate a Beta prior (e.g. Beta(1,1) uniform) over the alternative to get the *marginal likelihood* under the alternative; compare to the likelihood under the null (p_0). The sequential **mixture likelihood ratio (LR)** is

$$
\text{LR}_n = \frac{ \mathrm{Beta}(k+a, n-k+b) / \mathrm{Beta}(a,b) }{ p_0^{k} (1-p_0)^{n-k} } .
$$

Stop and **declare evidence for (p>p_0)** when (\text{LR}_n > B) (choose (B), e.g. (B=1/\alpha) heuristically), and optionally declare evidence against when (\text{LR}_n < b). This is Bayesian-inspired but is used as an mSPRT; calibration of (B) gives approximate frequentist control (and is conservative in many settings). It’s excellent for classroom intuitions.

```python
import numpy as np
from math import log
from scipy.special import betaln
import matplotlib.pyplot as plt

# Beta-Binomial marginal log-likelihood (log scale for stability)
def log_marginal_beta_binomial(k, n, a=1.0, b=1.0):
    # log Beta(k+a, n-k+b) - log Beta(a,b)
    return betaln(k + a, n - k + b) - betaln(a, b)

def log_likelihood_null(k, n, p0=0.5):
    # log p0^k (1-p0)^(n-k)
    return k * np.log(p0) + (n - k) * np.log(1 - p0)

def run_mixture_sequential(p_true=0.5, p0=0.5, a=1, b=1,
                           threshold_B=20,  # e.g. 1/alpha ~ 20 for alpha~0.05
                           n_max=2000, rng_seed=None, peek_every=1):
    if rng_seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(rng_seed)

    k = 0
    lr_vals = []
    times = []
    decision = None
    for n in range(1, n_max + 1):
        if rng.random() < p_true:
            k += 1
        if n % peek_every == 0:
            log_alt = log_marginal_beta_binomial(k, n, a=a, b=b)
            log_null = log_likelihood_null(k, n, p0=p0)
            logLR = log_alt - log_null
            LR = np.exp(logLR)
            lr_vals.append(LR)
            times.append(n)
            if LR > threshold_B:
                decision = ('reject H0 (p>p0)', n, LR)
                break
            # optional lower bound to accept null: if LR < 1/B accept null
            if LR < 1/threshold_B:
                decision = ('accept H0 (no evidence)', n, LR)
                break

    return dict(decision=decision, times=np.array(times), lrs=np.array(lr_vals), k=k, n=n)

# Demo: single run + plotting
res = run_mixture_sequential(p_true=0.55, p0=0.5, threshold_B=20, n_max=2000, rng_seed=1, peek_every=5)
print(res['decision'])
plt.plot(res['times'], res['lrs'], '-o')
plt.axhline(20, color='red', linestyle='--', label='threshold B')
plt.yscale('log')
plt.xlabel('n')
plt.ylabel('LR (log scale)')
plt.legend()
plt.show()
```

**Notes on this code**

* `a=b=1` is uniform prior; you can pick informative priors if desired (e.g., Beta(2,2) shrinks toward 0.5).
* Threshold `B` can be set to `1/alpha` as a rule-of-thumb (e.g. `B=20` for `α≈0.05`), but more careful calibration (via simulation) can be used to obtain exact frequentist control for your design.
* This test is easy to explain to students: it tracks cumulative evidence (likelihood ratio) and lets you stop whenever it crosses a threshold.

---

## Software notes & pointers

* **R — Group sequential:** `gsDesign`, `rpact` (comprehensive, used in industry/academia). If you teach clinical-trial style designs, use those.
* **R — Online FDR / alpha-investing:** look into `onlineFDR` and related packages for online multiple testing.
* **Python:** There is no single canonical package as mature as `gsDesign`, but mixture SPRT for Bernoulli is trivial to code (see snippet above). For more industrial setups, consider calling R from Python (`rpy2`) to use `rpact`/`gsDesign`.
* **Always-valid / e-values:** research code is available in various repos (authors like Howard, Ramdas, Wasserman); you can demonstrate basic e-processes via simple betting strategies or the Beta-mixture LR above (which is an e-value in many settings).

---

## Classroom progression

1. **Start with the Beta–Binomial mixture test**: show students how LR evolves and how early stopping can be justified — implement the toy code above and run tournaments.
2. **Show the failure** of naive repeated p-value peeking (contrast with the mixture test).
3. **Introduce group-sequential** (gsDesign/rpact) when discussing pre-specification and regulatory settings.
4. **Show modern e-value ideas** if you want advanced students to see the latest theory for online control and FDR.

---

