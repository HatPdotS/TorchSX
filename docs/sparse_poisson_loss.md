# Censored Poisson Loss for Pixel-Level SFX Refinement

## Setup

Every pixel $i$ on the detector has:
- $b_i$: background rate from radial model (known, fixed during refinement)
- $t_i$: censoring threshold (derived from $b_i$ via Poisson significance test)
- $s_i(\theta)$: predicted signal from the forward model (function of parameters $\theta$)

The predicted total rate at pixel $i$:

$$\lambda_i = b_i + s_i(\theta)$$

The data is partitioned into two sets:
- $\mathcal{S}$: sparse pixels with observed count $k_i > t_i$
- $\mathcal{C}$: censored pixels where we only know $k_i \leq t_i$

## Four cases

Every pixel falls into exactly one of these:

### Case 1: Sparse pixel, under a predicted spot ($i \in \mathcal{S}$, $s_i > 0$)

Standard Poisson NLL. This is where F, orientation, scale are constrained.

$$\mathcal{L}_1 = \lambda_i - k_i \ln \lambda_i$$

where $\lambda_i = b_i + s_i(\theta)$.

### Case 2: Censored pixel, under a predicted spot ($i \in \mathcal{C}$, $s_i > 0$)

The model predicts signal but the pixel was below threshold. Censored Poisson:

$$\mathcal{L}_2 = -\ln F(t_i \mid \lambda_i)$$

where $F(t \mid \lambda) = \sum_{k=0}^{t} \frac{\lambda^k e^{-\lambda}}{k!}$ is the Poisson CDF.

When $s_i$ is small, $\lambda_i \approx b_i$, and $F(t_i \mid b_i) \approx 1 - \varepsilon$ by construction (since $t_i$ was chosen so background pixels pass). Cost $\approx 0$.

When $s_i$ is large, $\lambda_i \gg t_i$, and $F(t_i \mid \lambda_i) \to 0$. Cost $\to \infty$.

**This is the term that penalises the model for predicting signal where none was observed.**

### Case 3: Sparse pixel, not under any prediction ($i \in \mathcal{S}$, $s_i = 0$)

Unexplained signal. The model says background only, but $k_i > t_i$:

$$\mathcal{L}_3 = b_i - k_i \ln b_i$$

This is the Poisson NLL evaluated at $\lambda_i = b_i$ against an observation well above background. It carries a high cost because $b_i$ cannot explain $k_i \gg t_i$.

**This is the term that penalises the model for failing to predict observed spots.**

### Case 4: Censored pixel, not under any prediction ($i \in \mathcal{C}$, $s_i = 0$)

Background pixel, consistent with background. Trivially:

$$\mathcal{L}_4 = -\ln F(t_i \mid b_i) \approx 0$$

This dominates by pixel count (~99% of detector) but contributes negligible cost and zero gradient. **Ignore entirely.**

## Total loss

$$\mathcal{L}(\theta) = \sum_{i \in \mathcal{S},\, s_i > 0} \left[ \lambda_i - k_i \ln \lambda_i \right]
+ \sum_{i \in \mathcal{C},\, s_i > 0} \left[ -\ln F(t_i \mid \lambda_i) \right]
+ \sum_{i \in \mathcal{S},\, s_i = 0} \left[ b_i - k_i \ln b_i \right]$$

## Notes on each term

**Term 1** (sparse + predicted): Main driver of refinement. Gradients flow through $s_i(\theta)$ to structure factors, orientation, scale, partiality. This is the standard crystal refinement signal.

**Term 2** (censored + predicted): Regulariser against hallucinated spots. If the model orientation drifts and predicts reflections in empty regions, this term catches it. Differentiable via the regularised incomplete gamma function:

$$F(t \mid \lambda) = \frac{\Gamma(t+1, \lambda)}{\Gamma(t+1)} = Q(t+1, \lambda)$$

and $\frac{\partial}{\partial \lambda} \ln F(t \mid \lambda)$ involves the Poisson PMF at $t$ divided by the CDF, which is straightforward.

**Term 3** (sparse + not predicted): Constant with respect to $\theta$ — no gradients flow because $s_i = 0$ and $b_i$ is fixed. But it provides a **diagnostic**: the total cost of unexplained signal. If this is large, the model is missing reflections. These pixels can be clustered and compared against the predicted lattice to flag indexing failures, secondary lattices, or ice.

Importantly: if you later expand the model to predict those spots (e.g. by discovering a second lattice or correcting indexing), those pixels move from Term 3 to Term 1, and the total loss drops. So Term 3 acts as a standing incentive for model completeness.

## Degeneracy analysis

- **Scale → 0**: Kills all $s_i$. Term 1 becomes $b_i - k_i \ln b_i$, which is high for strong spots. Prevented.
- **Orientation drift**: Moves predicted spots off observed positions. Term 1 cost increases (predicting background where signal exists), Term 2 cost increases (predicting signal where nothing exists). Doubly prevented.
- **F → 0 for all reflections**: Same as scale → 0. Term 1 prevents it.
- **Predict everywhere**: Large $s_i$ at censored pixels. Term 2 prevents it.
- **Mosaicity → 0**: Spots become infinitely sharp, most predicted pixels have $s_i \approx 0$. Effectively same as missing the spots — Term 1 penalises.

The only minimum is the correct one: predict signal where sparse pixels are, predict nothing where they aren't.

## Practical evaluation

Per predicted reflection with footprint of ~$N$ pixels:
- Look up which of those $N$ pixels are in the sparse tensor: typically a few pixels → Term 1
- The rest are censored → Term 2
- Both are fast to evaluate

Unexplained sparse pixels (Term 3): compute once per epoch as a diagnostic, no gradient needed.

Total cost per frame: ~100 reflections × ~20 pixel footprint = ~2000 pixel evaluations, dominated by sparse lookups and a few CDF evaluations. Very fast.

## Signal/background weighting

The earlier idea of weighting by signal/background can optionally be applied on top as observation weights $w_i$ on Term 1 to focus the geometry refinement on the strongest spots:

$$\mathcal{L}_1^{weighted} = w_i \left[ \lambda_i - k_i \ln \lambda_i \right]$$

where $w_i = \max\left(\frac{k_i - b_i}{b_i},\, \varepsilon\right)$. This is not statistically necessary (the Poisson model already naturally downweights low-count observations) but can help with convergence stability by letting the strongest spots dominate the early gradient updates.