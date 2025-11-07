---
title: Tackling Information Leakage in Concept Bottleneck Models 
date: 2025-06-26
links:
  - type: site
    url: https://github.com/ahabisaax/xai-crcbm
tags:
  - Deep Learning
  - Adversarial Training
  - Information Theory
  - Interpretability 
---

Deep Neural Networks (DNNs) are powerful black boxes, but their lack of transparency makes them unsuitable for high-stakes domains like healthcare. **Concept Bottleneck Models (CBMs)** were proposed as a solution.

A CBM is forced to follow an interpretable, two-stage reasoning path:
1.  **Encoder ($g$):** Maps a high-dimensional input $x$ to a vector of human-understandable concepts, $c$.
2.  **Task Predictor ($f$):** Makes a final prediction $y$ using *only* the concepts $c$

The model's prediction is a function $y = f(g(x))$. This allows for human-in-the-loop intervention; a user can check the concepts and even correct them to fix the model's reasoning.

It also introduces a more interpretable setup where users can deduce from the predicted concepts why a model has given a certain final output $y$.
However, standard CBMs suffer from **Concept-Task Leakage (CTL)**. The concept encoder $g$ "cheats" by encoding spurious, non-concept information from $x$ directly into the concept representations $\hat{c}$ to boost its accuracy. This makes the concepts "misinterpretable" and untrustworthy, as the model isn't truly reasoning via the pure concepts.
Concept-Task Leakage occurs due to the presence of spurious features in our training data alongside the nature of neural network optimisation. Our model exploits these to boost task accuracy.
From an information-theoretic view, CTL is present if the predicted concepts $\hat{c}$ contain information about the input $x$ that is not present in the true concepts $c$:

$$
\max\bigg(0, \frac{I(\hat{C}; Y) - I(C; Y)}{H(Y)}\bigg) > 0 
$$

This leakage breaks the model's core Markov assumption ($X \to C \to Y$). 
## The Solution: An Adversarial Critic

While prior work tried to penalize this leakage by directly estimating the mutual information, these estimators (like MINE) are high-variance and produce unstable training signals.

Our project's main contribution was to introduce an adversarial classifier which is trained alongside the CBM however we hypothesise that leakage is present whenever the following statement below is true:
$$
Acc(Y | \hat{C}) > Acc(Y | C)
$$

In simple terms: a "cheating" classifier using the *leaky* concepts ($\hat{C}$) can achieve a higher accuracy than an "honest" classifier using only the *pure* concepts ($C$).

We exploit this gap by creating a **CBM with an Adversarial Critic**, trained as a min-max game:

1.  **The CBM:** Includes its own "honest" task predictor, $f_\phi$.
2.  **The Adversarial Critic ($h_\psi$):** A separate classifier that *also* tries to predict $y$ from the same predicted concepts $\hat{c}$.

The CBM's total objective becomes:

$$
\min_{\theta, \phi} \max_{\psi} \left( \mathcal{L}_{task} + \lambda_c \mathcal{L}_{concept} - \lambda_{adv} \mathcal{L}_{critic} \right)
$$

whilst the inner optimisation of the adversarial is just maximization of the equivalent:

$$
\min_{\psi} \left(\mathcal{L}_{critic} \right)
$$

Since the loss function used for the adversary is the same as the task loss we can make the following simplification for clarity:

$$
\min_{\theta, \phi} \max_{\psi} \left( \lambda_{adv} \mathcal{L}_{task} + \lambda_c \mathcal{L}_{concept} + (1- \lambda_{adv}) (\mathcal{L}_{task} - \mathcal{L}_{critic}) \right)
$$

We can see the third term represents a "loss gap" between our true predictor and our adversarial classifier, this can be
viewed as a proxy for an accuracy gap.

Our adversarial classifier has the same structure as our main predictor and the  same loss function (BCE for binary classification)

The CBM is trained to minimize its own task and concept losses, but it is simultaneously trained to *maximize* the critic's loss. This adversarial pressure forces the concept encoder $g_\theta$ to "sanitize" its representations $\hat{c}$, removing any spurious leakage until the powerful critic $h_\psi$ can perform no better than the honest predictor $f_\phi$.

## Results

By implementing a two-stage training setup where the adversarial critic is introduced late in the process, we were able to successfully "fine-tune" the CBM. This approach **reduces CTL to near-zero levels** while maintaining a high task accuracy that is competitive with standard (leaky) CBMs. This work provides a stable and effective method for training robust, interpretable models suitable for safety-critical domains.

## Interconcept-Leakage

There is also a second form of leakage which is termed interconcept-leakage and this occurs when the learned representation of concept $c_i$ encodes additional information about 
$c_j$. Often two concepts ($c_i$, $c_j$) are correlated and so there should be some learned correlation however, ICL occurs when the learned correlation is beyond this "ground truth" level 
of correlation.

We find that our adversarial classifier decreases ICL which isn't an obvious consequence as our classifier features in the loss function solely to decrease the CTL through the "loss gap".
We hypothesis that ICL and CTL are linked to some extent. If we encode additional information about our task across all concepts, we must implicitly be "overcorrelating" our concepts since each 
concept encodes similar task-relevant features which in turn increases the level of correlation between them.


Our adversarial is therefore implicitly decorrelating our concept representations implicilty by removing this additional task-relevant information that is shared across all tasks. It can't remove all 
ICL as we hypothesis some of leakage stems from the concept encoder itself.

<!--more-->
