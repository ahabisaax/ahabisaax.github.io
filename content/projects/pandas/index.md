---
title: Toward using GANs in astrophysical Monte-Carlo simulations
date: 2024-10-26
links:
  - type: site
    url: https://arxiv.org/abs/2402.12396
tags:
  - Deep Learning
  - GANs
  - Generative Models
  - Astrophysics
  - Monte-Carlo Methods

# Optional external URL for project (replaces project detail page).
external_link: ''

image:
  caption: 'Basic architecture of a Generative Adversarial Network'
  focal_point: Smart
---

# High-Level Overview

Accurate modeling of spectra from X-ray sources requires computationally expensive Monte-Carlo (MC) simulations. These simulations must sample from numerous complex probability distributions—a process that is a significant computational bottleneck.

This project explores using Generative Adversarial Networks (GANs) to replace these time-consuming sampling steps.

### The Model

A GAN framework consists of two competing neural networks:
1.  **The Generator (G)**: Acts like a counterfeiter, learning to produce synthetic data (in our case, samples from a distribution) that looks real.
2.  **The Discriminator (D)**: Acts like the police, tasked with learning to distinguish between real data and the generator's synthetic fakes.

\[
\mathcal{L}_{GAN} = \mathbb{E}_{x}[\log(D(x))] + \mathbb{E}_{z}[\log(1-D(G(z)))]
\]
This competition drives both models to improve. The generator gets better at making fakes, and the discriminator gets better at distinguishing. The process continues until the generator's synthetic data is statistically indistinguishable from the real data.

## The Challenge: Sampling the Maxwell-Jüttner Distribution

Our main goal was to replicate the **Maxwell-Jüttner (MJ) distribution**, a family of distributions used in astrophysics to describe the speed of relativistic electrons. This distribution is conditioned on a temperature parameter T which produces highly skewed distributions for high T making for difficult sampling.

While GANs are powerful, they are difficult to train. It's common for the Discriminator to learn too quickly, leaving the Generator unable to learn. Given GAN training is often formulated as a min-max game where we are jointly doing gradient descent over some parameters and ascent over others we end up at a saddle point which is often an unstable loss landscape. Another issue is "mode collapse," where the generator finds one "fake" sample that works and produces it over and over, rather than learning the full, diverse distribution.

## Our Approach & Results

We focused on stabilizing the training process to overcome these challenges. By carefully modifying the training objective and network architecture, we ensured the generator received a strong, consistent learning signal.

This stable approach allowed us to successfully train a GAN to replicate the complex MJ distribution. Our results show that the generated samples are statistically indistinguishable from the true distribution, demonstrating that GANs are a viable and highly efficient alternative to traditional, slow Monte-Carlo sampling in astrophysics.

An extension this work to train a CGAN where the temperature is passed in as a hyperparameter so that we learn the full conditional distribution and can load in the MJ for a given T.

<!--more-->
