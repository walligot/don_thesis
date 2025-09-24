# Mechanistic Interpretations of Multi-Block State Space Models

This repository accompanies David O'Neill's masters thesis undertaken at the Gatsby Institute, for **MSc Computational Statistics and Machine Learning** at UCL (2025).  
It explores how **similarity metrics** and **linear algebraic tools** can be used to move beyond black-box evaluation and reveal the internal dynamics of artificial and biological neural networks.  

---

## 🚀 Overview

- **Topic**: Developed methods to interpret **state space models (SSMs)** used in machine learning and neuroscience.  
- **Contribution**: Showed how tools from **dynamical systems theory and linear algebra** expose differences in network mechanisms invisible at the input–output level.  
- **Skills demonstrated**:  
  - Machine learning research and experimental design  
  - Mathematical modelling (linear algebra, dynamical systems)  
  - Analysis of recurrent neural networks and state-space architectures  
  - Python (PyTorch, scikit-learn, NumPy, SciPy, Matplotlib)  
  - Neural data analysis (dimensionality reduction, similarity metrics, decoding tasks)  

---

## 📖 Thesis Summary

- **Motivation**: Neural networks and biological circuits can behave identically at the output level while relying on distinct internal computations. Understanding these mechanisms is critical for both interpretability and neuroscience.  

- **Approach**:  
  1. Built controlled benchmarks with **toy recurrent neural networks** on memory tasks.  
  2. Applied **similarity metrics** (Procrustes alignment, Dynamical Similarity Analysis) to compare latent trajectories.  
  3. Extended framework to **structured state-space models (S5 blocks)** trained on primate neural recordings.  

- **Key Results**:  
  - **Procrustes alignment** detected differences in model mechanisms, even when outputs looked the same.  
  - **Dynamical Similarity Analysis** was less effective in input-driven regimes.  
  - In multi-block SSMs, **later layers refined task-relevant geometry**, and **Gated Linear Units acted as contextual switches** shaping dimensionality.  
  - Demonstrated that internal dynamics are **core computational primitives**, not incidental by-products of architecture.  

---

## 📂 Repository Contents

- `thesis_report.pdf` — Full MSc thesis report, including background, methodology, experiments, and results.  

---

## 🛠️ Skills & Methods

- **Machine Learning**: recurrent networks, state-space models, gating mechanisms, neural decoding.  
- **Mathematics**: linear algebra (eigendecomposition, SVD, Procrustes), dynamical systems, similarity metrics.  
- **Data Analysis**: PCA, dimensionality reduction, trajectory visualisation, bootstrapped confidence intervals.  
- **Neuroscience Applications**: analysis of primate motor cortex recordings, linking behaviour decoding with latent dynamics.  

---

## 📜 Citation

If referencing this work:  
**Mechanistic Interpretations of Multi-Block State Space Models**
David O'Neill 
MSc Thesis, Computational Statistics and Machine Learning, UCL, 2025.  

---
