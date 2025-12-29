# NSGA-II and MCDM-Based Material Selection Framework

## Pipeline for Multi-Objective Optimization and Multi-Criteria Decision-Making

This repository provides the full computational pipeline supporting the research paper  
**"Decision-support analytics for material selection for production tooling: a systematic review and multi-objective optimisation of biocomposites"**.  
It implements a transparent and reproducible framework combining **NSGA-II multi-objective optimization** with **multi-criteria decision-making (MCDM) methods** for material selection.

All datasets, scripts, and figures required to reproduce the results presented in the paper are provided.

---

## Table of Contents
- [Context](#context)
- [Objectives](#objectives)
- [Methodological Pipeline](#methodological-pipeline)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Generated Results](#generated-results)
- [Reproducibility Notes](#reproducibility-notes)
- [References](#references)

---

## Context

Material selection for engineering applications is a multi-objective and multi-criteria problem involving trade-offs between performance, robustness, cost, environmental & social impact.

In this work:
- NSGA-II is used to explore the Pareto-optimal space of weighting strategies.
- Classical MCDM methods (TOPSIS, VIKOR, ARAS) and weighting method (Entropy, CRITIC) are applied for ranking.

---

## Objectives

1. Provide a reproducible NSGA-II optimization pipeline.
2. Compare NSGA-II reference solutions with MCDM-based rankings.
3. Ensure open and transparent access to all data and scripts.
4. Enable verification, reuse, and extension of the proposed methodology.

---

## Methodological Pipeline

The pipeline is structured into three main stages:

### 1. Data Processing and Normalization
- Load material property data from CSV files.
- Apply benefit/cost normalization to all criteria.
- Ensure robust matching between datasets and ranking tables.

### 2. NSGA-II Multi-Objective Optimization
- Generate Pareto-optimal weight vectors.
- Optimize three objectives:
  - f1: maximization of overall material score
  - f2: robustness (minimum score)
  - f3: weight balance (imbalance minimization)
- Identify representative NSGA-II solutions:
  - Knee solution
  - Balanced solution
  - Robust solution

### 3. MCDM Evaluation and Comparison
- Compute Entropy and CRITIC-based weights.
- Apply TOPSIS, VIKOR, and ARAS methods.
- Integrate user-provided MCDM rankings.
- Compare NSGA-II and MCDM results through rankings and visualizations.

---

## Technologies Used

### Programming Language
- Python (>= 3.8)

### Scientific Libraries
- numpy
- pandas
- matplotlib
- seaborn
```bash
pip install numpy pandas matplotlib seaborn
```
### Optimization and Decision-Making
- NSGA-II (custom implementation)
- Entropy weighting
- CRITIC weighting
- TOPSIS
- VIKOR
- ARAS

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
