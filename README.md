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
- [Installation](#installation)
- [Usage](#usage)
- [Generated Results](#generated-results)
- [References](#references)
- [Data Description](#data-description)
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
```
## References

If you use this work, please consider citing the following research papers in your references: (to update)



## Data Description

This repository includes an Excel workbook providing the complete input data used for the MCDM and NSGA-II analyses.  
The file consolidates environmental, mechanical, physical, and economic indicators, as well as weighting schemes and ranking results, to ensure transparency and traceability of the decision process.

The workbook is organised into multiple sheets, each serving a specific purpose in the decision-support workflow.

---

### Sheet: `MCDM_criteria_9.xlsx`

This sheet contains the core decision matrix used for MCDM analyses.

**Content:**
- List of candidate materials (alternatives)
- Nine evaluation criteria:
  - Environmental impact (single-score indicator) (µpoints) - the full environmental impacts are also available
  - Density (g/cm³)
  - Tensile modulus(GPa)
  - Tensile strength (MPa)
  - Elongation at break (%)
  - Flexural modulus (GPa)
  - Flexural strength (MPa)
  - Coefficient of thermal expansion (CTE) (µm/m·°C)
  - Raw material cost (€/kg)

**Additional information:**
- Each criterion is explicitly classified as a *benefit* or *cost* attribute.
- Units are provided for all quantitative indicators.
- Environmental impact values are derived from Life Cycle Assessment (LCA) calculations performed using [OpenLCA](https://www.openlca.org) with the [ecoinvent database](https://ecoinvent.org) and some personal datas.
- Mechanical, physical, and cost data correspond to representative values reported in the literature and industrial datasheets.

This sheet represents the raw input matrix used by both classical MCDM methods and the NSGA-II-based analysis.

---

### Sheet: `LCA Criteria`

This sheet details the environmental assessment underlying the environmental impact indicator.

**Content:**
- Environmental impact metrics computed using LCA
- Aggregated single-score values used in the decision matrix
- Consistent system boundaries and assumptions across materials

---

### Sheet: `Ranking & analysis`

This sheet gathers the ranking results obtained from different weighting and decision strategies.

**Content:**
- Rankings produced using:
  - Entropy-based weights
  - CRITIC-based weights
  - Randomised weighting schemes
  - Manually defined (user-adjustable) weights
- Comparison of rankings across methods
- Identification of stable and divergent ranking patterns

This sheet supports comparative analysis of decision outcomes under alternative weighting assumptions.

---

### Sheet: `raw ranking`

This sheet contains the unprocessed ranking outputs directly resulting from the application of MCDM scoring methods.

**Content:**
- Raw scores and ranks before aggregation or interpretation
- Method-specific ranking outputs

---

### Sheet: `raw properties`

This sheet reports the original material property values prior to any normalisation or transformation.

**Content:**
- Mechanical, physical, environmental, and economic indicators
- Source-consistent values used to build the decision matrix


---

## Weighting Schemes

The Excel file includes multiple weighting strategies to reflect different decision perspectives:

- **Entropy weights:** objective weights derived from data dispersion
- **CRITIC weights:** objective weights accounting for contrast intensity and inter-criteria correlation
- **Randomised weights:** exploratory weights used to test ranking sensitivity
- **Manual weights:** user-defined weights allowing interactive scenario analysis

The sum of weights is explicitly controlled to ensure consistency across all analyses.
The computation is made on masked sheets and the selection of the weighting method is possible on the toggle cell on the first sheet.
---

## Usage Notes

- Users may modify manual weights directly in the Excel file to explore alternative decision scenarios.
- Any change in weights or criteria values can be propagated to the Python scripts by exporting the updated tables as CSV files.
- The Excel workbook is provided as a complementary, human-readable representation of the data and does not replace the scripted workflow.

---

## Data Transparency

All data included in this workbook are intended to support reproducibility and methodological clarity.  
Environmental impact values are based on established LCA tools and databases, while other indicators are sourced from the literature and industrial references.

