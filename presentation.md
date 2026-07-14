---
marp: true
theme: rosepine
---

# Hi-SAGACT: **Hi**gh-throughput **S**creening of **A**ggregate **G**ene-set **A**ssociated **C**ollapse to **T**reatment 

---

## How to combine genomics with high-throughput drug screening ?

- Mutation content of cell as predisposing factor to drug response
- **Excess mutation** associated with **drug response** ?
    - Genome is high-dimensional and redundant
        - Mutation effect at the Gene Set level to detect systemic failure
    - Drug response is diverse and hard to generalise
        - Drug specific estimation of effect

---

### Data used

![bg contain left](./img/DepMapTitleLogoStacked.svg)

- Joint effort of Broad and Sanger institutes
- Genomics of Drug Sensitivity in Cancer (GDSC datasets)
    - 969 cancer cell lines
        - WES data for variants
    - 297 drugs at 7 doses

---

## Estimating excess mutational burden

---

### Leave-one out approach

- Interested in Gene set failure, not individual variant effect
    - mutation burden in terms of injured gene count
        - Variant considered if non-silent mutation (WES)
- The aggregate cell panel mutation content is unspecific
    - Ideal null against which to detect specific mutation burden
- For a given gene-set in a given cell line, evaluate deviation from aggregate leave-one out average injured gene count

---

![bg contain](./img/mutation_burden_scatter_zscore.png)

---

### Merits of this empirical Null

- (+)
    - By definition accounts for gene-set specific confounding factors
    - Automatically adjust for cell line specific baseline mutation rate
- (-)
    - Need a broad cell line panel

---

![bg contain](./img/Gene_set_mutation_zsore_bar.png)

---

## Estimating drug sensitivity

---

### Using the mass of non-responding wells

- Most cell:drug pairs don't produce any effect
    - Focus on subset where $IC50 \gt Concentration_{max}$ 
    - Ideal Null set against which to evaluate significance of observed effect

---

### AUC to quantify effect

![bg contain left](./img/AUC_explainer.png)

- AUC: 
    - $0 \lt AUC \lt 1$
    - small AUC $\to$ more responsive
    - $\beta$ distribution
---

![bg contain](./img/null_AUC_kde.png)

---

### Accounting for null set size difference

- Estimate null $\beta$ distribution using null samples
    - Different drugs will have varying number of null samples
    - Smaller null set size noisier
- Use Shrinkage to pool information across drugs and have more robust estimate
    - James Stein estimator: weigthed average between global and drug specific parameter values based on null set size


---

![bg contain](./img/null_AUC_kde.png)

![bg contain](./img/shrunken_null_AUC_kde.png)

---

### Adaptive AUC significance

- For each cell-line:drug combination AUC, estimate significance using drug specific null
    - drug-specific $\beta$ null cdf for observed AUC


---
![bg contain](./img/auc_ranking_correction.png)

---

## Estimating association between excess mutation and drug sensitivity

---

### Focusing on extremely responsive cell lines

- Applying minimum hypergeometric test (GSEA) on cell lines ranked by drug sensitivity
- Estimating enrichment of specific Gene Set at the top of this cell-line ranking
    - Subset of cell lines where Gene Sets significantly enriched in excess mutations


---

### The impact of thresholding mutational burden

---

![bg contain](./img/high_sensitivity.png)
![bg contain](./img/high_resilience.png)

---

#### Find optimal z-score for association

- For each drug:Gene-set pair find the z-score with maximum drug response association
- Magnitude of z-score is important for interpretation:
    - small z-score: Gene-set very sensitive
    - large z-score: Gene-set quite resilient before collapse
    - Non-monotonic pattern: Compensation mechanisms triggered passed a certain level of injury?


---
## TODO:

- Extract underlying gene level effect
    - Leading edge analysis
    - Synthetic lethality architecture for each drug
- Identify illustrative examples
- Extend to morphological response (imaging feature instead of cell viability)
