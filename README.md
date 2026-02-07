# Hyperspectral Endmember Extraction (PCA + PPI)

This project implements **Endmember Extraction** for hyperspectral imagery using **Principal Component Analysis (PCA)** and **Pixel Purity Index (PPI)**.

##  Methodology
1. **Dimensionality Reduction (PCA)**: 
   - Determined the number of endmembers based on the cumulative energy ratio (99% criterion) of eigenvalues.
   - Selected the top $r$ principal components where energy concentration is highest.
2. **Endmember Searching (PPI)**: 
   - Performed PPI on the PCA-reduced feature space.
   - Compared iteration counts (1000 vs 10000) to verify stability.

##  Experimental Results
### 1. Eigenvalue Distribution
The energy is concentrated in the first few components. The number of endmembers ($r$) was determined when cumulative energy reached 99%.

### 2. PCA Visualization
Visualized the first 8 principal components. The structural information is clear in earlier components, while noise increases in later components.

### 3. PPI Detection
Experiments showed that PPI results are stable across different iteration counts (1000 vs 10000), consistently identifying similar endmember locations in both Raw and PCA spaces.

##  Usage
```bash
python main.py
