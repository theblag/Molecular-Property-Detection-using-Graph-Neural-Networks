# Molecular Property Prediction & ADMET Profiling Using Hybrid Graph Neural Networks & KAN
**CSE311: Artificial Intelligence — Final Project Report & Presentation Reference**

---

## Part 1: Presentation Deck Guide (Mapped to CSE311 Template)

### Slide 1: Title Slide (30s)
* **Project Title:** Molecular Physicochemical Property Prediction & ADMET Profiling Using Hybrid Graph Neural Networks and Kolmogorov-Arnold Networks
* **Course:** CSE311 Artificial Intelligence | Final Project Review
* **Group Details:** Group Number `[ ]` | Members: Aditya Arun & Teammate `[Roll Numbers]`

---

### Slide 2: Introduction (1 min)
* **Problem Addressed:** Early-stage Computer-Aided Drug Discovery (CADD) relies heavily on identifying high-affinity, drug-like compounds. However, measuring physicochemical and pharmacokinetic ADMET properties (solubility, hydration energy, lipophilicity) using wet-lab assays is slow, expensive, and low-throughput.
* **Objective:** Build an end-to-end, interpretable AI system that accepts raw 2D molecular SMILES strings, predicts multi-property ADMET targets using Graph Neural Networks, profiles drug-likeness (Lipinski's Rule of 5, QED), and visually explains functional group contributions.
* **Relevance:** Over 85% of clinical drug candidates fail due to poor ADMET profiles. Accurate computational screening dramatically shortens early hit-to-lead pipelines.

---

### Slide 3: Methodology (2 min)
* **AI Techniques / Algorithms Used:**
  1. **Graph Message Passing (Microscopic):** 3-layer Graph Convolutional Network (GCNConv) with Batch Normalization and Dropout (0.2).
  2. **Dual-Readout Graph Aggregation:** Global Mean Pooling (capturing bulk molecular environment) concatenated with Global Max Pooling (preserving reactive functional group signals).
  3. **Thermodynamic Hybrid Fusion:** Concatenation with 5 macroscopic physicochemical descriptors computed on the fly via RDKit (Molecular Weight, TPSA, LogP, Rotatable Bonds, H-Bond Donors).
  4. **Readout Heads:** Standard Multi-Layer Perceptron (MLP) baseline vs. Kolmogorov-Arnold Network (KAN) employing learnable Radial Basis Function (RBF) spline edges.
  5. **Explainable AI (XAI):** Atom-level occlusion saliency maps rendered onto 2D molecular structures via RDKit Cairo engine.
* **Architecture Workflow:**
  $$\text{SMILES} \longrightarrow \begin{cases} \text{Graph Matrix } (X \in \mathbb{R}^{N \times 9}, A, W) \xrightarrow{\text{3x GCN}} \text{Dual Pooling} \rightarrow \mathbf{128\text{-d}} \\ \text{RDKit Tabular Descriptors } (T \in \mathbb{R}^5) \xrightarrow{\text{BatchNorm}} \mathbf{5\text{-d}} \end{cases} \longrightarrow \mathbf{133\text{-d}} \longrightarrow \text{Readout Head} \longrightarrow \hat{y}$$
* **Implementation Stack:** Python, PyTorch Geometric (PyG), PyTorch, RDKit, Matplotlib, Pandas, CUDA T4 GPU.

---

### Slide 4: Results (2.5 min)
* **1. Delaney ESOL Architecture Ablation Study:**
  * Baseline GCN (Graph Only, Mean Pool): **$1.2797\ \log S$**
  * Regularized GCN (Batch Normalization + Dropout): **$1.0660\ \log S$**
  * **Hybrid KAN-GNN (GCN + Descriptors + KAN Readout):** **$0.7139\ \log S$**
  * **Hybrid GNN-MLP (GCN + Descriptors + MLP Head):** **$0.6036\ \log S$** *(52.8% error reduction over baseline)*
* **2. Multi-Property Generalization Across MoleculeNet:**
  * **Aqueous Solubility (Delaney ESOL):** Best Test RMSE = **$0.6036\ \log_{10}(\text{mol/L})$**
  * **Hydration Free Energy (FreeSolv SAMPL):** Best Test RMSE = **$1.1418\ \text{kcal/mol}$**
  * **Distribution Ratio (Lipophilicity ChEMBL):** Best Test RMSE = **$0.7369\ \log_{10}(\text{ratio})$**
* **3. Model Explainability Observations:**
  * Saliency maps confirm the model learns authentic chemistry: hydrophilic groups ($-\text{COOH}$, $-\text{OH}$) strongly promote aqueous solubility (positive attribution), whereas non-polar aromatic benzene rings penalize solubility (negative attribution).

---

### Slide 5: Conclusion (30s)
* **Work Completed:** 
  1. Built end-to-end graph featurization pipeline directly from SMILES.
  2. Implemented and benchmarked 4 distinct model architectures including a novel KAN-GNN readout head.
  3. Trained models on 3 standard MoleculeNet datasets.
  4. Built an integrated ADME Drug Discovery Card profiler and substructure attribution visualizer.
* **Main Outcome:** Proved that pure GNNs fail to capture macroscopic molecular volume and that hybridizing graph message passing with bulk thermodynamic descriptors cuts prediction error by more than half.
* **Limitations:** 
  * 2D graph representations do not explicitly model 3D torsional conformation or stereoisomer spatial orientations.
  * KAN requires more training stability tuning on smaller sample regimes compared to standard linear layers.

---

### Slide 6: References (30s)
* **Research Papers:**
  1. Delaney, J. S. (2004). *ESOL: Estimating aqueous solubility directly from molecular structure.* Journal of Chemical Information and Computer Sciences, 44(3), 1000–1005.
  2. Kipf, T. N., & Welling, M. (2017). *Semi-Supervised Classification with Graph Convolutional Networks.* ICLR.
  3. Liu, Z., et al. (2024). *KAN: Kolmogorov-Arnold Networks.* arXiv preprint arXiv:2404.19756.
  4. Wu, Z., et al. (2018). *MoleculeNet: A benchmark for molecular machine learning.* Chemical Science, 9(2), 513–530.
* **Datasets & Repositories:**
  * MoleculeNet / DeepChem S3 Data Repository (`delaney-processed.csv`, `SAMPL.csv`, `ChEMBL Lipo`).
  * PyTorch Geometric (PyG) and RDKit (Open-Source Cheminformatics Toolkit).

---

## Part 2: Technical Deep Dive & Chemistry Reference

### 1. Delaney's General Solubility Equation Rationale
Aqueous solubility ($\log S$) is governed by cavitation energy and electrostatic solvation payoff. Standard GCN layers operate via localized neighbor aggregation (1- to 3-hop radius)[cite: 1]. They fail to perceive total molecular size or bulk hydrophobic volume. By providing **5 whole-molecule physical descriptors** (Molecular Weight, TPSA, Wildman-Crippen LogP, Rotatable Bonds, and H-Bond Donors)[cite: 1], the hybrid model solves this limitation and reduces the physical prediction error factor from $\sim 19.0\times$ to $\sim 4.0\times$[cite: 1].

### 2. Kolmogorov-Arnold Network (KAN) Readout Details
* Standard MLPs apply static weights on edges and fixed activation functions on nodes ($\text{ReLU}$).
* KAN places **learnable activation functions on edges** based on the Kolmogorov-Arnold representation theorem:
  $$\text{Output} = \text{SiLU}(x) \cdot W_{\text{base}} + \sum_{n=1}^B \exp\left(-(x - \mu_n)^2\right) \cdot W_{\text{spline}}$$
* Implemented using 8 Radial Basis Function (RBF) spline grids between $[-2.0, 2.0]$[cite: 1]. It achieved an RMSE of **$0.7139\ \log S$**[cite: 1], outperforming all pure graph configurations without requiring wide MLP hidden layers.

### 3. Model Explainability Mechanism (XAI)
Atom-level occlusion saliency masks out each atom's 9-dimensional node features sequentially and computes the prediction residual:
$$\Delta y_i = \hat{y}_{\text{full}} - \hat{y}_{\text{masked } i}$$
The weights are rendered into 2D contour similarity maps via `rdMolDraw2D.MolDraw2DCairo`[cite: 1]. Red contours represent functional groups promoting solubility, while blue contours represent hydrophobic regions penalizing solubility.