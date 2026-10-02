# Molecular Property Prediction Using Graph Neural Networks (GNNs)

## 1. Project Objective
Predict aqueous solubility ($\log S$) directly from chemical SMILES strings using Graph Neural Networks (GNNs), enabling rapid in silico filtering of poorly soluble drug candidates prior to wet-lab synthesis.

---

## 2. Completed Work (Phase 1)
* **Data Ingestion:** Processed the Delaney ESOL benchmark (1,128 compounds) using RDKit and PyTorch Geometric.
* **Chemical Featurization:** Constructed molecular graphs using **9 atom descriptors** (atomic number, degree, formal charge, hybridization, aromaticity, H-count, radical electrons, ring membership, chirality) and bond-order edge weights.
* **GNN Architecture:** Implemented a 3-layer Graph Convolutional Network (`GCNConv`) with Batch Normalization and ReLU non-linearities.
* **Readout Layer:** Integrated dual pooling (`global_mean_pool` + `global_max_pool`) into a 128-dimensional molecular embedding to represent both overall mass and extreme polar groups.
* **Regularization & Optimization:** Added Dropout ($p=0.2$), Adam L2 weight decay ($10^{-4}$), and best-checkpoint weight restoration to resolve severe overfitting.
* **Evaluation & Testing:** Reduced Test RMSE from an initial **1.28** down to **1.066 logS**. Built an inference pipeline and verified actual-vs-predicted correlations on unseen compounds.

---

## 3. Current Benchmark Summary
| Metric | Baseline | Target (Phase 2) |
| :--- | :--- | :--- |
| **Test RMSE** | **1.066 logS** | **$< 0.75\ \log S$** |
| **Input Representation** | 9-feature graph nodes + bond weights | 9-feature nodes + multi-dimensional bond tensors |
| **Splitting Method** | Random 80/20 split | Scaffold Split (Bemis-Murcko) |

---

## 4. Pending Tasks (Phase 2 Roadmap)
1. **Bond Embedding Upgrade:** Transition from scalar-weighted `GCNConv` to `GINEConv` or Directed MPNN to embed explicit chemical bond types (single, double, triple, aromatic).
2. **Hybrid Descriptors:** Concatenate global physicochemical properties (Molecular Weight, TPSA, Rotatable Bonds) into the post-pooling dense layer.
3. **Scaffold Splitting:** Evaluate on novel chemical backbones rather than random splits to measure true structural generalization.
4. **Interactive Demo:** Deploy a lightweight Streamlit/Gradio web frontend that accepts user-drawn or typed SMILES and outputs real-time solubility predictions.