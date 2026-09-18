<div align="center">

# MGCL-DA

### Multi-view Graph Contrastive Learning with Dynamic Self-aware and Cross-sample Topology Augmentation for Brain Disorder Diagnosis

[![MICCAI 2025](https://img.shields.io/badge/MICCAI-2025-2F6FBB.svg)](https://papers.miccai.org/miccai-2025/0623-Paper2205.html)
[![Early Accept](https://img.shields.io/badge/Early_Accept-red.svg)](https://papers.miccai.org/miccai-2025/0623-Paper2205.html)
[![Paper](https://img.shields.io/badge/Paper-Open_Access-4CAF50.svg)](https://papers.miccai.org/miccai-2025/paper/2205_paper.pdf)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Hao Zhang<sup>1</sup>, Xiaoyun Liu<sup>2</sup>, Shuo Huang<sup>1</sup>, Yonggui Yuan<sup>2</sup>, Daoqiang Zhang<sup>3</sup>, Li Zhang<sup>1</sup>**

<sup>1</sup> Nanjing Forestry University  
<sup>2</sup> Southeast University  
<sup>3</sup> Nanjing University of Aeronautics and Astronautics

</div>

<p align="center">
  <img src="img/framework.png" alt="MGCL-DA framework" width="95%">
</p>

## Overview

**MGCL-DA** is a multi-view graph contrastive learning framework for rs-fMRI-based brain disorder diagnosis. It is designed to model both individual-specific brain topology and functional heterogeneity across subjects.

Unlike graph contrastive learning methods that rely on static graph augmentations, MGCL-DA constructs two complementary augmented views and updates them dynamically during training. The framework then applies semantic-aware contrastive constraints among the original, self-aware, and cross-sample views to learn more discriminative brain-network representations.

The method was evaluated on a major depressive disorder (MDD) dataset and was accepted by **MICCAI 2025 as an Early Accept paper**.

## Method Highlights

- **Self-aware topology augmentation** suppresses redundant information and emphasizes subject-specific functional patterns.
- **Cross-sample topology augmentation** captures inter-subject heterogeneity through interactions across brain-network samples.
- **Dynamic view updating** progressively refines the augmented representations instead of keeping them fixed throughout training.
- **Multi-view contrastive learning with min-max constraints** aligns semantically related views while preserving complementary information between different augmentation strategies.
- **ST-GCN-based representation learning** jointly models spatial brain connectivity and temporal rs-fMRI dynamics.

## Repository Structure

| Path | Description |
| --- | --- |
| `net/model.py` | Main MGCL-DA model and the three ST-GCN branches |
| `net/SelfAwareAugmented.py` | Self-aware topology augmentation module |
| `net/CrossSampleAugmented.py` | Cross-sample topology augmentation module |
| `net/DynamicUpdate.py` | Dynamic update strategy for augmented views |
| `net/tgcn.py` | Temporal graph convolution implementation |
| `Loss.py` | Classification and multi-view contrastive objective |
| `dataset_prep.py` | PyTorch dataset wrapper |
| `img/framework.png` | Overview of the proposed framework |

## Input Convention

Following the notation in the paper, the BOLD signals of each subject form an undirected spatio-temporal brain network:

```text
G ∈ R^(T × N)
```

where `T` is the number of rs-fMRI time points and `N` is the number of brain regions defined by the atlas. For a mini-batch of `B` subjects, the inputs are extended to:

```text
G ∈ R^(B × T × N)
A ∈ R^(B × N × N)
```

where `A` is the functional connectivity matrix computed from the Pearson correlation coefficients between all ROI pairs. In the experiments, the AAL atlas divides the brain into `N = 116` regions.

The current ST-GCN implementation introduces singleton channel and instance dimensions, so the paper-level input `G ∈ R^(B × T × N)` should be provided to `Model.forward` as:

```text
B × 1 × T × N × 1
```

In `net/model.py`, the local variable named `N` denotes the batch size, while `V` denotes the number of brain regions. The shared adjacency matrix is loaded from `adj_matrix.npy` under the configured `root_path` and normalized internally.

> This repository currently provides the core model, topology augmentation modules, dataset wrapper, and learning objective. Dataset preprocessing should follow the rs-fMRI and brain-network construction protocol described in the paper.

## News

- **Jun. 2025** — MGCL-DA was accepted by MICCAI 2025 as an **Early Accept** paper.

## Citation

If you find this work useful, please cite:

```bibtex
@InProceedings{ZhaHao_Multiview_MICCAI2025,
  author    = {Zhang, Hao and Liu, Xiaoyun and Huang, Shuo and Yuan, Yonggui and Zhang, Daoqiang and Zhang, Li},
  title     = {Multi-view Graph Contrastive Learning with Dynamic Self-aware and Cross-sample Topology Augmentation for Brain Disorder Diagnosis},
  booktitle = {Medical Image Computing and Computer Assisted Intervention -- MICCAI 2025},
  year      = {2025},
  publisher = {Springer Nature Switzerland},
  volume    = {LNCS 15971},
  pages     = {532--542}
}
```

## License

This project is released under the [MIT License](LICENSE).
