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

**MGCL-DA** is a multi-view graph contrastive learning framework for rs-fMRI-based brain disorder diagnosis. It dynamically constructs self-aware and cross-sample topology augmentations and applies semantic-aware contrastive constraints to learn discriminative brain-network representations. The work was accepted by **MICCAI 2025 as an Early Accept paper**.

## Method Highlights

- **Complementary topology augmentation:** models individual-specific patterns and inter-subject functional heterogeneity.
- **Dynamic view updating:** progressively refines the augmented brain-network representations during training.
- **Multi-view contrastive learning:** uses min-max constraints to preserve both shared and complementary semantics.

## Repository Structure

| Network module | Description |
| --- | --- |
| `net/model.py` | Main MGCL-DA architecture with three ST-GCN branches |
| `net/SelfAwareAugmented.py` | Self-aware topology augmentation |
| `net/CrossSampleAugmented.py` | Cross-sample topology augmentation |
| `net/DynamicUpdate.py` | Dynamic augmentation update mechanism |
| `net/tgcn.py` | Temporal graph convolution layer |

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
