<div align="center">
  
# 🚧 Work-in-Progress

# Unsupervised Hierarchy-Agnostic Segmentation: Parsing Semantic Image Structure (NeurIPS 2024) 
<!--
[![Project](http://img.shields.io/badge/Project%20Page-3d3d8f.svg)](https://lukemelas.github.io/deep-spectral-segmentation/)
[![Demo](http://img.shields.io/badge/Demo-9acbff.svg)](https://huggingface.co/spaces/lukemelas/deep-spectral-segmentation)
-->
[![Conference](http://img.shields.io/badge/Conference-NeurIPS_2024-4b44ce.svg?)]([#](https://neurips.cc/virtual/2024/poster/96040))
[![Paper](https://img.shields.io/badge/Paper-OpenReview-b31b1b)](https://openreview.net/forum?id=ELnxXc8pik)

</div>

## Description
This code accompanies the paper [Unsupervised Hierarchy-Agnostic Segmentation: Parsing Semantic Image Structure](https://openreview.net/forum?id=ELnxXc8pik) by [Simone Rossetti](https://github.com/rossettisimone) and [Fiora Pirri](https://github.com/fiora0).

## Abstract
Unsupervised semantic segmentation aims to discover groupings within images, capturing objects' view-invariance without external supervision. This task is inherently ambiguous due to the variable levels of granularity in natural groupings. Existing methods often bypass this ambiguity using dataset-specific priors. In our research, we address this ambiguity head-on and provide a universal tool for pixel-level semantic parsing of images guided by the latent representations encoded in self-supervised models. We introduce a novel algebraic methodology for unsupervised image segmentation. The innovative approach identifies scene-conditioned primitives within a dataset and creates a hierarchy-agnostic semantic region tree of the image pixels. The method leverages deep feature extractors and graph partitioning techniques to recursively identify latent semantic regions, dynamically estimating the number of components and ensuring smoothness in the partitioning process. In this way, the model captures fine and coarse semantic details, producing a more nuanced and unbiased segmentation. We present a new metric for estimating the quality of the semantic segmentation of elements discovered on the different levels of the hierarchy. The metric is beneficial because it validates the intrinsic nature of the compositional relations among parts, objects, and scenes in a hierarchy-agnostic domain. Our results unequivocally demonstrate the power of this methodology. It uncovers detailed and hierarchical semantic regions without prior definitions and scales effectively across various datasets. This robust framework for unsupervised image segmentation provides richer and more accurate semantic hierarchical relationships between scene elements than traditional algorithms. The experiments underscore its potential for broad applicability in image analysis tasks, showcasing its ability to deliver a detailed and unbiased segmentation that surpasses existing unsupervised methods.

## Examples

## How to run

1. install ```requirements.txt``` line by line,
2. setup datasets paths in ```config/datasets.json```, such as image and annotation sources for each dataset set,
3. run experiments in ```config``` directory:
   ```bash
   torchrun --nnodes 1 --nproc_per_node 4 --conf configs/voc12.json # run voc12 NMCovering experiment
   torchrun --nnodes 1 --nproc_per_node 4 --conf configs/coco_stuff.json # run coco_stuff NMCovering experiment
   torchrun --nnodes 1 --nproc_per_node 4 --conf configs/coco_things.json # run coco_thing NMCovering experiment
   torchrun --nnodes 1 --nproc_per_node 4 --conf configs/cityscapes.json # run cityscapes NMCovering experiment
   torchrun --nnodes 1 --nproc_per_node 4 --conf configs/partimagenet.json # run partimagenet NMCovering experiment
   torchrun --nnodes 1 --nproc_per_node 4 --conf configs/pascalpart.json # run pascalpart NMCovering experiment
   ```
4. results are stored in the newly created ```data``` folder.

## Citation

If you find this code useful for your research, please consider citing our paper:
```
@inproceedings{rossetti2024unsupervised,
  title={Unsupervised Hierarchy-Agnostic Segmentation: Parsing Semantic Image Structure},
  author={Simone Rossetti and Fiora Pirri},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024},
  url={https://openreview.net/forum?id=ELnxXc8pik}
}
```

## License
This code is released under the MIT License (refer to the LICENSE file for details).
