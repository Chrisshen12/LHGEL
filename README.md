# LHGEL
Code of the paper: LHGEL: Large Heterogeneous Graph Ensemble Learning using Batch View Aggregation

The camera-ready paper for ICDM 25 can be found at: [LHGEL](https://www.computer.org/csdl/proceedings-article/icdm/2025/959900a713/2eowomO8ivm)

## Requirements

#### 1. Neural network libraries for GNNs

* [pytorch](https://pytorch.org/get-started/locally/)
* [pytorch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

Please check your cuda version first and install the above libraries matching your cuda. If possible, we recommend to install the latest versions of these libraries.

## Data preparation

* HGB datasets for node classification
* Ogbn-mag

These datasets include four medium-scale datasets. Please download them from pytorch geometric [pytorch-geometric-dataset](https://pytorch-geometric.readthedocs.io/en/2.5.3/modules/datasets.html#heterogeneous-datasets).

---

## Citation

If you find this work useful in your research, please cite:

```bibtex
@inproceedings{lhgel2025icdm,
  title = {LHGEL: Large Heterogeneous Graph Ensemble Learning using Batch View Aggregation},
  author = {Jiajun Shen and Yufei Jin and Yi He and Xingquan Zhu},
  booktitle = {Proceedings of the IEEE International Conference on Data Mining (ICDM)},
  year = {2025},
  month = {11},
  pages = {713--722}
  doi = {10.1109/ICDM65498.2025.00079}
}
```

If you encounter any issues, please feel free to reach out to me at jshen2024@fau.edu.
