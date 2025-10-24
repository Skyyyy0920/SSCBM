# SSCBM

This is the code implementation for our paper [Semi-supervised Concept Bottleneck Models](https://arxiv.org/abs/2406.18992)

## Cite our work
```
@InProceedings{Hu_2025_ICCV,
    author    = {Hu, Lijie and Huang, Tianhao and Xie, Huanyi and Gong, Xilin and Ren, Chenyang and Hu, Zhengyu and Yu, Lu and Ma, Ping and Wang, Di},
    title     = {Semi-supervised Concept Bottleneck Models},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {2110-2119}
}
```


## Dataset Preparation

### CUB_200_2011
- Download the dataset (CUB_200_2011.tgz) from https://www.vision.caltech.edu/datasets/cub_200_2011/.
- Unpack CUB_200_2011.tgz to the `data/` directory in this project.
- Download the concept annotations (class_attr_data_10) from https://worksheets.codalab.org/bundles/0x5b9d528d2101418b87212db92fea6683, which is from the vanilla CBM.

