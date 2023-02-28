# MDGCN
## Description
This is the repository for the TGRS paper [Multiscale Dynamic Graph Convolutional Network for Hyperspectral Image Classification].
Abstract: Convolutional Neural Network (CNN) has demonstrated impressive ability to represent hyperspectral images and to achieve promising results in hyperspectral image classification. However, traditional CNN models can only operate convolution on regular square image regions with fixed size and weights, so they cannot universally adapt to the distinct local regions with various object distributions and geometric appearances. Therefore, their classification performances are still to be improved, especially in class boundaries. To alleviate this shortcoming, we consider employing the recently proposed Graph Convolutional Network (GCN) for hyperspectral image classification, as it can conduct the convolution on arbitrarily structured non-Euclidean data and is applicable to the irregular image regions represented by graph topological information. Different from the commonly used GCN models which work on a fixed graph, we enable the graph to be dynamically updated along with the graph convolution process, so that these two steps can be benefited from each other to gradually produce the discriminative embedded features as well as a refined graph. Moreover, to comprehensively deploy the multi-scale information inherited by hyperspectral images, we establish multiple input graphs with different neighborhood scales to extensively exploit the diversified spectral-spatial correlations at multiple scales. Therefore, our method is termed `Multi-scale Dynamic Graph Convolutional Network' (MDGCN). The experimental results on three typical benchmark datasets firmly demonstrate the superiority of the proposed MDGCN to other state-of-the-art methods in both qualitative and quantitative aspects.


## Requirements

- Tensorflow (1.14.0)

## Usage

You can conduct classification experiments on hyperspectral images (e.g., Indian Pines) by running the 'Main.m' file.

## Cite
Please cite our paper if you use this code in your own work:

```
@ARTICLE{8907873, 
    author={S. {Wan} and C. {Gong} and P. {Zhong} and B. {Du} and L. {Zhang} and J. {Yang}}, 
    journal={IEEE Transactions on Geoscience and Remote Sensing}, 
    title={Multiscale Dynamic Graph Convolutional Network for Hyperspectral Image Classification}, 
    year={2019}, 
    volume={}, 
    number={}, 
    pages={1-16}, 
    keywords={Hyperspectral imaging;Convolution;Feature extraction;Kernel;Support vector machines;Training;Dynamic graph;graph convolutional network (GCN);hyperspectral image classification;multiscale information.}, 
    doi={10.1109/TGRS.2019.2949180}, 
    ISSN={1558-0644}, 
    month={}
}
```
