# Uncertainty-based Selective Clustering for Active Learning - _Official Pytorch implementation of the IEEE Access, vol.10, 2022_

__*Sekjin Hwang, Jinwoo Choi, Joonsoo Choi*__

Official Pytorch implementation for [the paper](https://ieeexplore.ieee.org/abstract/document/9925155) published on IEEE Access titled "_Uncertainty-based Selective Clustering for Active Learning_".

<img src="./pipeline.png" width="80%" height="80%" alt="Pipeline"></img>

## Abstract
_Labeling large amount of data is one of important issues in deep learning due to high labeling cost. One method to address this issue is the use of active learning. Active learning selects from a large unlabeled data pool, a set of data that is more informative to training a model for the task at hand. Many active learning approaches use uncertainty-based methods or diversity-based methods. Both have had good results. However, using uncertainty-based methods, there is a risk that sampled data may be redundant, and the use of redundant data can adversely affect the training of the model. Diversity-based methods risk losing some data important for training the model. In this paper, we propose the uncertainty-based Selective Clustering for Active Learning (SCAL), a method of selectively clustering for data with high uncertainty to sample data from each cluster to reduce redundancy. SCAL is expected to extend the area of the decision boundary represented by the sampled data. SCAL achieves a cutting-edge performance for classification tasks on balanced and unbalanced image datasets as well as semantic segmentation tasks._

## Prerequisites:   
- Linux or macOS
- Python 3.6/3.7
- CPU compatible but NVIDIA GPU + CUDA CuDNN is highly recommended.
- pytorch 1.13.1
- cuda 11.8
- Scikit-learn 1.1.0

## Running code
To train the model(s) and evaluate in the paper, run this command:

```train
python scal.py
```
Before running the code, you check the settings in config.py 
