# GMM4AnomalySeg: A Generative Method based on Constrastive Learning and Edge Post-processing


## Abstract
In this paper, we propose a novel method in pursuit of solving the anomaly segmentation task. Based on previous generative approaches to anomaly segmentation, we follow the fashion of modeling likelihood $p(feature | class)$ instead of posterior $p(class | feature)$ as is adopted by discriminative counterparts. We also introduce contrastive learning to our model where we randomly sample from GMMs representing different pixel classes and perform contrastive learning on the sampled feature vectors. One of the most salient differences from previous work is that we continuously finetune the GMM components whether we are simply fitting GMM components to training data or training our feature extraction module discriminatively. We testify our method mainly on Fishyscapes benchmark, and we outperform previous GMM-based approaches to some extent. We believe this work sheds light on GMM-based anomaly segmentation and other relative fields.

## Dataset Preparation
We suggest downloading datasets under a new directory **data/**.

## Installation
This implementation is built on [MMSegmentation v0.22.1](https://github.com/open-mmlab/mmsegmentation/tree/v0.22.1). Many thanks to the contributors for their great efforts.

Please follow the [get_started](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/get_started.md#installation) for installation and [dataset_prepare](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#prepare-datasets) for dataset preparation.

Other requirements: `pip install timm==0.5.4 einops==0.4.1`

## Usage
```shell
# multi-gpu train
bash tools/dist_train.sh configs/_gmmseg/segformer_mit-b5_gmmseg_1024x2048_160k_cityscapes.py $GPU_NUM

# single-gpu test
python tools/test_fishyscapes.py configs/_gmmseg/segformer_mit-b5_gmmseg_fishyscapes.py /path/to/checkpoint_file

```

## Acknowledgement

Our code implementation is based on [GMMSeg](https://github.com/leonnnop/GMMSeg), and we would like to thank the authors for their comprehensive work.
