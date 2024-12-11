# Multi-scale feature consistent model under stochastic grayscale perturbation for semi-supervised medical image segmentation

Under review.

No public links currently.

In this paper, we propose novel a Multi-scale feature consistent model under stochastic Grayscale Perturbation (MGP). Specially, we include the use of a Non-Linear Transformation (NLT) to impose an stochastic grayscale perturbation. Various non-linear mappings on inputs’ intensity allow the network to learn the anatomical structure appearance (shape and intensity distribution) in medical images. Apart from regularly constructing perturbation-based regularization at the output level, we explore feature information to further promote segmentation performance. An Attention-guided Multi-scale Feature Consistency (AMFC) is proposed to learn transformation-invariant robust representations with multi-scale semantic information. Experimental results show that our method achieves state-of-the-art performance on 3 public datasets ACDC, BraST2019, and PDDCA. Our method achieves stable and excellent segmentation results in various tasks with high label efficiency, showing its generality and promising potential for medical care. Our codes will be publicly available soon.

# Reproduce Experiments

This section gives a short overview of how to reproduce the experiments presented in the paper. 

### subdirection and its contents.
We place our code on the `./code`, and the dataset on the `./data`. The output model will be generated and save in the file `./model`. Here we provided our pretrained model in the `./pretrained_model`.



## Dependencies
Set up the environment.

```
cd code
pip install -r requirements.txt
```

## Data

ACDC dataset has already put in the `./data`.


## Train models

Show the run instruction to train the model like:

```
./train_acdc_unet_semi_seg.sh
```

## Test models

Show the run instruction to test the model like:

```
./test_acdc_unet_semi_seg.sh
```

