# [Verifix] - Post-Training Correction to Improve Label Robustness with Verified Samples

This repository implements the code for our paper [Verifix](https://arxiv.org/abs/2403.08618). 


## Dependency Installation
To set up the environment and install dependencies, follow these steps:
### Installation using conda
Install the packages either manually or use the environment.yml file with conda. 
- Installation using yml file
    ```bash
    conda env create -f environment.yml
    ```
    OR
- Manual Installation with conda environment 
    ```bash    
    ### Create Envirornment (Optional, but recommended)
        conda create --name verifix python=3.11.4
        conda activate verifix

        ### Install Packages
        pip install wandb 
        pip install argparse 
        pip install scikit-learn
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        pip install matplotlib
        pip install seaborn
        pip install SciencePlots
    ```



## Supported Datasets
### Real-world noisy dataset
The real-world noisy dataset used in this project is the WebVision dataset, designed to facilitate research on learning visual representation from noisy web data. 

1. [WebVision 1.0](https://data.vision.ee.ethz.ch/cvl/webvision/dataset2017.html)- The WebVision dataset is designed to facilitate the research on learning visual representation from noisy web data. See ```data/WebVision1.0/``` to download the data directory and process data in supported format (Refer repository [WebVision1.0](https://github.com/sangamesh-kodge/WebVision1.0)). 

2. [Mini-WebVision](https://arxiv.org/abs/1911.09781)- is a subset of the first 50 classes of Goole partition of WebVision 1.0 dataset (contains about 61234 training images). See ```data/MiniWebVision/``` to download the data directory and process data in supported format (Refer repository [Mini-WebVision](https://github.com/sangamesh-kodge/Mini-WebVision)).


3. [Clothing1M](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Xiao_Learning_From_Massive_2015_CVPR_paper.pdf)- is a 14 class dataset containing clothes (dataset has 1000000 training images with noisy labels).  See ```data/Clothing1M/``` to download the data directory and process data in supported format (Refer repository [Clothing1M](https://github.com/sangamesh-kodge/Clothing1M)).


### Synthetic Noise in standard dataset. 
In addition to the real-world noisy dataset, synthetic noise is introduced into standard datasets for further analysis and evaluation of label noise robustness. Set the ```--percentage-mislabeled``` command line argument to desired level of label noise percentage for adding synthetic uniform noise to standard dataset. The following standard datasets are used with synthetic noise (We use torchvision and hence do not require any preprocessing for these datasets):
- [MNIST](https://ieeexplore.ieee.org/document/6296535)
- [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html)
- [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html)
- [ImageNet](https://www.image-net.org/)


## Supported Network Architectures
- [LeNet5](https://ieeexplore.ieee.org/document/726791) - for MNIST
- [ResNets](https://arxiv.org/pdf/1512.03385.pdf) - for CIFAR, ImageNet , WebVision and Clothing1M
- [VGGs](https://arxiv.org/pdf/1409.1556.pdf) - for CIFAR, ImageNet and WebVision
- [InceptionResNetv2](https://arxiv.org/pdf/1602.07261.pdf) - for CIFAR, ImageNet , WebVision and Clothing1M
- [Vision Transformers (ViTs)](https://arxiv.org/pdf/2010.11929.pdf) - for CIFAR, ImageNet, WebVision and Clothing1M

## Supported Noise-Robust Training Algorithms 
Refer repository [LabelNoiseRobustness](https://github.com/sangamesh-kodge/LabelNoiseRobustness/) 

1. Vanilla SGD - Standard Stochastic Gradient Descent algorithm. 
2. [Mixup](https://arxiv.org/pdf/1806.05236.pdf)- enhances model robustness by linearly interpolating between pairs of training examples and their corresponding labels. Specifically, it generates augmented training samples by blending two input samples and their labels. This process introduces beneficial noise during training, which helps the model learn more effectively even when the training data contains noisy labels. To use mixup add the cli argument ```--mixup-alpha <value-of-hyperparameter-alpha>```. For example, ```--mixup-alpha 0.2``` means the alpha hyperparameter is set to 0.1.

2. [SAM (Sharpness-Aware Minimization)](https://arxiv.org/pdf/2010.01412.pdf)- Instead of solely minimizing the loss value, SAM aims to find a balance between low loss and smoothness. It encourages the model to explore regions with uniformly low loss, avoiding sharp spikes that might lead to overfitting. SAM exhibits remarkable resilience to noisy labels. To use SAM add the cli argument ```--sam-rho <value-of-hyperparameter-rho>```. For example, ```--sam-rho 0.1``` means the rho hyperparameter is set to 0.1.


4. [MentorMix](https://arxiv.org/pdf/1911.09781.pdf) develops on the idea of MentorNet and Mixup. To use MentorMix, you can add the cli argument ```--mnet-gamma-p <value-of-hyperparameter-gamma-p> --mmix-alpha <value-of-hyperparameter-alpha >```. For example, ```--mnet-gamma-p 0.85 --mmix-alpha  0.2``` means using gamma-p  of 0.85 and alpha 0.2.



# DEMO
To run the demo script, run the following command form terminal:
```bash
mkdir -p images
mkdir -p pretrained_models/2DSpiral
python demo_spiral.py
```

# Results
Check the examples scripts in ```./example_scripts``` to get the results below. We use seeds - 12484, 32087 and 35416.
## Synthetic Noise in standard dataset
Test accuracy for CIFAR10 and CIFAR100 dataset averaged over 3 randomly chosen seeds. 
We show the Baseline Accuracy and Accuracy when Verifix is applied to the baseline model.

### CIFAR10 Dataset on VGG11_BN with 25% label noise (trained from Scratch)
| Method        | Baseline          | Verifix  (Val Set)        | Improvements  |
|---------------|:-------:          |:------------------------: |:----------:   | 
| Vanilla SGD   |$72.19 \pm 0.33$   |$83.38 \pm 0.47$           |$11.19$         |
| SAM           |$87.05 \pm 0.28$   |$87.17 \pm 0.38$           |$0.12$         |


### CIFAR10 Dataset on ResNet18 with 25% label noise (trained from Scratch)
| Method        | Baseline          | Verifix  (Val Set)        | Improvements  |
|---------------|:-------:          |:------------------------: |:----------:   | 
| Vanilla SGD   |$78.37 \pm 0.18$   |$86.96 \pm 0.14$           |$8.59$         |
| SAM           |$83.76 \pm 0.11$   |$87.74 \pm 0.38$           |$3.98$         |


### CIFAR100 Dataset on VGG11_BN with 25% label noise (trained from Scratch)
| Method        | Baseline          | Verifix  (Val Set)        | Improvements  |
|---------------|:-------:          |:------------------------: |:----------:   | 
| Vanilla SGD   |$49.03 \pm 0.69$   |$56.81 \pm 0.37$           |$7.78$         |
| SAM           |$54.31 \pm 0.21$   |$58.33 \pm 0.11$           |$4.02$         |


### CIFAR100 Dataset on ResNet18 with 25% label noise (trained from Scratch)
| Method        | Baseline          | Verifix  (Val Set)        | Improvements  |
|---------------|:-------:          |:------------------------: |:----------:   | 
| Vanilla SGD   |$57.60 \pm 0.17$   |$61.42 \pm 0.46$           |$3.82$         |
| SAM           |$58.82 \pm 0.76$   |$62.19 \pm 0.66$           |$3.37$         |




## Real-world noisy dataset
Test/Val accuracy for Mini-WebVision dataset averaged over 3 randomly chosen seeds. We show the Baseline Accuracy and Accuracy when Verifix is applied to the baseline model. 

### Mini-WebVision Dataset on InceptionResNetv2 (trained from Scratch)
| Method        | Baseline          | Verifix  (Val Set)        | Improvements  |
|---------------|:-------:          |:------------------------: |:----------:   |   
| Vanilla SGD   |$63.81 \pm 0.38$   |$64.96 \pm 0.53$           |$1.15$         |
| MixUp         |$65.01 \pm 0.40$   |$66.21 \pm 0.58$           |$1.20$         |
| MentorMix     |$65.35 \pm 0.65$   |$65.76 \pm 0.88$           |$0.41$         |
| SAM           |$65.68 \pm 0.57$   |$66.10 \pm 0.46$           |$0.43$         |
### WebVision1.0 Dataset on InceptionResNetv2 (trained from Scratch)
| Method        | Baseline      | Verifix (Val Set) | Improvements |
|---------------|:-------:      |:-------:          |:-------:|
| Vanilla SGD   |$64.86\pm0.53$ |$65.8 \pm 0.49$  |$0.40$    |

### Clothing1M Dataset on ResNet50 ( finetuned from PyTorch model pretrained on ImageNet1K)
| Method        | Baseline          | Verifix (Test Set)| Improvements  |
|---------------|:-------:          |:-------:          |:-------:      |
| Vanilla SGD   |$67.48 \pm 0.64$   |$70.11 \pm 0.76$   |$2.63$         |
| MixUp         |$67.89 \pm 0.63$   |$69.84 \pm 1.16$   |$1.94$         |

## License

This project is licensed under the [Apache 2.0 License](LICENSE).




# Citation
Kindly cite the [paper](https://arxiv.org/abs/2403.08618) if you use the code. Thanks!

### APA
```
Kodge, S., Ravikumar, D., Saha, G., & Roy, K. (2024). Verifix: Post-Training Correction to Improve Label Noise Robustness with Verified Samples. https://arxiv.org/abs/2403.08618
```
or 
### Bibtex
```
@misc{kodge2024verifix,
      title={Verifix: Post-Training Correction to Improve Label Noise Robustness with Verified Samples}, 
      author={Sangamesh Kodge and Deepak Ravikumar and Gobinda Saha and Kaushik Roy},
      year={2024},
      eprint={2403.08618},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```