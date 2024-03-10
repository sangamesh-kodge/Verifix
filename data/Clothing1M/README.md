# Clothing1M
This project preprocess the [Clothing 1M Dataset](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Xiao_Learning_From_Massive_2015_CVPR_paper.pdf) and gives a classwise directory structure. The raw dataset can be obtained by following steps on [official repository](https://github.com/Cysu/noisy_label?tab=readme-ov-file)

### Dataset details

Check using ```find test/ -maxdepth 2 -type f | wc -l``` in terminal
 
- Number of nosiy train images - 1000000 
- Number of clean train images - 47570
- Number of val images - 10526
- Number of test images - 14313

### Below is the final directory structure:
The images in the directroy for each partition of the dataset are arranged in the directory with q00xx where xx is the class number from 0-13.
```
└── Clothing1M
    ├── clean_train
    |   ├── q0000
    |   ├── q0001
    |   └── ...
    |   └── q0013
    |
    ├── noisy_train
    |   ├── q0000
    |   ├── q0001
    |   └── ...
    |   └── q0013
    |
    ├── val
    |   ├── q0000
    |   ├── q0001
    |   └── ...
    |   └── q0013
    |
    ├── test
    |   ├── q0000
    |   ├── q0001
    |   └── ...
    |   └── q0013
    |
    ├── xxxx
    ├── xxxx
    ├── xxxx
    ├── xxxx
    ├── xxxx
    └── creat_dataset.sh
    └── helper.py
    └── Readme.md
    └── Citation.cff
    └── LICENSE
```



## Instructions
1. Get access to dataset following steps on [official repository](https://github.com/Cysu/noisy_label?tab=readme-ov-file)
2. Download the dataset from the download link obtained in step 1 in this directory. (Or move data after download. )
3. Unzip files in ```images/``` directory using ```tar -xf <file_name.tar>```. Additionally unzip file in the root directory using ```unzip <file_name.zip>```. (See the commented lines 20-29 in create_dataset.sh). At this stage the cloned repository should have ```images/``` directory containing raw images and files with the label information for each image. 
4. Run the following command in your terminal/command contraining the cloned repository 
 
    ```bash
    sh create_dataset.sh
    ```