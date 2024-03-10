# Mini-WebVision
This project preprocess the Google images partition of [WebVision 1.0 Dataset](https://data.vision.ee.ethz.ch/cvl/webvision/dataset2017.html) to obtain Mini-WebVision dataset and gives a directory structure [ImageNet1k dataset](https://www.image-net.org/). 

[Mini-WebVision](https://arxiv.org/abs/1911.09781) - contains about 61K Google images on the first 50 classes from the WebVision dataset. 
- Number of train images - 61234
- Number of val images - 2500 (50 per class)



Below is the final directory structure for this project:
```
└── Mini-WebVision
    ├── train
    |   ├── nxxxxxxxx
    |   ├── nxxxxxxxx
    |   └── ...
    |
    └── val
    |   ├── nxxxxxxxx
    |   ├── nxxxxxxxx
    |   └── ...
    |
    └── info
    |   ├── xxxx
    |   ├── xxxx
    |   └── ...
    |
    └── create_MiniWebVision_as_ImageNet.sh
    └── helper.py
    └── Readme.md
    └── Citation.cff
    └── LICENSE
```


## Instructions

1. Clone this repository
2. Navigate to the root of this project
3. Run the following command in your terminal/command prompt: 
 
    ```bash
    sh create_MiniWebVision_as_ImageNet.sh
    ```


## Expected Terminal logs
```
(base) Mini-WebVision >sh create_MiniWebVision_as_ImageNet.sh 
--2024-02-07 11:04:19--  https://data.vision.ee.ethz.ch/cvl/webvision/google_resized_256.tar
Resolving data.vision.ee.ethz.ch (data.vision.ee.ethz.ch)... 129.132.52.178, 2001:67c:10ec:36c2::178
Connecting to data.vision.ee.ethz.ch (data.vision.ee.ethz.ch)|129.132.52.178|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 16980316160 (16G) [application/x-tar]
Saving to: ‘google_resized_256.tar’

100%[================================================================================>] 16,980,316,160 16.2MB/s   in 11m 59s

2024-02-07 11:16:19 (22.5 MB/s) - ‘google_resized_256.tar’ saved [16980316160/16980316160]

--2024-02-07 11:16:19--  https://data.vision.ee.ethz.ch/cvl/webvision/val_images_256.tar
Resolving data.vision.ee.ethz.ch (data.vision.ee.ethz.ch)... 129.132.52.178, 2001:67c:10ec:36c2::178
Connecting to data.vision.ee.ethz.ch (data.vision.ee.ethz.ch)|129.132.52.178|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 873574400 (833M) [application/x-tar]
Saving to: ‘val_images_256.tar’

100%[===================================================================================>] 873,574,400 24.2MB/s   in 35s    

2024-02-07 11:16:55 (23.7 MB/s) - ‘val_images_256.tar’ saved [873574400/873574400]

--2024-02-07 11:16:55--  https://data.vision.ee.ethz.ch/cvl/webvision/info.tar
Resolving data.vision.ee.ethz.ch (data.vision.ee.ethz.ch)... 129.132.52.178, 2001:67c:10ec:36c2::178
Connecting to data.vision.ee.ethz.ch (data.vision.ee.ethz.ch)|129.132.52.178|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 190914560 (182M) [application/x-tar]
Saving to: ‘info.tar’

100%[===================================================================================>] 190,914,560 24.6MB/s   in 8.3s   

2024-02-07 11:17:04 (21.9 MB/s) - ‘info.tar’ saved [190914560/190914560]

----------------------------------------------------------------
Creating directory structure similar to ImageNet for training dataset
----------------------------------------------------------------
----------------------------------------------------------------
Creating directory structure similar to ImageNet for val dataset
----------------------------------------------------------------
----------------------------------------------------------------
Removing Redundant files.
----------------------------------------------------------------
----------------------------------------------------------------
Mini-WebVision Dataset Processed!
----------------------------------------------------------------
```
