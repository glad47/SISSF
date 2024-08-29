

# Improving Social Information Sensitive Semantic Fusion Using Contrastive Learning in Conversational Recommender System
This is my implementation to preprocess the dataset of the paper SISSF 2024 Paper [**"Improving Social Information Sensitive Semantic Fusion Using Contrastive Learning in Conversational Recommender System"**](https://xxxx)
Mohammed Abdulaziz, Zhang Mingwei



## Introduction
We have used the dataset used by the [C2CRS](https://arxiv.org/abs/2201.02732) compared with the original [ReDial dataset](https://papers.nips.cc/paper/2018/file/800de15c79c8d840f4e78d3af937d4d4-Paper.pdf) to include some addtional information needed by our framework, we have used [NGCF](https://arxiv.org/abs/1905.08108) to infer ratings, we have include a new modified version to work with the new dataset gernerated, you can find rar folder NGCF-PyTorch-master.rar 


Hope it can help you!

## Environment Requirement
The code has been tested under Python 3.12.5. The required packages are as follows:

* pytorch == 1.3.1
* scipy == 1.14.0
* pandas == 2.2.2
* numpy == 2.0.1
* contractions == 0.1.73
* scikit-learn == 1.1.0

## Instructions to Run the Code Correctly


## Datasets
We use the input dataset to generate the new dataset with more information, all of which have been uploaded to [Google Drive](https://drive.google.com/file/d/1oB3ZbC8l8wcgZEgdwGdkEEDWuS2mVC_t/view?usp=sharing) and [Baidu Netdisk](https://pan.baidu.com/s/1A8xtR0h-1jgCvbkV1m3KdQ) (password: fggj). after unrar the folder copy the input, `outputNGCF`, `outputSISSF` directly to the main folder of the `ExtractProject`


## Steps to Generate the Dataset
You can directly get the input and output datasets. To download the input datasets:

* Download the dataset and move only the `input` folder to the main folder of the ExtractProject. The `input` folder contains the datasets used by C2CRS and ReDial.

* Run `extractNGCF.py`, which will generate datasets and store them in the outputNGCF folder. Copy the generated files into the `NGCF project` folder `Data/amazon-book`.

* Run the `NGCF project`. After you finish training, run `inferRatings.py` to generate `ratings.xlsx`.

* Copy ratings.xlsx to the input folder of the ExtractProject.


* Run `extractDatasetSiSSF.py`, which generates the `outputSISSF` folder that contains the modified dataset used by the SISSF project. 

* Move these files into the `data/dataset` folder under the `SISSF` folder to prepare the project for semantic fusion training.

* Copy `item_list.txt` and `user_list.txt` from `outputNGCF` into the `data/dataset` folder under the `SISSF` folder to prepare the project for semantic fusion training.