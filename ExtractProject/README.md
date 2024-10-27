



# Improving Social Information Sensitive Semantic Fusion Using Contrastive Learning in Conversational Recommender System
This is our implementation to preprocess the dataset of the paper SISSF 2024 Paper [**"Improving Social Information Sensitive Semantic Fusion Using Contrastive Learning in Conversational Recommender System"**](https://xxxx)
Abdulaziz Mohammed, Mingwei Zhang, Gehad Abdullah Amran, Husam M. Alawadh, Wang Ruizhe, Amerah ALabrah, and Ali A. Al-Bakhrani



## Introduction
We have used the dataset used by the [C2CRS](https://arxiv.org/abs/2201.02732) compared with the original [ReDial dataset](https://papers.nips.cc/paper/2018/file/800de15c79c8d840f4e78d3af937d4d4-Paper.pdf) to include some additional information needed by our framework for [ReDial dataset], For [INSPIRED dataset](https://aclanthology.org/2020.emnlp-main.654.pdf) we have used the MovieTrucker Project first to add the necessary information about mentioned movies in the conversations, who mentioned it, whether the user liked, suggested or seen it, similar to the [ReDial dataset](https://papers.nips.cc/paper/2018/file/800de15c79c8d840f4e78d3af937d4d4-Paper.pdf), next, we also have convert [INSPIRED dataset](https://aclanthology.org/2020.emnlp-main.654.pdf) to exact [ReDial dataset](https://papers.nips.cc/paper/2018/file/800de15c79c8d840f4e78d3af937d4d4-Paper.pdf) structure. Then, we used [NGCF](https://arxiv.org/abs/1905.08108) to infer ratings for both datasets. We have included a new modified version to work with the new dataset generated. You can find the rar folder NGCF-PyTorch-master.rar 


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
We use the input dataset to generate the new datasets with more information, all of which have been uploaded to [Google Drive](https://drive.google.com/file/d/1CHN0zI8663EedQx1djIdHmiRkJSigMss/view?usp=sharing) and [Baidu Netdisk](https://pan.baidu.com/s/13lVffOCG-NtakT0dwpHx9w) (password: ttmz). after unrar the folder copy its content to the `input` folder directly to the main folder of the `ExtractProject`


## Steps to Generate the ReDial Dataset
You can directly downlaod output datasets. To generate ReDail datasets:

* Download the dataset and move only the `input` folder to the main folder of the ExtractProject. The `input` folder contains the datasets used by C2CRS and ReDial.

* Run `extractNGCF.py`, which will generate datasets and store them in the `outputNGCF` folder. Copy the generated files into the `NGCF project` folder `Data/gowalla`.

* Run the `NGCF project`. After you finish training, run `inferRatings.py` to generate `ratings.xlsx`.

* Copy ratings.xlsx to the input folder of the `ExtractProject/input/redial`.

* Move `user_list.txt` and `item_list.txt` under `outputNGCF` folder to `ExtractProject/input/redial`. 

* Run `extractDatasetSiSSF.py`, which generates the `outputSISSF` folder that contains the modified dataset used by the SISSF project. 

* Move these files into the `data/dataset/ReDial` folder under the `SISSF` folder to prepare the project for semantic fusion training.

## Steps to Generate the INSPIRED Dataset
You can directly downlaod output datasets. To generate INSPIRED datasets:

* Download the dataset and move only the `input` folder to the main folder of the ExtractProject. The `input` folder contains the datasets used by C2CRS and ReDial.

* Run `extractConvInspired.py` under `MovieTracker` folder, which will generate two files `conversations.json` and `mentionedMovies.json` under `MovieTracker` folder. 

* Run `server.py`, and go inside `movie-tracker` react project build it and run it. compelete processing all conversations by adding the meta data. this step will generate a `tarck.json` file copy this file into `input/inspired`.

* Run `convertInspired.py`,this will generate `input/inspired/conversations.json` that will be used further.

* Run `extractNGCFyInspired.py`, which will generate datasets and store them in the `outputNGCFInspired` folder. Copy the generated files into the `NGCF project` folder `Data/gowalla`.

* Run the `NGCF project`. After you finish training, run `inferRatings.py` to generate `ratings.xlsx`.

* Copy ratings.xlsx to the input folder of the `ExtractProject/input/inspired`.

* Move `user_list.txt` and `item_list.txt` under `outputNGCFInspired` folder to `ExtractProject/input/inspired`. 

* Run `extractDatasetSISSFInspired.py`, which generates the `outputSISSFInspired` folder that contains the modified dataset used by the SISSF project. 

* Move these files into the `data/dataset/INSPIRED` folder under the `SISSF` folder to prepare the project for semantic fusion training.