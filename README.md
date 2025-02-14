# MULocDeep
MULocDeep is a deep learning model for protein localization prediction at both sub-cellular level and sub-organellar level. It also has the ability to interpret localization mechanism at a amino acid resolution. Users can go to our webserver at https://www.mu-loc.org/ for localiztion prediction and visualization. This repository is for running MuLocDeep locally.
## Installation

  - Installation has been tested in Windows, Linux and Mac OS X with Python 3.7.4. 
  - Keras version: 2.3.0
  - For predicting, GPU is not required. For training a new model, the Tensorflow-gpu version we tested is: 1.13.1
  - Users need to install the NCBI Blast+ for the PSSM. The download link is https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/. The version we tested is 2.9.0. The database can be downloaed at https://drive.google.com/drive/folders/19gbmtZAz1kyR76YS-cvXSJAzWJdSJniq?usp=sharing. Put the downloaded 'db' folder in the same folder as other files in this project.

## Running on GPU or CPU

If you want to use GPU, you also need to install [CUDA]( https://developer.nvidia.com/cuda-toolkit) and [cuDNN](https://developer.nvidia.com/cudnn); refer to their websites for instructions.
 
## Usage:
### Train a model
To train a MULocDeep model: (Here we use the non-redundant training data, since the entire training data is too big to upload to Github)
```sh
python train.py --lv1_input_dir ./data/UniLoc_train_40nr/ --lv2_input_dir ./data/UniLoc_train_40nr/ --model_output ./model_xxx/ --MULocDeep_model
```
To train a variant model (eg. using deeploc training data to train a 10-class classification model):
```sh
python train.py --input_dir ./data/deeploc_40nr_8folds/ --model_output ./var_model_xxx/
```
### Predict protein localization
Predicting protein localization using the pretrained model is fairly simple. There are several parameters that need to be specified by users. They are explained as below:
  - --input filename: The sequences of the proteins that users want to predict. Should be in fasta format.
  - --output dirname: The name of the folder where the output would be saved.
  - --existPSSM dirname: This is optional. If the pssm of the protein sequences are already calculated, users can specify the path to that folder. This will save a lot of time, since calculating pssm is time consuming. Otherwise, the prediction program will automaticlly start to generate the pssm for the prediction.
  - --att: Add this if users want to see the attention visualization. It is for interpreting the localization mechanism. Amino acids with high attention weights are considered related to the sorting signal.
  - --no-att: Add this, no attention visualization figure.

#### Example (using our provided example data): 


For GPU usage:
```sh
python predict.py -input ./wiki_seq.txt -output ./test --att --gpu
```
For CPU usage:
```sh
python predict.py -input ./wiki_seq.txt -output ./test --att --cpu
```

#### Note
  - The prediction result is given by the non-redundant model. This is for reproducibility and method comparison using our non-redundant dataset (UniLoc_train_40nr). For better prediction performance, users could use the model trained using the UniLoc_train dataset (this model and the corresponding predict code are in the "gpu_model_UniLoc_train" folder, the performance of this model is also claimed in the MULocDeep paper).
  - The GPU model is trained using CuDNNLSTM, while the CPU model is trained using LSTM. So, the prediction results are slightly different between the two models. The results shown in the MULocDeep paper were obtained using the GPU model.
  - Users are encoraged to use our webserver at https://www.mu-loc.org/ (the model used for the webserver will be updated regularly). The latest version now supports species-specific prediction. The performance is better than the general MULocDeep model when species information is known.

## Citation
MULocDeep web service for protein localization prediction and visualization at subcellular and suborganellar levels, Nucleic Acids Research, 2023, 10.1093/nar/gkad374  
  
MULocDeep: A deep-learning framework for protein subcellular and suborganellar localization prediction with residue-level interpretation, Computational and Structural Biotechnology Journal, Volume 19,
2021, Pages 4825-4839, ISSN 2001-0370, https://doi.org/10.1016/j.csbj.2021.08.027.  


## Contacts
Our webserver/stand-alone package are free of charge for academic users; other users are requested to contact the corresponding author of this work at
  - Email: xudong@missouri.edu

If you ever have any question or problem using our tool, please contact us.
  - Email: yjm85@mail.missouri.edu
