# A class-based styling approach for Real-Time Doamin Adaptation in Semantic Segmentation
This project is part of The Machine Learning & Deep Learing cource (MLDL 2021 ) , Politecnico Di Torino , Master Of Data Science & Engineering Program 
## BiseNet
Starting code for the student belonging to the project "Real-time domain adaptation in semantic segmentation" <br>
# Implementation Details :
- The Semantic Segmentation Model is based on <b>BiSeNet<b>, pytorch 0.4.1 and python 3.6
- All the experiments were run on Google Colab Tesla P100 15 GB GPUs


## Dataset  
- Download [CamVid dataset] (https://drive.google.com/file/d/1CKtkLRVU4tGbqLSyFEtJMoZV2ZZ2KDeA/view?usp=sharing)
- Download [IDDA dataset](https://drive.google.com/file/d/1GiUjXp1YBvnJjAf1un07hdHFUrchARa0/view?usp=sharing)
- Note: classes_info.json file needs to be modified by changing the first couple of brakets '[]' to {} and deleting the last comma.

  
## Train Model
```
python train.py --
```  

## Test Model
```
python test.py --
```


# Project Paper :

