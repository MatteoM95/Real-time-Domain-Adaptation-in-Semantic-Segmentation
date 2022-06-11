# A class-based styling approach for Real-Time Doamin Adaptation in Semantic Segmentation
This project is part of The Machine Learning & Deep Learing cource (MLDL 2021 ) , Politecnico Di Torino , Master Of Data Science & Engineering Program 
## BiseNet
Starting code for the student belonging to the project "Real-time domain adaptation in semantic segmentation" <br>
# Implementation Details :
- The Domain adaptation Model is based on [**BiSeNet**](https://github.com/MatteoM95/Real-time-Domain-Adaptation-in-Semantic-Segmentation/tree/MatteoBranch/Master%20code%20original/BiseNetv1-master) and the semantic segmentation model is based on [**AdaptSegNet**](https://github.com/MatteoM95/Real-time-Domain-Adaptation-in-Semantic-Segmentation/tree/MatteoBranch/Master%20code%20original/AdaptSegNet-master)
- The pytorch version used is 0.4.1 and python version 3.6
- All the experiments were run on Google Colab Tesla P100 15 GB GPUs


## Dataset  
- Download [CamVid dataset](https://github.com/MatteoM95/Real-time-Domain-Adaptation-in-Semantic-Segmentation/tree/MatteoBranch/Datasets/CamVid)
- Download [IDDA dataset](https://github.com/MatteoM95/Real-time-Domain-Adaptation-in-Semantic-Segmentation/tree/MatteoBranch/Datasets/IDDA)
- Note: classes_info.json file needs to be modified by changing the first couple of brakets '[]' to {} and deleting the last comma.

## Colab Setting
```
from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/MyDrive/ProjectFolder
!pip install tensorboardX
%load_ext tensorboard
```
## Train Model
```
!python train.py
```  

## Test Model
```
python test.py --
```
Or watch the result with TensorBoardX
```
%tensorboard --logdir ./runs
```

## Project Paper :
Final [paper](https://github.com/MatteoM95/Real-time-Domain-Adaptation-in-Semantic-Segmentation/blob/MatteoBranch/Report_Real%20time%20Domain%20Adaptation%20in%20Semantic%20Segmentation.pdf) of the project

<a name="contributors" />

### Contributors

<a href="https://github.com/Gabrysse/CBS-realtimeDA-semSeg/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Gabrysse/CBS-realtimeDA-semSeg" />
</a>
