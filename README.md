# A class-based styling approach for Real-Time Doamin Adaptation in Semantic Segmentation
This project is part of The Machine Learning & Deep Learing cource (MLDL 2021 ) , Politecnico Di Torino , Master Of Data Science & Engineering Program 

# Implementation Details :
- The Real-Time semantic segmentation Model is based on [**BiSeNet**](https://github.com/MatteoM95/Real-time-Domain-Adaptation-in-Semantic-Segmentation/tree/MatteoBranch/Master%20code%20original/BiseNetv1-master) and the domain adaptation model is based on [**AdaptSegNet**](https://github.com/MatteoM95/Real-time-Domain-Adaptation-in-Semantic-Segmentation/tree/MatteoBranch/Master%20code%20original/AdaptSegNet-master)
- The pytorch version used is 0.4.1 and python version 3.6
- All the experiments were run on Google Colab Tesla P100 15 GB GPUs

## Segmentation Network: BiSeNet
For the segmentation network, we adopt [BiSeNet](https://arxiv.org/abs/1808.00897?context=cs) (Bilateral Segmentation Network) that
is a state-of-the-art approach to Real-time Semantic Segmentation. With this network we have the possibility to choose between ResNet-18 or ResNet-101 as our segmentation baseline model.
BiSeNet is composed by two main components: Spatial Path (SP) and Context Path (CP).
The aim of the spacial path is to encode significant spatial information by preserving the spatial information contained in the original input image. This component is composed by three convolution layers with stride = 2, followed by a batch normalization and a ReLU.
On the other hand, context path is used to obtain a large receptive field using a light-weight model and global average pooling. To better refine the features at each stage an Attention refinement module (ARM) is used.
The output of these two components could be simply summed up together. But to combine features efficiently a specific module called Feature Fusion Module (FFM) is introduced.

## Class-Based Styling (CBS)
Our proposed expansion to the previously described architecture was inspired by the [styling technique](https://arxiv.org/abs/2003.00867) used by Kim & Byun, and also to the [real-time Class-Based Styling](https://arxiv.org/abs/1908.11525) (CBS) method displayed by Kurzman et al. . When training a segmentation network on a synthetic dataset, it’s always possible for the network to overfit on specific textures, especially when they are repeat-
edly used in different images. Using a neural style transfer algorithm is useful to diversify the textures and improve the
final performances. In [22], the stylized datasets were prepared beforehand using the [StyleSwap](https://www.arxiv-vanity.com/papers/1612.04337/) algorithm, which cannot be used in a real-time setting. Instead, we exploited the style-transfer algorithm used in CBS to stylize specific classes in a
live video. This algorithm was introduced by [Johnson et al.](https://arxiv.org/abs/1603.08155) in 2016, and it’s able to train a model to translate images to different styles in real-time. We introduced the style-transfer step in our architecture as a data augmentation step. Every
time an image is taken from the source dataset, a style model is selected and is applied to stylize the image.

Several style models were trained (see Fig. 2), and they can be divided into two main categories:
- CamVid styles: these models were trained on images taken from the CamVid dataset. These models were used to change the synthetic textures and make the segmentation network more familiar with the target images colors.
- Texture styles: these models were trained on different images and textures that are not easily seen on a street. These models were used to prevent the segmentation network to overfit on the synthetic textures.

![Fig. 2](https://github.com/MatteoM95/Real-time-Domain-Adaptation-in-Semantic-Segmentation/blob/MatteoBranch/Images/style_recap.jpg)

The proposed class-based styling algorithm is very similar to the CBS algorithm, and works as follows. For each run, a set of style models I<sub>s</sub> is used. Consider an image I<sub>s</sub> , extracted from the source set X<sub>s</sub>, and its ground-truth segmentation maps Y<sub>s</sub>. Then,
given a style set T , we randomly select a style *t*. We also randomly select a subset of source semantic classes where each
class has a probability *p* of being selected. *p* can change or remain constant throughout the whole training, the probability
used in our runs is better explained in section IV.
After this initial step, the whole image is stylized using the style *t*. Then a mask is created based on Y<sub>s</sub> , and is used to identify the selected classes that will be stylized from the others. The resulting image consists of the original image in background, and all the stylized classes in foreground.

## Discriminator: AdaptSegNet
the discriminator network was taken from [AdaptSegNet](https://arxiv.org/abs/1802.10349), and consists of 5 convolutional layers with kernel
4 × 4, stride 2 and channel numbers {64, 128, 256, 512, 1}. Each convolutional layer (with the exception of the last layer) is followed by a Leaky ReLU parametrized by 0.2.

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
