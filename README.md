# A class-based styling approach for Real-Time Doamin Adaptation in Semantic Segmentation
This project is part of The Machine Learning & Deep Learing cource (MLDL 2021) , Politecnico Di Torino , Master Of Data Science & Engineering Program. The project was supervised by prof. Barbara Caputo and prof. Antonio Tavera. 

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


![Fig. 2](https://github.com/MatteoM95/Real-time-Domain-Adaptation-in-Semantic-Segmentation/blob/MatteoBranch/Images/style_recap.jpg "Fig. 2")


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

## Run the code: Colab Setting
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
or training with Class-Based Styling 
```
!python train_DA_CBS.py
``` 

## Test Model
```
python test.py --
```
Or watch the result with TensorBoardX
```
%tensorboard --logdir ./runs
```

## Results
In the first step, we exploited the BiSeNet network to perform semantic segmentation on the CamVid dataset. The network was trained with two different backbones, ResNet-18 and ResNet-101, and their performances at 50 and 100 epochs were compared in order to find the best configuration and get an upper bound for the next steps results. The dice loss was used in all these trainings. The results of this experiment are presented in the first part of Table. It’s possible to see that training the network for 100 epochs seems to yield better results than training it for 50 epochs, while there is no
significant difference between using ResNet-18 and ResNet-101 as the backbone. After the first results were obtained, the network was trained
again while using two different data augmentation techniques on the dataset. The implemented techniques are horizontal flips and Gaussian blur on the whole image. Each time an image was taken from the dataset, there was a 1/4 probability for each of the following scenarios to take place: no change
was made, the image was flipped horizontally, the image was blurred, the image was both flipped horizontally and blurred.
This addition improved the results especially when using the ResNet-101 backbone as shown in the second part of Table. After assessing these results, we decided to keep using the ResNet-101 backbone to conduct further experiments, and we moved to the second step of our experiment.

In the second step, the network architecture was expanded in order to perform unsupervised domain adaptation on two different datasets. In the following experiments, we used the IDDA dataset as the labeled source dataset, while the CamVid dataset was used as the unlabeled target dataset. We also introduced a discriminator network to perform adversarial training between target and source domain.
We ran multiple trainings to test different configurations: during each epoch, the network was trained with all the images from the CamVid dataset and a random subset of images taken from the IDDA dataset to match the size of CamVid. The most significant results of our runs are presented in the last part of Table I.
At first, the network was trained for 100 epochs to compare dice and crossentropy losses. The best results were obtained using the crossentropy loss that was able to reach a pixel precision of 66.1 and a mIoU of 30.8. Considering that the best results of both runs were achieved in the first 50 epochs, and didn’t significantly improve over the last 50 epochs, the training of the next runs was reduced to 50 epochs in total.
By reducing the number of epochs to 50, the performance improved to a pixel precision score of 67.6 and a mIoU score of 31.9 with crossentropy.

![Table I](https://github.com/MatteoM95/Real-time-Domain-Adaptation-in-Semantic-Segmentation/blob/MatteoBranch/Images/TableI.png "Table I")

After assessing that the best performances can be reached training over 50 epochs using the crossentropy loss, this configuration was used to test the expansion described in previous section.
For each image, the probability p to stylize a class was set to 0.5 for the whole duration of the training. The network was trained several times with different combinations of style models t, that can be summed up as follows:
- 1 single CamVid style,
- 1 single Texture style,
- 5 different CamVid styles,
- 5 different Texture styles,
- 5 different CamVid styles and 5 different Texture styles.
All the results are reported in the following Table II. It’s possible to see that while some runs didn’t improve the performances obtained without styling, other runs were able to significantly improve both the mIoU and pixel precision scores. In particular, the runs with multiple styles achieved better results overall. Unexpectedly, the best results were obtained by the network trained with the Camvid5 style only, which was able to reach a mIoU
score of 34.7 and a pixel precision score of 68.7 at the end of the training.

![Table II](https://github.com/MatteoM95/Real-time-Domain-Adaptation-in-Semantic-Segmentation/blob/MatteoBranch/Images/TableII.png "Table II")

## Final Discussion
With the proposed expansion, we were able to improve the baseline results achieved by the network used in the second step of our experiment. We believe that there is still room for improvements. For example, it could be interesting to perform a finetuning of the styles based on the single class IoU scores and use the best styles for each specific class. In particular, the runs that used mixed styles achieved improved results, but still underperformed while predicting harder classes such as the bicycle class where single styles achieved better results.
Another aspect that needs to be taken into account is the consistency and reproducibility of the results. In the described runs, only a random subset of the source dataset was used to train the network in each epoch. Furthermore, the proposed expansion includes stochastic features such as the style model
being and the stylized class subset being selected randomly for each image. It could be interesting to reproduce the experiments and train the network with the entire source dataset.

The [paper](https://github.com/MatteoM95/Real-time-Domain-Adaptation-in-Semantic-Segmentation/blob/MatteoBranch/Report_Real%20time%20Domain%20Adaptation%20in%20Semantic%20Segmentation.pdf) of the project is available in this repository

<a name="contributors" />

### Contributors

<a href="https://github.com/Gabrysse/CBS-realtimeDA-semSeg/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Gabrysse/CBS-realtimeDA-semSeg" />
</a>
