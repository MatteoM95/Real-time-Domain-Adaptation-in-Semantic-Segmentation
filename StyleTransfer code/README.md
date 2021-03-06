# Fast-neural-style :city_sunrise: :rocket:
This repository contains a pytorch implementation of an algorithm for artistic style transfer. The algorithm can be used to mix the content of an image with the style of another image. For example, here is a photograph of a door arch rendered in the style of a stained glass painting.

The original model uses the method described in [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155) along with [Instance Normalization](https://arxiv.org/pdf/1607.08022.pdf). 

<p align="center">
    <img src="images/style-images/camvid5.jpg" height="200px">
    <img src="images/content-images/idda.jpg" height="200px">
    <img src="images/output-images/camvid5.jpg" height="200px">
</p>

### Contents
- [Requirements](#Requirements)
- [Usage](#Usage)
- [Models](#Models)

<a name="Requirements"/>

## Requirements
The program is written in Python, and uses [pytorch](http://pytorch.org/), [scipy](https://www.scipy.org). A GPU is not necessary, but can provide a significant speed up especially for training a new model. Regular sized images can be styled on a laptop or desktop using saved models.

<a name="Usage"/>

## Usage
**Stylize image**
```
python neural_style/neural_style.py eval --content-image </path/to/content/image> --model </path/to/saved/model> --output-image </path/to/output/image> --cuda 0
```
* `--content-image`: path to content image you want to stylize.
* `--model`: saved model to be used for stylizing the image (eg: `mosaic.pth`)
* `--output-image`: path for saving the output image.
* `--content-scale`: factor for scaling down the content image if memory is an issue (eg: value of 2 will halve the height and width of content-image)
* `--cuda`: set it to 1 for running on GPU, 0 for CPU.

**Train model**
```bash
python neural_style/neural_style.py train --dataset </path/to/train-dataset> --style-image </path/to/style/image> --save-model-dir </path/to/save-model/folder> --epochs 2 --cuda 1
```

There are several command line arguments, the important ones are listed below
* `--dataset`: path to training dataset, the path should point to a folder containing another folder with all the training images. A smaller dataset of IDDA is used to train the model
* `--style-image`: path to style-image.
* `--save-model-dir`: path to folder where trained model will be saved.
* `--cuda`: set it to 1 for running on GPU, 0 for CPU.

Refer to ``neural_style/neural_style.py`` for other command line arguments. For training new models you might have to tune the values of `--content-weight` and `--style-weight`. The mosaic style model shown above was trained with `--content-weight 1e5` and `--style-weight 1e10`. The remaining 3 models were also trained with similar order of weight parameters with slight variation in the `--style-weight` (`5e10` or `1e11`).

<a name="Models"/>

## Models

Model are store in the folder [models](https://github.com/MatteoM95/Real-time-Domain-Adaptation-in-Semantic-Segmentation/tree/MatteoBranch/StyleSwap%20code/models).

All the model are trained tuning the `--style-weight` and `--content-weight`. The images are cropped to 256 pixel per square adding `--image-size 256`:

```bash
!python neural_style/neural_style.py train --epochs 4 --style-weight 3e9 --image-size 256 --style-image images/style-images/camvid16.jpg --dataset ../datasets/IDDA_2 --save-model-dir ./models --checkpoint-model-dir ./checkpoints --cuda 1
```

<div align='center'>
  <img src='images/content-images/idda.jpg' height="174px">		
</div>

<div align='center'>
  <img src='images/style-images/camvid7.jpg' height="174px">
  <img src='images/output-images/camvid7.jpg' height="174px">
  <br>
  <img src='images/style-images/sketch.jpg' height="174px">
  <img src='images/output-images/output5.jpg' height="174px">
  <br>
  <img src='images/style-images/rain-princess.jpg' height="174px">
  <img src='images/output-images/output2.jpg' height="174px">
  <br>
  <img src='images/style-images/sidewalk.jpg' height="174px">
  <img src='images/output-images/outputSidewalk.jpg' height="174px">
  
</div>
