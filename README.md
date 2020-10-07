# HISNav
`HISNav` - Habitat-based Instance segmentation, Slam and Navigation Dataset

`HISNav` is a dataset, which consists of various robot movements tracks, recorded in virtual environment Habitat. Tracks were built on 49 unique scenes from Matterport3D ([pdf](https://arxiv.org/pdf/1709.06158.pdf)) that present rooms with different styles. Each scene has no more than 5 trajectories with 3 different levels of noise in camera images and in actions.

We pursue the goal to research the steadiness of the developed framework to the noise. We use three levels of noise in images: without noise, light Gaussian noise , strong Gaussian noise. The examples of images from the dataset are shown here:

![](imgs/Figure_Dataset_Images.jpg)
Examples of images from HISNav Dataset with three levels of noise
