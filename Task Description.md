
# Traffic Sign Recognition 

Imagine you are working in self-driving project as a machine learning engineer. One of the important problems for self-driving cars is to follow traffic signs. A car moves and collects video from the front camera. In each separate image you could localize the signs and after that understand what sign did we find.

We can assume that someone else is responsible for localizing and tracking signs, and your task is to do the classification part.

In this assignment you will be asked to develop a traffic sign recognition system. It should take a sign image with a small context around it and classify which of 43 signs it belongs to.

In addition to code you will send a report.

# Dataset

In this problem you will be given a benchmark dataset [The German Traffic Sign Recognition Benchmark: A multi-class classification competition](https://www.researchgate.net/publication/224260296_The_German_Traffic_Sign_Recognition_Benchmark_A_multi-class_classification_competition). 

![](https://www.researchgate.net/profile/Marc_Schlipsing/publication/224260296/figure/fig1/AS:648242664595473@1531564503803/A-single-traffic-sign-track.png)


Here is a information, [provided](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) by autors:

#### Overview

 

*    Single-image, multi-class classification problem
*    More than 40 classes
*    More than 50,000 images in total
*    Large, lifelike database
*    Reliable ground-truth data due to semi-automatic annotation
*    Physical traffic sign instances are unique within the dataset
    (i.e., each real-world traffic sign only occurs once)

 
#### Structure

The training set archive is structured as follows:

 *   One directory per class
 *   Each directory contains one CSV file with annotations ("GT-\<ClassID\>.csv") and the training images
 *   Training images are grouped by tracks
 *   Each track contains 30 images of one single physical traffic sign


#### Image format

 *   The images contain one traffic sign each
 *   Images contain a border of 10 % around the actual traffic sign (at least 5 pixels) to allow for edge-based approaches
 *   Images are stored in PPM format (Portable Pixmap, P6)
 *   Image sizes vary between 15x15 to 250x250 pixels
 *   Images are not necessarily squared
 *   The actual traffic sign is not necessarily centered within the image.This is true for images that were close to the image border in the full camera image
 *   The bounding box of the traffic sign is part of the annotatinos (see below)

 
#### Annotation format

Annotations are provided in CSV files. Fields are separated by ";"   (semicolon). Annotations contain the following information:

 *   Filename: Filename of corresponding image
 *   Width: Width of the image
  *  Height: Height of the image
  *  ROI.x1: X-coordinate of top-left corner of traffic sign bounding box
 *   ROI.y1: Y-coordinate of top-left corner of traffic sign bounding box
 *   ROI.x2: X-coordinate of bottom-right corner of traffic sign bounding box
 *   ROI.y2: Y-coordinate of bottom-right corner of traffic sign bounding box


The training data annotations will additionally contain

 *   ClassId: Assigned class label


# Task
Here is a sequence of steps you should do in this assignment. 

### Download

Download the [dataset](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html). Use *GTSRB_Final_Training_Images.zip* for training and *GTSRB_Final_Test_Images.zip* + *GTSRB_Final_Test_GT.zip* for testing.


And [here](http://benchmark.ini.rub.de/Dataset/GTSRB_Python_code.zip) is a script which reads the data, you can use it in your code.

### Transform images to same size


Train and test images differ in size and shape: some of them are square, others are rectangular.

To be able to produce the same number and kind of features from all images, we must ensure they have the same size. For this, you are asked to:

1) Padd rectangular images to square shape - add zero pixels to the shorter side. Alternatively, you can duplicate border pixels of the shorter side to get the required square shape.

2) When all images have the desired square shape, resize them to the same dimensionality - 30x30 pixels.

Include in the **Report** the result of padding: take any picture from the dataset and show its original and padded version. 


### Split data

Write your own implementation of the train-validation split (not using sklearn or any other ready implementation) and split your data in 80%-20% proportion. Pay attention to the fact that images are organized in tracks - there are 30 photos (video frames) for each sign taken from different distances. Make sure that images from the particular track end up being in either training or validation split, otherwise validation will be faulty (Think why?).

Don't forget to shuffle your data.

### Frequencies
Find class frequencies in the resulting training set.
Include a bar chart in your **Report**. It should be done after reducing the amount of data, if your did it.

### Augmentation

You can see from the bar-chat you've just produced that the classes are are highly imbalanced.

One of the ways to overcome this problem is to produce synthetic images from the existing ones and then add them to the training set (we already know their labels, right?). This is called data augmentation and it is a very common technique, especially in the images domain. Check these resources ([1](http://courses.d2l.ai/berkeley-stat-157/slides/3_14/image-augmentation.pdf), [2](https://medium.com/@ODSC/image-augmentation-for-convolutional-neural-networks-18319e1291c)) to see what are the possible ways to augment your data and choose two-three techniques that are appropriate in our particular case. In the **Report** you should *justify your choice* and provide examples of augmentation (original vs augmented images). 

NB! The number of images of each class should be the same after the augmenation.

![image alt](https://i.imgur.com/z4Kxfq8.png)
]

### Image normalization

Normalize every image separately such that pixel values are between 0 and 1. Do it for both train and valiation splits. 

### What are the features?

Finally, we represented all pixels as a number from 0 to 1. Now for each image we can ravel this matrix and present each sample as 1-dimentional vercor whith the same size.
Stack those vectors to matrix.

### Train model

Train RandomForest trafic signs classifier based on your training data.


### Evaluate
All estimates in this section should be described in the **Report**.

First, calculate the resulting *overall accuracy* on the test set. 

However, this estimate is not very representative for all classes - some are underrepresented. Hence, measure recall and precision for each class separately to analyze problem areas. 

Also, visualize samples that were classified incorrectly and add them to the **Report**


### Experiment and analyze!
Now, when you have the base model, it's time to do the performance analysis. 

1) Try training the model on non-augmented data. Compare models' performance. Analyze the effect of augmentation. Describe your findings in the **Report** 

2) Try changing the size of the images. Does increasing it results in better performance? Choose 5 different size options and include to the **Report** 2 plots: how accuracy and time depends on image size.



# Code structure

Organize your code in such way to make all experiments without rerun your script with different paramethers or copypasting large blocks of code inside it.


# Submission

You should submit 2 files without an archive:

1. *.py* script file with your code
2. *.pdf* report

# Report

Read [Report guidelines](https://hackmd.io/J89FQKJdRWmZCtku1pP04w?view). 

Everything with a label **Report** should be included, but not only. Make sure it satisfy guidelines.

# Policies

Taking others code or allow somebody else to see your code is not allowed. The same for your report. You might discuss ideas and problems with your classmate without sharing code and reprot.

# References

[The German Traffic Sign Recognition Benchmark: A multi-class classification competition](https://www.researchgate.net/publication/224260296_The_German_Traffic_Sign_Recognition_Benchmark_A_multi-class_classification_competition)


[Preprocessed images](https://www.kaggle.com/valentynsichkar/traffic-signs-preprocessed)

[Report guidelines](https://hackmd.io/J89FQKJdRWmZCtku1pP04w?view)



<!-- # Comments
### HOG -->

<!-- In computer vision we often use *descriptors* instead of the raw pixels data. It could be considered as a feature selection for computer vision problems. One of the most popular *global descriptor* is a *Histogram of Oriented Gradients*. For this task you might consider this algorithm as a blackbox. However, if interested, you can check [this link](https://www.learnopencv.com/histogram-of-oriented-gradients/), to get more details.  Find the appropriate paramethers which gives you a maximum accuracy. 

```
 import cv2
 hog = cv2.HOGDescriptor()
 im = cv2.imread(sample)
 h = hog.compute(im)
 ```
)
 -->