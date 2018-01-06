# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* example_code_training_ec2_gpu.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results
* README.md Readme file
* run1.mp4 Video in autonomous mode

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 24 and 64 (model.py lines 106-110) 

The model includes RELU layers to introduce nonlinearity (code line 106-110), and the data is normalized in the model using a Keras lambda layer (code line 104). 

#### 2. Attempts to reduce overfitting in the model

Both of my training and validation accuracy are low.
Therefore, I am not using any dropout layer in my code.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 117).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used the given sample data for this project instead of my collected data.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use NVIDIA self-driving architecture.

My first step was recording several laps in the training mode and spliting the data into a training and validation set in my code.
In my code, I started with a two layers convolution neural network model, since I want to see if it can load the data correctly and drive the car in the autonomous mode, even though not as good as expect. However, I found using the given sample data having a better performance so I decided to use given sample data instead of my collected data set

Then I start adding Lambda layer for normalization and applying the LeNet architecture in my code since it was doing a great job in my previous work. However, my car was driving off the road in a few seconds after running in the autonomous mode.

I added augmented data and tried again. My car can driving slowly since it is changing the steering frequently. Finally, it will run off the road before getting on the bridge or stuck in the road ledge for some reason.

Then I added multiple cameras and found the result become worse. So I set the training data set back to center camera data set.

Starting from adding multiple cameras, the computation time increased heavily. So I need to reduce the training data set or architecture or move the computation onto AWS EC2. I choosed to move my computation onto AWS EC2. And I also adding cropping image and using generator in my code to accelerate the processing time. It is very impressive, running on my machine 1 epoch took 1800s, on AWS took 300s, with cropping and generator took 30~50s.

However, my car still running off the road therefore I changed the architecture from LeNet to NVIDIA.
It works well! It still running off the road but can recover to the road and this is the first time my car can run the full lap.

To improve the image quality and avoid any imcompatible, I created a bgr2rgb function in my code to handle the difference between the drive.py and cv2.imread.

Then I readded multiple cameras and used correction to tune the angle in left or right camera data set then augmented the data.
However, the result is not as good as expect or even worse, my car will fell off the bridge or running out of the track in the first sharp left turn after passing the bridge. So I disabled the multiple cameras use in my code.

Finally, I was changing the Simulator settings to see if there is any improvement and using the following settings works fine to my model. At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

* Simulator settings
* Screen resolution: 1280 x 960
* Graphics quality: Simple
* Select monitor: Display 1 (Left)
* Windowed: Checked



#### 2. Final Model Architecture

The final model architecture (example_code_training_ec2_gpu.py lines 104-115) consisted of a convolution neural network with the following layers and layer sizes.

*1 layer: Lambda, input size: 160*320*3
*2 layer: Cropping
*3 layer: Convolution2D, 24 classes with 5x5 kernel, activation = relu
*4 layer: Convolution2D, 36 classes with 5x5 kernel, activation = relu
*5 layer: Convolution2D, 48 classes with 5x5 kernel, activation = relu
*6 layer: Convolution2D, 64 classes with 3x3 kernel, activation = relu
*7 layer: Convolution2D, 64 classes with 3x3 kernel, activation = relu
*8 layer: Flatten
*9 layer: Full-connected layer, 100
*10 layer: Full-connected layer, 50
*11 layer: Full-connected layer, 10
*12 layer: Full-connected layer, 1

#### 3. Creation of the Training Set & Training Process

The given sample data set was used in my model training.
