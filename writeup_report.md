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
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I tried using LeNet at first, but after training even upto 20 epochs the car wasn't sticking into the lane even 20% of the time (sometimes i took control in autonomous mode and steered the car into lane manually to see if there are only specific places it fails but 80% of the time car wasn't staying in the lane, going into water or dirt road.)

Therefore, I decided to use Nvidia Autonomous Car Transfer learning model from there project paper (https://arxiv.org/pdf/1604.07316v1.pdf) . On the dataset generate (mentioned below) I trained for 20 epochs and the model did very well to stay in lane (although it was making some mistakes which i made while recording the laps and that told me that model is possibly overfitting). The model architecture is discussed in model architecture and training stargey section.

The model consists of a convolution neural network and includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer and cropped using Keras cropping layer. 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. Also the model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. Additionally number of epochs were reduced to 5 to prevent overfitting.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data generated using driving into simulator was chosen to keep the vehicle driving on the road. I used a combination of counter clockwise and clockwise center lane driving. Further details in next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

Even before using the Nvidia's model, some data preprocessing was required. Therefore I added Lambda layer to normalized the data to zero mean by dividing each pixel by 255 and subtracing 0.5. Further, I added Cropping layer to use only the area of interest for training. Although while writing this I realized, I should have cropped first and then would apply used Lambda normalization, might have saved few cpu cycles. Further I started using the Nvidia model architecture and in the end an Output layer was added to have single output as steering angle.

This model did well as i was able to complete the track, though slightly going over the yellow line at couple of places and i realized that while recording data, i made those mistakes as well, that hinted me model is overfitting, therefore I added a dropout layer with probablity 0.5 after Flattening. After that the car was able to drive through track one successufly multiple times without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers:

```
Layer            Output Shape                 Action           
-----------------------------------------------------------------

Lambda           (None, 160, 320, 3)          Normalization
-----------------------------------------------------------------
Cropping2D       (None, 90, 320, 3)           Cropping
-----------------------------------------------------------------
Convolution2D    (None, 43, 158, 24)          Relu Activation
-----------------------------------------------------------------
Convolution2D    (None, 20, 77, 36)           Relu Activation
-----------------------------------------------------------------
Convolution2D    (None, 8, 37, 48)            Relu Activation
-----------------------------------------------------------------
Convolution2D    (None, 6, 35, 64)            Relu Activation
-----------------------------------------------------------------
Convolution2D    (None, 4, 33, 64)            Relu Activation  
-----------------------------------------------------------------
Flatten          (None, 8448)                 Flatten
-----------------------------------------------------------------
Dropout          (None, 8448)                 Dropout with 0.5 
-----------------------------------------------------------------
Dense            (None, 100)                  Dense 100
-----------------------------------------------------------------
Dense            (None, 50)                   Dense 50
-----------------------------------------------------------------
Dense            (None, 10)                   Dense 10
-----------------------------------------------------------------
Dense            (None, 1)                    Dense 1

-----------------------------------------------------------------
```


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving clockwise and two laps on track one using center lane driving counter clockwise to prevent right bias. (Although i tried flipping the image instead of driving counter clockwise but that didn't help.)

I tried to use images from left and right cameras as well and data from track 2 but didn't see any improvements may be because the number of epochs were just 5. 

I finally randomly shuffled the data set using sklean shuffle and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I used an adam optimizer so that manually training the learning rate wasn't necessary.
