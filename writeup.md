# **Behavioral Cloning** 
---

**Behavioral Cloning Project**

[//]: # (Image References)

[image1]: ./Writup_Images/Unprocessed.JPG "Unprocessed Images"
[image2]: ./Writup_Images/Flipped.JPG "Flipped Images"
[image3]: ./Writup_Images/Cropped.JPG "Cropped Images"
[image4]: ./Writup_Images/Normalized.JPG "Normalized Images"

---
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

My model consists of a convolution neural network with 5x5 filter sizes and depths between 24 and 48. 

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. 
   

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.
I have splitted the data into two parts 80% data was used for training and 20% for validation.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used images from all the three cameras(center, left and right) to train and validate the model. I used the udacity dataset for training and validation.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the LeNet I thought this model might be appropriate.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set.

Initially I used LeNet but the car was not able to complete the track and failed to recovr at sharp turns. In the next attempt i used NVIDIA architecture the car was able to complete the track. But as the architecture is complex, model took more time to train. In an attempt to reduce the complexity of the model I removed two convolutional layers and two fully connected layers. Now with only three convolutional layers and three fully connected layers 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, to improve the driving behavior in these cases, data preprocessing and augmentation is done.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes ...

- Convolution Layer : 24 5x5 kernels
- Convolution Layer : 36 5x5 kernels
- Convolution Layer : 48 5x5 kernels
- Fully Connected Layer : 50 output size
- Fully Connected Layer : 10 output size.
- Fully Connected Layer : 1 output size.

#### 3. Creation of the Training Set & Training Process

I have used the dataset provided by udacity to train my model.
I used the images from left and right cameras so that the car would learn to recover from the left side and right sides of the road back to center. Here is an example images from center, left and right camras.

![Original Images][image1]

To augment the data set, I also flipped images and angles from all cameras thinking that this would generate more data and all the turn scenarios are covered For example, here is an image that has then been flipped:

![Flipped Images][image2]

After the collection process, I had six times the number of data points provided by udacity dataset. I then preprocessed this data by cropping, resizing and normalizing all the images.

![Cropped Images][image3]
![Normalized Images][image4]

I finally randomly shuffled the data set and put 20% of the data into a validation set. 
I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 2 as evidenced by very small value of validation loss(0.016). I used an adam optimizer so that manually training the learning rate wasn't necessary.
