# Traffic Sign Recognition

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:

- Load the data set (see below for links to the project data set)
- Explore, summarize and visualize the data set
- Design, train and test a model architecture
- Use the model to make predictions on new images
- Analyze the softmax probabilities of the new images
- Summarize the results with a written report

  [//]: #"Image References"
  [image1]: ./writeup_pics/visualization.jpg "Visualization"
  [image2]: ./writeup_pics/signs.jpg  "Normal Signs"
  [image3]: ./writeup_pics/graysigns.jpg "GrayScale Signs"
  [image4]: ./writeup_pics/generate.jpg "Generate additional images"
  [image5]: ./writeup_pics/finaltest.jpg "5 test pictures"
  [image6]: ./writeup_pics/featuremap.jpg "Feature Map"

## Rubric Points

### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data distributes over each sign class. It shows that the amount of examples varies vastly over classes. The biggest class is 'Speed limit (50km/h)' which has 2010 examples. The smallest class is 'Speed limit (20km/h)' which has 180 examples.                              

![all text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because it makes tensorflow to train better models.

Here is an example of 5 normal traffic signs images and 5 signs images after grayscaling.

![alt text][image2]
![alt text][image3]

As a last step, I normalized the image data because it makes the image's pixels distribute around origin and its shape on axies is more like a circle rather a ellipse which will make gradient descent more efficient.

I decided to generate additional data because the train set's examples varies vastly over different classes. If a class has many more examples than other classes, the model will be incliend to predict this class to yeild higher accuracy which is not proper for a good model. Generating additional data will make the difference on amount of examples between the biggest dataset and the smallest smaller.

To add more data to the the data set, I used the following techniques: shear transformation, translation and rotation, because it simulates the way we see traffic signs on road. We actually see signs from different position and this will make the sign transform from the one show directly in front. 

Here is an example of  an augmented image (rotation, shear, translation):

![all text][image4]


The difference between the original data set and the augmented data set is the following ... Examples' amount after generating fake images: 65617
Fake examples generated: 26408

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

|      Layer      |               Description                |
| :-------------: | :--------------------------------------: |
|      Input      |            32x32x1 RGB image             |
| Convolution 5x5 | 1x1 stride, valid padding, outputs 28x28x6 |
|      RELU       |                                          |
|   Max pooling   |       2x2 stride,  outputs 14x14x6       |
| Convolution 5x5 | 1x1 stride, valid padding, outputs 10x10x16 |
|      RELU       |                                          |
|   Max pooling   |       2x2 stride,  outputs 5x5x16        |
|    Flattern     |               400 outputs                |
| Fully connected |               120 outputs                |
|      RELU       |                                          |
| Fully connected |                84 outputs                |
|      RELU       |                                          |
| Fully connected |                10 outputs                |
|     Softmax     |                                          |

####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an 128 batch size, 50 epochs and 0.001 learning rate. I use the adam optimizer in TensorFlow. The introduction is here:

> an algorithm for first-order gradient-based optimization of stochastic objective functions, based on adaptive estimates of lower-order moments. The method is straightforward to implement, is computationally efficient, has little memory requirements.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.999
* validation set accuracy of 0.994
* test set accuracy of 0.922

If a well known architecture was chosen:
* What architecture was chosen? 
  I choose the LeNet architecture which has been using for decades, and LeNet-5 is the convolutional network designed for handwritten and machine-printed character recognition which can be used for traffic sign classification.
* Why did you believe it would be relevant to the traffic sign application?
  I think the CNN have the ability to recognize patterns in a picture. LeNet is proved to classify efficiently handwritten and printed characters, so I think it will also work out fine on traffic sign classification
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
  The accuracy on train and validation set is 0.999 and 0.994 which is very good, but it's 0.922 on test set which shows the model have an overfitting issue. I tried the dropout technique but it doesn't work out fine for now and I will try to adjust the architecture later using dropout and L2 regularization.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image5]

The third image might be difficult to classify because the exclamation mark seems connected so it's like a straight line. All the other 4 images' qualities are quite good to classify.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

|         Image         |      Prediction       |
| :-------------------: | :-------------------: |
| Speed limit (70 km/h) | Speed limit (70 km/h) |
| Speed limit (60 km/h) | Speed limit (60 km/h) |
|    General caution    |      Pedestrians      |
|    Turn left ahead    |    Turn left ahead    |
|      No passing       |      No passing       |


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 0.922

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a Speed limit (70 km/h) sign (probability of 100%), and the image does contain a Speed limit (70 km/h) sign. The top five soft max probabilities were

| Probability |      Prediction       |
| :---------: | :-------------------: |
|    100%     | Speed limit (70 km/h) |
|      0      | Speed limit (30 km/h) |
|      0      | Speed limit (20 km/h) |
|      0      | Speed limit (50 km/h) |
|      0      | Speed limit (60 km/h) |


For the second image, the model is relatively sure that this is a Speed limit (60 km/h) sign (probability of 100%), and the image does contain a Speed limit (60 km/h) sign. The top five soft max probabilities were

| Probability |                Prediction                |
| :---------: | :--------------------------------------: |
|    100%     |          Speed limit (60 km/h)           |
|      0      |          Speed limit (80 km/h)           |
|      0      |          Speed limit (50 km/h)           |
|      0      | End of no passing by vehicles over 3.5 metric ... |
|      0      |            End of no passing             |

For the third image, the model is relatively sure that this is a Pedestrians sign (probability of 99.4%), and the image does contain a General caution sign. The top five soft max probabilities were

| Probability |              Prediction               |
| :---------: | :-----------------------------------: |
|    99.4%    |              Pedestrians              |
|    0.6%     |            General caution            |
|      0      | Right-of-way at the next intersection |
|      0      |            Traffic signals            |
|      0      |         Go straight or right          |

For the fourth image, the model is relatively sure that this is a Turn left ahead sign (probability of 100%), and the image does contain a Turn left ahead sign. The top five soft max probabilities were

| Probability |      Prediction       |
| :---------: | :-------------------: |
|    100%     |    Turn left ahead    |
|      0      |      Keep right       |
|      0      | Go straight or right  |
|      0      |      Ahead only       |
|      0      | Speed limit (60 km/h) |

For the fifth image, the model is relatively sure that this is a No passing sign (probability of 100%), and the image does contain a No passing sign. The top five soft max probabilities were

| Probability |                Prediction                |
| :---------: | :--------------------------------------: |
|    100%     |                No passing                |
|      0      |      Vehicles over 3.5 metric tons       |
|      0      |          Speed limit (60 km/h)           |
|      0      | Vehicles over 3.5 metric tons prohibited |
|      0      |                 No entry                 |

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

The visual output of the first convolution layer is more understandable. It shows that the network use the edge detected to make classifications.
The visual output of the second convolution layer is relatively vague. But it probably shows that the network is finding different part of feature on the image.

![all text][image6]

