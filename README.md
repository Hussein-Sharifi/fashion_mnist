
# Fashion MNIST Data Classification Report

May 18th, 2023  
**Author:** Hussein Sharifi 
**Course:** STAT426 Final Project
 

---

## Introduction

This report involves using various classification methods in R for predicting clothing item types in the Fashion MNIST dataset. This dataset includes seventy thousand entries of low-resolution images, each of which would be difficult to classify with direct observation. It is split into a training set of sixty thousand entries, and a test set of ten thousand entries. The images are defined with 784 variables representing pixel information, and one variable representing article clothing type.

![Two images from the Fashion MNIST dataset](https://github.com/Hussein-Sharifi/fashion_mnist/blob/master/figures/example_items.jpg "Two images from the Fashion MNIST dataset")

The following methods will be used: K-nearest neighbors, Linear and Radial Support Vector Machines, Linear Discriminant Analysis, Quadratic Discriminant Analysis, Deep Neural Networks, and Multinomial Logistic Regression. Moreover, each of the above methods will also be trained using dimensionally reduced data via Principal Component Analysis, and the difference in error rate and computation speed will be examined. It should be noted that I was unsuccessful with coding the Radial SVM and Multinomial Logistic Regression approaches.

## Principal Component Analysis (PCA)

PCA uses Eigen Decomposition to capture as much signal as possible in a reduced-dimensional setting, and therefore has the potential to vastly improve computation. In this case, 90% of the signal was captured by 74 as opposed to the original dataset’s 784 dimensions. Then, each of the following methods will be evaluated with an embedding of 75 dimensions, counting the dimension specifying article clothing type.

## Tenfold Cross-Validation (CV)

A sample of 1200 datapoints were chosen at random from the training set, and 600 points were similarly chosen from the testing set. In order to derive a predictive model, a subset of these points must be used to fit the model, and another subset to test this fitted model. A split of 80%/20% may be used, but this does not fully utilize the available data for training. In this case, Tenfold Cross-Validation was used to overcome this problem. It divides the sample into 10 sets of 120 datapoints, uses 9 of them for training and 1 for testing, then repeats this process several times by switching the training set. With the exception of KNN models, 5 repetitions were used for training cross-validated models. 10 repetitions were used for KNN.

## K-Nearest Neighbors (KNN)

The K-Nearest Neighbors model classifies datapoints by comparing them to the classification of their closest K neighbors. Cross-validation was used here to find that the optimal number of neighbors for prediction in this case is 7. The processing time for this model using the PCA (74-dimension) data was 0.3s, and 1.9s for the full (784 dimension) data. Moreover, the expected accuracy of prediction here is 79.2% using PCA, and 73.8% without, meaning an error rate of 26.2% and 20.8% respectively. This is highly overestimating the true accuracy of this model. In fact, the error rate on the testing sample was found to be 78.7% using PCA, and 78.8% without.

## Linear Support Vector Machines (SVMs)

The Linear SVMs model finds the hyperplane that optimally separates classes within the data. This linear hyperplane is a line in the two-dimensional case. In practice, however, a clear separation is unlikely. In the “soft case,” the Linear SVMs model allows for some misclassification while optimizing to lower an objective function. There is an additional penalty for misclassified points. Processing time: 58s with PCA, 607s without. Error rate: 23% with PCA, 21% without.

## Radial Support Vector Machines (SVMs)

The Radial SVM model is similar to the linear version but allows for classification of data that is not linearly separable. It does this using a kernel function that maps features into a higher-dimensional space. I was not able to tune the Radial SVM model.

## Linear Discriminant Analysis (LDA)

In LDA, the model is assumed to be normal with the same covariance for each class. The model assigns each point to the maximum likelihood class. Processing time: 0.05s with PCA, 3.39s without. Error rate: 21.7% with PCA, 35.5% without.

## Quadratic Discriminant Analysis (QDA)

QDA is similar to LDA but allows each class to have its own variance. I was not able to run QDA without PCA. With PCA: Processing time: 0.02s, Error rate: 26.3%.

## Random Forests (RFs)

Random Forests average many iterations of decision trees optimized over bootstrapped samples. The model randomly chooses a subset of parameters used to make each split. Processing time: 4.4s with PCA, 23s without. Error rate: 20.5% with PCA, 19.7% without.

## Deep Neural Nets (DNNs)

Deep Neural Nets involve alternating layers of linear and nonlinear combinations of inputs. The architecture used here: 256 → 128 → 10 units, with ReLU functions. Processing time: 8.5s with PCA, 134s without. Error rate: 19.4% with PCA, 11.5% without.

![Loss Tracking](https://github.com/Hussein-Sharifi/fashion_mnist/blob/master/figures/loss_tracking.jpg "Loss Tracking")

## Multinomial Logistic Regression

Multinomial Logistic Regression is used to calculate the probability of a data point belonging to a class. I was not able to correctly code the model for this report.

## Conclusion

Of all the models, Deep Neural Nets had the greatest accuracy due to their ability to leverage large datasets to fit low-bias models. However, they lack interpretability, and optimizing their architecture requires significant trial and error.
