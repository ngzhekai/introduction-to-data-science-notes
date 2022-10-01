# Introduction to Deep Learning

## The Loss Function
It measures the disparity (great difference) between the target's true value and the value the model predicts. In short, it measures how good the network's predictions are.
- Regression problems such as predicting the calories in *80 Cereals*, rating in *Red Wine Quality*.
- **Mean absolute error (MAE)**, a common loss function used for regression problems.
- ![](https://i.imgur.com/VDcvkZN.png)
>The *mean absolute error* is the average length between the fitted curve and the data points.


## Stochastic Gradient Descent (SGD)
An optimization algorithm often used in machine learning applications to find th emodel parameters that correspond to the best fit between predicted and the actual outputs.
- The **gradient** is a vector (shows the direction the weights need to go). In a more detailed form, it tells us how we could change the weights to make the loss change fastest.
- The **descent** term here, refers to the uses of the gradient to descend the loss curve towards a minimum.
- The **stochastic** means *"determined by chance."* Training is stochastic because the minibatches (batchd) are random samples from the dataset.