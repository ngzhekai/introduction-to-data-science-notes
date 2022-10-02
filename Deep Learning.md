# Introduction to Deep Learning

## The Loss Function
 
It measures the disparity (great difference) between the target's true value and the value the model predicts. In short, it measures how good the network's predictions are.

* Regression problems such as predicting the calories in *80 Cereals*, rating in *Red Wine Quality*.
* **Mean absolute error (MAE)**, a common loss function used for regression problems.

![](https://i.imgur.com/VDcvkZN.png)
> The *mean absolute error* is the average length between the fitted curve and the data points.

## Stochastic Gradient Descent (SGD)

An optimization algorithm often used in machine learning applications to find the model parameters that correspond to the best fit between predicted and the actual outputs.

* The **gradient** is a vector (shows the direction the weights need to go). In a more detailed form, it tells us how we could change the weights to make the loss change fastest.
* The **descent** term here, refers to the uses of the gradient to descend the loss curve towards a minimum.
* The **stochastic** means *"determined by chance."* Training is stochastic because the minibatches (batchd) are random samples from the dataset.

## Special Layers to Prevent Overfitting and Stabilize Training

### Dropout Layer

Overfitting is caused by the network learning spurious (false and not what it appeas to be) patterns in the training data. To recognize these spurious patterns, a network will often rely on a very specific combinations of weight, a kind of *"conspiracy"* of weights. In more detailed, they tend to be fragile: as remove one of them the conspiracy falls apart.

The idea behind **dropout** is to break up these conspiracies. So, we randomly *drop out* some fraction of a layer's input units every step of training, making it much harder for the network to learn those spurious (false and not what it appeas to be) patterns in the training data.

In return, it has to search for broad, general patterns, whose weight patterns tend to be more robust (strong and unlikely to break or fail).

![](https://i.imgur.com/a86utxY.gif)
> 50% *dropout* has been added between the two hidden layers.

#### Adding Dropout

In *Keras*, the dropout rate argument `rate` defines what percentage of the input units to shut off. Put the `Dropout` layer just before the layer you want the dropout applied to:

``` python
keras.Sequential([
    # ...
    layers.Dropout(rate=0.3), # apply 30% dropout to the next layer
    layers.Dense(16),
    # ...
])
```

### Batch Normalization

*Batch normalization* can help to correct training that is slow or unstable. In neural network, it's generally a good idea to put all of your data on a common scale (e.g. scikit-learns's *StandardScaler* or *MinMaxScaler*). 

The reason behind this is due to SGD will shift the network weights in proportion to how large an activation the data produces. Where *features* that tend to produce activations of very different sizes can make for unstable training behavior.

Therefore, it is good to normalize the data before it goes into the network, maybe also normalizing inside the network would be better! 

Here comes a special kind of layer that can do it, the **batch normalization layer**. This layer looks at each batch as it comes in, first normalizing the batch with its own *mean* and *standard deviation*, and then putting the data on a new scale with two trainable rescaling parameters.

Most often, batch normalization layer acts as an aid to the optimization process (it can sometimes also help prediction performance). Models with *batchnorm* (batch normalization layer) tend to need fewer epochs (the number of complete passes throughthe training dataset) to complete training. Furthermore, it can also fix various problems that an cause the training to get "stuck". *If you are having trouble during training, consider adding batch normalization to your models.* 

#### Adding Batch Normalization

1. You can put if after a layer,

``` python
layers.Dense(16, activation='relu'),
layers.BatchNormalization(),
```

2. Or between a layer and its activation function

``` python
layers.Dense(16),
layers.BatchNormalization(),
layers.Activation('relu'),
```

## Binary Classification

Classification into one of two classes is a common machine learning problem. You might want to predict whether or not a customer is likely to make a purchase, whether or not a credit card transaction was fraudulent. These are all **binary classification** problems.

In your raw data, the classes might be represented by strings like `"Yes"` and `"No"`, or `"Dog"` and `"Cat"`. Before using this data, let's assign a class label with one class having `0` and the other will be `1`. Assigning numeric labels puts the data in a form a neural network can use.

### Accuracy and Cross-Entrophy

**Accuracy** is one of themany metrics in use for measuring success on a classification problem. Accuracy is the ratio of correct predictions to total predictions: `accuracy = number_correct / total`.  

The problem with accuracy (and most other classfication metrics) is that it cannot be used as a loss function. SGD requires a loss function that changes smoothly, yet accuracy, being a ration of counts, changes in "jumps". So, we have to choose a substitute to act as the loss function. This substitute is the *cross-entropy* function.

### Sigmoid Function

The cross-entrophy and accuracy functions both require probabilities as inputs, having numbers from 0 to 1. To convert the real-valued outputs produced by a dense layer into probabilities, we will have to attach a new kind of activation function, the **sigmoid activation**.

![](https://i.imgur.com/FYbRvJo.png)
> *The sigmoid function maps real numbers into the interval [0,1].*

To get the final class prediction, will have to define a *threshold* probability. Which will be 0.5, therefore that rounding will give us the correct class: where below 0.5 means the class will label it with `0` and 0.5 or above means the class with label `1`. A 0.5 threshold is what Keras uses by default with its accuracy metric.