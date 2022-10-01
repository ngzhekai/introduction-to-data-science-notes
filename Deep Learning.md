# Introduction to Deep Learning

## The Loss Function
It measures the disparity (great difference) between the target's true value and the value the model predicts. In short, it measures how good the network's predictions are.
- Regression problems such as predicting the calories in *80 Cereals*, rating in *Red Wine Quality*.
- **Mean absolute error (MAE)**, a common loss function used for regression problems.

![](https://i.imgur.com/VDcvkZN.png)
> The *mean absolute error* is the average length between the fitted curve and the data points.


## Stochastic Gradient Descent (SGD)
An optimization algorithm often used in machine learning applications to find th emodel parameters that correspond to the best fit between predicted and the actual outputs.
- The **gradient** is a vector (shows the direction the weights need to go). In a more detailed form, it tells us how we could change the weights to make the loss change fastest.
- The **descent** term here, refers to the uses of the gradient to descend the loss curve towards a minimum.
- The **stochastic** means *"determined by chance."* Training is stochastic because the minibatches (batchd) are random samples from the dataset.

## Special Layers to Prevent Overfitting and Stabilize Training

### Dropout Layer
Overfitting is caused by the network learning spurious (false and not what it appeas to be) patterns in the training data. To recognize these spurious patterns, a network will often rely on a very specific combinations of weight, a kind of *"conspiracy"* of weights. In more detailed, they tend to be fragile: as remove one of them the conspiracy falls apart.

The idea behind **dropout** is to break up these conspiracies. So, we randomly *drop out* some fraction of a layer's input units every step of training, making it much harder for the network to learn those spurious (false and not what it appeas to be) patterns in the training data.

In return, it has to search for broad, general patterns, whose weight patterns tend to be more robust (strong and unlikely to break or fail).

![](https://i.imgur.com/a86utxY.gif)
> 50% *dropout* has been added between the two hidden layers.

### Adding Dropout
In *Keras*, the dropout rate argument `rate` defines what percentage of the input units to shut off. Put the `Dropout` layer just before the layer you want the dropout applied to:

``` python
keras.Sequential([
    # ...
    layers.Dropout(rate=0.3), # apply 30% dropout to the next layer
    layers.Dense(16),
    # ...
])
```
