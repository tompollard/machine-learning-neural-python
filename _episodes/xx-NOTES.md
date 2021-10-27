---
title: "NOTES"
teaching: 20
exercises: 10
questions:
- "How do we predict outcomes with artificial neural networks?"
objectives:
- "Predict patient outcomes with artificial neural networks."
keypoints:
- "Neural networks are a flexible, powerful family of models."
---

{% include links.md %}

## Neural networks



THIS IS AWESOME: https://milliams.com/courses/

ALSO https://sebastianraschka.com/Articles/2015_singlelayer_neurons.html

- Find weights
- Define a loss function
- Gradient descent (non-convex)
- Back propagation

https://colab.research.google.com/github/google/eng-edu/blob/main/ml/cc/exercises/intro_to_neural_nets.ipynb


IMAGES: https://carpentries-incubator.github.io/machine-learning-librarians-archivists/03-what-is-ml-good-at/index.html

Neural networks are...

https://www.tensorflow.org/tutorials/images/cnn

https://www.tensorflow.org/tutorials/images/classification


## Create our neural network

Let's define a very simple neural network...

THIS IS GREAT

https://cnl.salk.edu/~schraudo/teach/NNcourse/intro.html

https://peterroelants.github.io/posts/neural-network-implementation-part01/

~~~
define neural network
~~~
{: .language-python}

## Specify our loss function and optimizer

As with our linear regression model, we need to define our loss function and an approach for optimising our model.

~~~
define neural network
~~~
{: .language-python}

## Training

We can now train our model by fitting it to our sample data.

[Note: Mention GPU support. Training can be a computationally expensive process. Often use GPUs.]

## Evaluation

We've successfully created a neural network for predicting outcomes! Let's predict outcomes and test.


