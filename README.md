# mlearn
Machine Learning for Python

Train a function on training data (x_i,y_i) by minimising the loss:
   sum_i loss(fx_i,y_i) + lambda * Regularizer          (1)

The two main objects in the module are:
1. mlmodel: the model to be fitted
2. decomposableloss: the loss that should be minimised

The mlmodel defines the function that maps the input to the output, and
defines the derivative of that function with respect to its parameters.
Next to this basic functionality, also a name and an initialisation
can be defined.
The loss can be a combination a data loss and a regulariser.

For a simple example, have a look at testml.py

There are situations where formulation (1) does not apply. For instance
when the Area Under the ROC curve is optimised. Then the (empirical)
loss cannot be decomposed into a sum over losses per training object. In
that case you have to define a nondecomposable loss.
