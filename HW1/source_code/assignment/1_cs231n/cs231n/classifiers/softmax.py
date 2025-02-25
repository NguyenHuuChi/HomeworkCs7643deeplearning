import numpy as np
from random import shuffle

def softmax(z):
  numerator=np.exp(z-np.max(z,axis=0))
  denom= np.sum(numerator,axis=0)
  softmax= np.divide(numerator,denom)
  return softmax
def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W, an array of same size as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  C,D =W.shape
  N = len(y)
  y_preds= np.matmul(W, X)
  soft= softmax(y_preds)

  regu= reg*np.sum(W*W)
  loss= -(1/N)*np.sum(np.log(soft[y,range(N)] + 1e-5))+regu

  soft[y,range(N)]-=1.0 # must infer from the formular
  dW= (1/N)*np.dot(soft,X.T)

  dW+= reg*W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
