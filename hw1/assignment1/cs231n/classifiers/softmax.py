from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # compute the loss and the gradient (from SVM)
    num_classes = W.shape[1]
    num_train = X.shape[0]

    for i in range(num_train):
      scores = X[i].dot(W)
      #use logC
      scores -= np.max(scores)
      correct_class_score = scores[y[i]]

      #begin counting up L_i and denominator summation
      l_sub_i = -correct_class_score
      sum_total = 0

      #calculate summation
      for j in range(num_classes):
          sum_total += np.exp(scores[j])
      
      #add summation to L_i
      l_sub_i += np.log(sum_total)

      #add L_i to overall loss
      loss += l_sub_i

      #add terms to dW for weights used in summation and for correct class weights used
      #we need to loop again because we needed to know the summation total for this
      #print(W.shape, X.shape)
      dW[:,y[i]] += -X[i] 
      for j in range(num_classes):
        dW[:,j] += (1/sum_total) * X[i] * np.exp(scores[j])
      
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]
    
    #get all scores via matrix multiplication
    scores = X.dot(W)
    #normalization trick to avoid instability
    scores -= np.expand_dims(np.max(scores, axis=1),-1)

    #start per example loss as -correct class score
    correct_class_scores = scores[np.arange(num_train),y]
    L_sub_is = -correct_class_scores

    #perform the summation using np.sum
    summation_totals = np.sum(np.exp(scores), axis=1)

    #add contributions of the summations to the gradient
    dW_per_example = (np.exp(scores)*np.expand_dims(1/summation_totals,-1))
    dW_per_example[np.arange(num_train),y] -= 1 #to give a -X contribution for every correct class
    dW += X.T.dot(dW_per_example)
  
    #add summation totals to the per example losses
    L_sub_is += np.log(summation_totals)

    #add up losses per example
    loss = np.sum(L_sub_is)
    
    #take avg and add regularization loss
    loss /= num_train
    loss += reg * np.sum(W * W)

    dW /= num_train
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
