from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin
                dW[:,j] += X[i]
                dW[:,y[i]] += -X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #update gradient for the extra term and the scaling
    dW /= num_train
    dW += 2*reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N_train = X.shape[0] #500

    #get every score
    scores = X@W

    #get correct class' scores
    correct_scores = scores[np.arange(N_train),y]

    #calculate all scores' diff from correct score (including the correct score w/ itself)
    score_diffs = scores - np.expand_dims(correct_scores, -1) + 1 #add delta=1

    #take max(0,margin)
    score_diffs_capped = np.maximum(np.zeros(score_diffs.shape),score_diffs)

    #sum up score diffs by row, subtract 1 for the mean(0,1) that resulted in each row from calculating a diff for the correct score vs itself
    Loss_i = np.sum(score_diffs_capped, axis=1) - 1 

    #normalize and add regularization term
    loss = np.sum(Loss_i, axis=0) / N_train + reg*np.sum(W*W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N_classes = W.shape[1]

    #start with the capped margins from earlier to use as multiples of X 
    dw_col = score_diffs_capped;

    #theres a 1 for ever term where the margin was high enough that we added Loss (including 1 unit of loss for correctxcorrect locations)
    dw_col[score_diffs_capped>0] = 1 #representative of the result of the "if margin>0" above
    times_contributed = np.sum(dw_col, axis=1)

    #at correct class locations the contribution to loss is actually -X so put a -1*(for each time one of the non-correct classes subtracted it, so times_contributed-1)
    dw_col[np.arange(N_train),y] = -(times_contributed-1)

    #add X to the gradient wherever the multiple was positive, add -X an appropriate # of times in the correct class places
    dW = X.T@dw_col

    #update gradient for the extra term and the scaling
    dW /= N_train
    dW += 2*reg*W
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
