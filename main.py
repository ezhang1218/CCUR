#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import svd
from scipy.linalg import pinv


# In[3]:


def deterministic_cur(A, k, top_selected):
    U, S, Vt = svd(A, full_matrices=False)
    column_scores = compute_leverage_scores(Vt, k)
    row_scores = compute_leverage_scores(U.T, k)
    
    top_col_indices = np.argsort(-column_scores)[:top_selected]
    top_row_indices = np.argsort(-row_scores)[:top_selected]
    low_col_indices = np.argsort(column_scores)[:top_selected]

    C = A[:, top_col_indices]
    R = A[top_row_indices, :]
    C_pinv = pinv(C)
    R_pinv = pinv(R)
    U = C_pinv @ A @ R_pinv
    
    return C, column_scores, R, top_col_indices, top_row_indices, low_col_indices


def compute_leverage_scores(V, k):
    """Compute the normalized statistical leverage scores."""
    leverage_scores = np.sum(V[:k, :]**2, axis=0)
    return leverage_scores

def scores(A, k):
    U, S, Vt = svd(A, full_matrices=False)
    column_scores = compute_leverage_scores(Vt, k)
    row_scores = compute_leverage_scores(U.T, k)
    return column_scores, row_scores

def ccur(X, Y, k, num_cols, epsilon=1e-6):
    leverage_scores_X, _ = scores(X, k)
    leverage_scores_Y, _ = scores(Y, k)
    ratios = leverage_scores_X / (epsilon + leverage_scores_Y)
    top_c_indices = np.argsort(-ratios)[:num_cols]
    return top_c_indices, ratios[top_c_indices], ratios



# In[ ]:




