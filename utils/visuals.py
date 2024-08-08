from math import sqrt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def vis_one_var(Y_true, Y_pred, sample_idx, var_idx):
    """
    Y_pred: (N, H, W, C) 
    Y_true: (N, H, W, C) 
    sample_idx: int
    var_idx: int
    """
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # Create subplots with 1 row and 3 columns
    fig_idx1, fig_idx2, fig_idx3  = 0, 1, 2
    
    axs[fig_idx1].imshow(Y_true[sample_idx, :, :, var_idx])
    axs[fig_idx1].set_title('Y_true (2018/06:00h)')
    axs[fig_idx1].set_xlabel('Longitude')
    axs[fig_idx1].set_ylabel('Latitude')
    fig.colorbar(axs[fig_idx1].imshow(Y_true[sample_idx, :, :, var_idx]), ax=axs[fig_idx1], orientation='horizontal')  # Add colorbar
    
    axs[fig_idx2].imshow(Y_pred[sample_idx, :, :, var_idx])
    axs[fig_idx2].set_title('Y_pred (2018/06:00h)')
    axs[fig_idx2].set_xlabel('Longitude')
    axs[fig_idx2].set_ylabel('Latitude')
    fig.colorbar(axs[fig_idx2].imshow(Y_pred[sample_idx, :, :, var_idx]), ax=axs[fig_idx2], orientation='horizontal')  # Add colorbar
    
    
    axs[fig_idx3].imshow(Y_pred[sample_idx, :, :, var_idx] - Y_true[sample_idx, :, :, var_idx])
    axs[fig_idx3].set_title('Difference (Y_pred - Y_test) (2018/06:00h)')
    axs[fig_idx3].set_xlabel('Longitude')
    axs[fig_idx3].set_ylabel('Latitude')
    fig.colorbar(axs[fig_idx3].imshow(Y_pred[sample_idx, :, :, var_idx]- Y_true[sample_idx, :, :, var_idx]), ax=axs[fig_idx3], orientation='horizontal')  # Add colorbar
    
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()



def vis_one_var_recon(X_orig, X_recon, sample_idx, var_idx):
    """
    X_recon: (N, H, W, C) 
    X_orig: (N, H, W, C) 
    sample_idx: int
    var_idx: int
    """
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # Create subplots with 1 row and 3 columns
    fig_idx1, fig_idx2, fig_idx3  = 0, 1, 2
    
    axs[fig_idx1].imshow(X_orig[sample_idx, :, :, var_idx])
    axs[fig_idx1].set_title('X_orig')
    axs[fig_idx1].set_xlabel('Longitude')
    axs[fig_idx1].set_ylabel('Latitude')
    fig.colorbar(axs[fig_idx1].imshow(X_orig[sample_idx, :, :, var_idx]), ax=axs[fig_idx1], orientation='horizontal')  # Add colorbar
    
    
    axs[fig_idx2].imshow(X_recon[sample_idx, :, :, var_idx])
    axs[fig_idx2].set_title('X_recon')
    axs[fig_idx2].set_xlabel('Longitude')
    axs[fig_idx2].set_ylabel('Latitude')
    fig.colorbar(axs[fig_idx2].imshow(X_recon[sample_idx, :, :, var_idx]), ax=axs[fig_idx2], orientation='horizontal')  # Add colorbar
    
    
    axs[fig_idx3].imshow(X_recon[sample_idx, :, :, var_idx] - X_orig[sample_idx, :, :, var_idx])
    axs[fig_idx3].set_title('Difference (X_recon - X_train)')
    axs[fig_idx3].set_xlabel('Longitude')
    axs[fig_idx3].set_ylabel('Latitude')
    fig.colorbar(axs[fig_idx3].imshow(X_recon[sample_idx, :, :, var_idx] - X_orig[sample_idx, :, :, var_idx]), ax=axs[fig_idx3], orientation='horizontal')  # Add colorbar
    
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()