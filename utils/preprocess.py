import numpy as np


def batch_norm(data, data_shape, batch_size):
    """
    input:
    @data: original ground truth data to normalize (N, H, W, C)
    @data_shape: prediction shape
    @batch_size: batch size to normalize or unnormalized

    output:
    @normalized data
    """
    # Calculate the number of batches
    num_batches = data_shape[0] // batch_size
    if data_shape[0] % batch_size != 0:
        num_batches += 1

    # Normalize data in batches
    normalized_batches = []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, data_shape[0])
        
        # Get a batch of data
        batch_data = data[start_idx:end_idx]
        
        # Normalize batch along variable dimension
        # mean = np.mean(batch_data, axis=(0, 1, 2), keepdims=True)
        # std = np.std(batch_data, axis=(0, 1, 2), keepdims=True)
        # normalized_batch = (batch_data - mean) / std
        
        min_per_channel = np.min(batch_data, axis=(0, 1, 2), keepdims=True)
        max_per_channel = np.max(batch_data, axis=(0, 1, 2), keepdims=True)
        normalized_batch = (batch_data - min_per_channel) / (max_per_channel - min_per_channel)
        
        normalized_batches.append(normalized_batch)
    
    # Concatenate normalized batches back together
    normalized_data = np.concatenate(normalized_batches, axis=0)

    return normalized_data



def batch_norm_inverse(unnorm_data, data, data_shape, batch_size):
    """
    input:
    @unnorm_data: original ground truth data without norm (N, H, W, C)
    @data: prediction with norm (N, H, W, C)
    @data_shape: prediction shape
    @batch_size: batch size to normalize or unnormalized

    output:
    @unnormalized data
    """
    # Calculate the number of batches
    num_batches = data_shape[0] // batch_size
    if data_shape[0] % batch_size != 0:
        num_batches += 1

    # Normalize data in batches
    normalized_batches_inv = []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, data_shape[0])
        
        # Get a batch of data
        batch_unnorm_data = unnorm_data[start_idx:end_idx]
        batch_data = data[start_idx:end_idx]
        
        # Normalize batch along variable dimension
        # mean = np.mean(batch_unnorm_data, axis=(0, 1, 2), keepdims=True)
        # std = np.std(batch_unnorm_data, axis=(0, 1, 2), keepdims=True)
        # normalized_batch_inv = batch_data * std + mean
        
        min_per_channel = np.min(batch_unnorm_data, axis=(0, 1, 2), keepdims=True)
        max_per_channel = np.max(batch_unnorm_data, axis=(0, 1, 2), keepdims=True)
        
        normalized_batch_inv = batch_data * (max_per_channel - min_per_channel) + min_per_channel
        
        normalized_batches_inv.append(normalized_batch_inv)
    
    # Concatenate normalized batches back together
    normalized_data_inv = np.concatenate(normalized_batches_inv, axis=0)

    return normalized_data_inv