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


def batch_norm_reverse(data, data_shape, batch_size, norm_data):
    """
    input:
    @data: original ground truth data to normalize (N, H, W, C)
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
    unnormalized_batches = []

    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, data_shape[0])
        
        # Get a batch of data
        batch_data = data[start_idx:end_idx]
        batch_data_norm = norm_data[start_idx:end_idx]
        
        # Normalize batch along variable dimension
        # mean = np.mean(batch_data, axis=(0, 1, 2), keepdims=True)
        # std = np.std(batch_data, axis=(0, 1, 2), keepdims=True)
        # normalized_batch = (batch_data - mean) / std
        
        min_per_channel = np.min(batch_data, axis=(0, 1, 2), keepdims=True)
        max_per_channel = np.max(batch_data, axis=(0, 1, 2), keepdims=True)
        
        unnormalized_batch = batch_data_norm * (max_per_channel - min_per_channel) + min_per_channel
        
        unnormalized_batches.append(unnormalized_batch)
    
    # Concatenate normalized batches back together
    unnormalized_data = np.concatenate(unnormalized_batches, axis=0)

    return unnormalized_data