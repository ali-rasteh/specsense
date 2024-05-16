import numpy as np

def upsample_by_two_with_zeros(signal):
    """
    Upsample a signal by a factor of 2 by inserting zeros between the original samples.

    Args:
        signal (np.array): Input signal to be upsampled.

    Returns:
        np.array: Upsampled signal with zeros inserted.
    """
    upsampled_length = 2 * len(signal)
    upsampled_signal = np.zeros(upsampled_length)

    # Assign the original signal values to the even indices
    upsampled_signal[::2] = signal

    return upsampled_signal

# Example usage
signal = np.array([1, 2, 3, 4, 5])
upsampled_signal = upsample_by_two_with_zeros(signal)
print("Original signal:", signal)
print("Upsampled signal:", upsampled_signal)
