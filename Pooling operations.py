import tensorflow as tf
import numpy as np

# Create a random 4x4 matrix as input
input_matrix = np.random.randint(0, 10, (1, 4, 4, 1)).astype(np.float32)  # Ensure it's float32

# Define max pooling layer
max_pooling = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')
max_pooled_output = max_pooling(input_matrix).numpy()

# Define average pooling layer
avg_pooling = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')
avg_pooled_output = avg_pooling(input_matrix).numpy()

# Print results
print("Original Matrix:\n", input_matrix[0, :, :, 0])
print("\nMax Pooled Matrix:\n", max_pooled_output[0, :, :, 0])
print("\nAverage Pooled Matrix:\n", avg_pooled_output[0, :, :, 0])
