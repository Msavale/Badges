import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Example data
# Assuming we have images of size 28x28 with 1 channel (grayscale)
# If you have colored images, you would typically have 3 channels (RGB)
input_shape = (28, 28, 1)  # Shape of each image

# Build the model
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),  # Convolutional layer with 32 filters
    MaxPooling2D(pool_size=(2, 2)),  # Max pooling layer
    Conv2D(64, kernel_size=(3, 3), activation='relu'),  # Convolutional layer with 64 filters
    MaxPooling2D(pool_size=(2, 2)),  # Max pooling layer
    Flatten(),  # Flatten layer to convert 2D feature maps to 1D
    Dense(128, activation='relu'),  # Fully connected layer with 128 neurons
    Dense(10, activation='softmax')  # Output layer with 10 neurons for classification
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Display the model summary
model.summary()
