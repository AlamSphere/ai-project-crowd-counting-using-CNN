
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Define your CNN model architecture
def crowd_counting_model(input_shape):
    # Indent the code block within the function
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1) # Output layer for crowd count prediction
    ])
    return model

# Generate some random data for demonstration purposes (replace this with your actual data)
train_images = np.random.rand(100, 128, 128, 3)
train_labels = np.random.randint(0, 100, size=(100, 1))
val_images = np.random.rand(20, 128, 128, 3)
val_labels = np.random.randint(0, 100, size=(20, 1))

# Define the input shape of your images
input_shape = (128, 128, 3) # Adjust these values according to your data

# Compile the model
model = crowd_counting_model(input_shape)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
model.fit(train_images,train_labels,epochs=10,batch_size=32,validation_data=(val_images,val_labels))

# Make predictions on some test data
test_images = np.random.rand(10, 128, 128, 3)
predictions = model.predict(test_images)
print(predictions)
