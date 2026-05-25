import cv2
import numpy as np

# Load the test image
#insert your image location in place of ("/content....')
test_image = cv2.imread('/content/IMG_43.jpg')

# Check if the image was successfully loaded
if test_image is None:
    print(f'Error: Could not open or find the image /content/IMG_43.jpg')
else:
    # Resize the image to match the expected input shape of the model
    test_image = cv2.resize(test_image, (128, 128))

    # Normalize pixel values to be in the range [0, 1]
    test_image = test_image.astype('float32') / 255.0

    # Make sure the image has the correct shape and dimensions
    test_image = np.expand_dims(test_image, axis=0)

    # Assuming 'model' is already defined and loaded
    # Make predictions on the test image
    predictions = model.predict(test_image)

    # Print the predictions
    print(predictions)
