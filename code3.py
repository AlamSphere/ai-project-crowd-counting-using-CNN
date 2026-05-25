import matplotlib.pyplot as plt

# Extract the single image from the batch
single_image = test_image[0]  # Get the first (and only) image

# Visualize the test image
plt.imshow(single_image)
plt.title('Test Image')
plt.axis('off')
plt.show()

# Display the predicted crowd count
print(f'Predicted Crowd Count: {predictions[0]}')
