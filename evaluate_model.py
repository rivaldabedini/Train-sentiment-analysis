import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('sentiment_rnn_model.h5')

# Load the x and y variables from the saved files
x_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)

# Print the loss and accuracy
print(f'Loss: {loss}')
print(f'Accuracy: {accuracy}')


# Loss: 0.3496704697608948
# Accuracy: 0.9101775884628296