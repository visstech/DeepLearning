import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Scale data
X_train_scaled = X_train / 255.0
X_test_scaled = X_test / 255.0

# One-hot encoding for categorical labels
y_train_categorical = keras.utils.to_categorical(y_train, num_classes=10)
y_test_categorical = keras.utils.to_categorical(y_test, num_classes=10)

# Class names
classes = ["airplane", "automobile", "bird", "cat", "deer", 
           "dog", "frog", "horse", "ship", "truck"]

# Model definition
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(32, 32, 3)),  
    keras.layers.Dense(3000, activation='relu'),  
    keras.layers.Dense(1000, activation='relu'),
    keras.layers.Dense(10, activation='softmax')  # Changed from sigmoid to softmax
])

# Compile the model
model.compile(optimizer='SGD',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train_scaled, y_train_categorical, epochs=50)

# Make predictions
predictions = model.predict(X_test_scaled)
predicted_class = np.argmax(predictions[0])

# Print results
print('Predicted class:', classes[predicted_class])
print('Actual class:', classes[y_test[0][0]])


'''
✅ Train the model before making predictions.
✅ Use softmax instead of sigmoid in the output layer.
✅ Use categorical_crossentropy only for one-hot encoded labels.
✅ Ensure correct prediction extraction with np.argmax().

You're correctly using categorical_crossentropy, but make sure:

Labels are one-hot encoded (which they are in y_train_categorical).
If using sparse labels (integer class labels like y_train before encoding), you should use sparse_categorical_crossentropy instead

Problem:
You're using sigmoid in the output layer:
keras.layers.Dense(10, activation='sigmoid')
sigmoid is generally used for binary classification or independent multi-label classification.
In multi-class classification (like CIFAR-10), you should use softmax instead.
Solution:
Change the activation function in the output layer to softmax:

python
Copy
Edit
keras.layers.Dense(10, activation='softmax')
'''