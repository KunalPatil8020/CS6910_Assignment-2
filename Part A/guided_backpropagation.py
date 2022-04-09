# Import the required libraries
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.compat.v1.enable_eager_execution()
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import shutil
import os

model = load_model("model-best.h5")

# Choose a random image from the dataset
random_class = os.listdir("inaturalist_12K/train")[np.random.randint(0, 10)]
random_class_images = os.listdir("inaturalist_12K/train/"+random_class)
random_image = random_class_images[np.random.randint(0, len(random_class_images))]

# Load the random image that we will use for guided backpropagation
img = tf.keras.preprocessing.image.load_img("inaturalist_12K/train/"+random_class+"/"+random_image,
                                            target_size=(224, 224))

# Display the random image
plt.imshow(img)
plt.axis("off")
plt.title("Random Image from the dataset")
plt.show()

# This custom model has the 5th convolutional layer as its final layer
guided_backprop_model = tf.keras.models.Model(inputs = [model.inputs], outputs = [model.get_layer(index=-8).output])
# Here we choose only those layers that have an activation attribute
layer_dictionary = [layer for layer in guided_backprop_model.layers[1:] if hasattr(layer,'activation')]

# Define a custom gradient for the version of ReLU needed for guided backpropagation
def guidedbackpropRelu(x):
    def grad(dy):
        return tf.cast(dy>0,"float32") * tf.cast(x>0, "float32") * dy
    return tf.nn.relu(x), grad

for l in layer_dictionary:
    # Change the ReLU activation to supress the negative gradients
    if l.activation == tf.keras.activations.relu:
        l.activation = guidedbackpropRelu

# The shape of the layer that we are interested in
conv_output_shape = model.layers[-8].output.shape[1:]

plt.figure(figsize=(30, 60))
for i in range(10):
    # Index of a random pixel
    neuron_index_x = np.random.randint(0, conv_output_shape[0])
    neuron_index_y = np.random.randint(0, conv_output_shape[1])
    neuron_index_z = np.random.randint(0, conv_output_shape[2])

    # Mask to focus on the outputs of only one neuron in the last convolution layer
    masking_matrix = np.zeros((1, *conv_output_shape), dtype="float")
    masking_matrix[0, neuron_index_x, neuron_index_y, neuron_index_z] = 1

    # Calculate the gradients
    with tf.GradientTape() as tape:
        inputs = tf.cast(np.array([np.array(img)]), tf.float32)
        tape.watch(inputs)
        outputs = guided_backprop_model(inputs) * masking_matrix

    grads_visualize = tape.gradient(outputs, inputs)[0]

    # Visualize the output of guided backpropagation
    img_guided_bp = np.dstack((grads_visualize[:, :, 0], grads_visualize[:, :, 1], grads_visualize[:, :, 2],)) 

    # Scaling to 0-1      
    img_guided_bp = img_guided_bp - np.min(img_guided_bp)
    img_guided_bp /= img_guided_bp.max()
    plt.subplot(10, 1, i+1)
    plt.imshow(img_guided_bp)
    plt.axis("off")

plt.show()