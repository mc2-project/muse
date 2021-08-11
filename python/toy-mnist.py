# The full neural network code!
###############################
import numpy as np
import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras import backend as K
from numpy.linalg import matrix_rank

K.set_floatx('float64')
train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

# Normalize the images.
train_images = (train_images / 255) - 0.5
test_images = (test_images / 255) - 0.5

# Flatten the images.
train_images = train_images.reshape((-1, 784))
test_images = test_images.reshape((-1, 784))

# Build the model.
model = Sequential([
  Dense(64, activation='relu', input_shape=(784,)),
  Dense(64, activation='relu'),
  Dense(10, activation='softmax'),
])

# Compile the model.
model.compile(
  optimizer='adam',
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)

# # Train the model.
# model.fit(
#   train_images,
#   to_categorical(train_labels),
#   epochs=5,
#   batch_size=32,
# )

# # Evaluate the model.
# model.evaluate(
#   test_images,
#   to_categorical(test_labels)
# )

# # Save the model to disk.
# model.save_weights('model.h5')

# Load the model from disk later using:
model.load_weights('model.h5')
layer_weights = []
for layer in model.layers:
    print(layer.activation)
    weights = layer.get_weights()
    layer_weights.append(K.constant(weights[0], dtype='float64'))
    print(np.array(weights[0]).shape)

print("last layer weights:")
print(np.array(layer_weights[2][0]))


def predict(cur_state, fault=1):
    cur_state = K.constant(cur_state, shape = (1, 784), dtype='float64')
    cur_state = K.relu(K.dot(cur_state, layer_weights[0]))
    cur_state = K.relu(K.dot(cur_state, layer_weights[1]))
    last_shape = K.shape(cur_state)
    shift = np.zeros(last_shape)
    shift[0][0] = fault
    cur_state = cur_state + K.constant(shift)
    cur_state = (K.dot(cur_state, layer_weights[2]))
    return cur_state


def predict_upto_2(cur_state, fault=1):
    cur_state = K.constant(cur_state, shape = (1, 784), dtype='float64')
    cur_state = K.relu(K.dot(cur_state, layer_weights[0]))
    last_shape = K.shape(cur_state)
    shift = np.zeros(last_shape)
    shift[0][0] = fault
    cur_state = cur_state + K.constant(shift)
    cur_state = K.dot(cur_state, layer_weights[1])
    return cur_state


def predict_recover_2(cur_state, fault=1):
    cur_state = predict_upto_2(cur_state, fault)
    
    cur_state = (K.dot(cur_state, layer_weights[2]));
    return cur_state


if __name__ == 'main.py':
    print("Matrix rank:")
    print(matrix_rank(np.array(layer_weights[2])))
    last_layer = np.array(layer_weights[2])
    last_layer_transpose = last_layer.T
    step1 = np.linalg.inv(np.dot(last_layer_transpose, last_layer))
    last_layer_inverse = np.dot(step1, last_layer_transpose)

    print("Testing")
    # Evaluate the model.
    for i in range(1):
        print("\n\nCustom predictor with fault 1")
        custom_result_1 = predict(test_images[i])
        print(np.array(custom_result_1))
        print(np.argmax(custom_result_1))

        print("\n\nCustom predictor without fault")
        custom_result_2 = predict(test_images[i], fault=0)
        print(np.array(custom_result_2))
        print(np.argmax(custom_result_2))

        print("Last layer row 0:")
        print(np.array(custom_result_1 - custom_result_2))

        print("\n\nDefault predictor")
        default_result = model.predict(test_images[i:i+1])
        print(np.array(default_result))
        print(np.argmax(default_result))

        print("Second last layer row 0:")
        result_upto_2_one = predict_upto_2(test_images[i])
        result_upto_2_two = predict_upto_2(test_images[i], fault=0)
        print(np.array(result_upto_2_one - result_upto_2_two))

        result_upto_2_one = predict_recover_2(test_images[i])
        result_upto_2_two = predict_recover_2(test_images[i], fault=0)
        val = np.array(result_upto_2_one - result_upto_2_two)
        print(np.dot(val, last_layer_inverse))
