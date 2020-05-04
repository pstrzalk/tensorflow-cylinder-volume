import math
import random
import sys
import tensorflow as tf

print("TensorFlow version", tf.__version__, "\n")

EPOCHS_COUNT = 15
MODEL_FILE_NAME = 'cylinder-volume-model.h5'
NUMBER_OF_DATA_POINTS = 100000
NUMBER_OF_TRAIN_POINTS = math.floor(NUMBER_OF_DATA_POINTS * 0.9)
NUMBER_OF_VALIDATION_POINTS = math.floor((NUMBER_OF_DATA_POINTS - NUMBER_OF_TRAIN_POINTS) / 2)
# ^^ the rest are test points
NUMBER_OF_TEST_CASES = 20 # additional manual test points

def calculate_volume(data_point):
    return math.pi * data_point[:, 0] * data_point[:, 0] * data_point[:, 1]

def generate_data_points(count):
    # points between .1 and .6 so that they don't have to be normalized later on
    # because 3.14 * .6 * .6 * .6 < 1
    # since the numbers get really small, use float64 not to get lost in precision
    return tf.random.uniform((count, 2), minval=0.000001, maxval=.6, dtype=tf.dtypes.float64)

def build_model(x_train, y_train, x_validate, y_validate):
    model = tf.keras.models.Sequential([
      tf.keras.layers.Dense(16, activation='elu', input_shape=(2,)),
      tf.keras.layers.Dense(16, activation='elu'),
      tf.keras.layers.Dense(16, activation='elu'),
      # Dropout layers make the learing slower and relative errors higher (during testing)
      # So far best was 3x16 or 3x32 or 3x128 using relu
      tf.keras.layers.Dense(1)
    ])

    model.compile(
        optimizer='adam', loss='mse', metrics=['mean_squared_error']
    )

    model.fit(
        x_train, y_train, epochs=EPOCHS_COUNT, validation_data=(x_validate, y_validate)
    )

    return model

data_points = generate_data_points(NUMBER_OF_DATA_POINTS)
volumes = tf.reshape(calculate_volume(data_points), (NUMBER_OF_DATA_POINTS, 1))

x_train = data_points[0:(NUMBER_OF_TRAIN_POINTS - 1)]
y_train = volumes[0:(NUMBER_OF_TRAIN_POINTS - 1)]

x_validate = data_points[NUMBER_OF_TRAIN_POINTS:(NUMBER_OF_TRAIN_POINTS + NUMBER_OF_VALIDATION_POINTS)]
y_validate = volumes[NUMBER_OF_TRAIN_POINTS:(NUMBER_OF_TRAIN_POINTS + NUMBER_OF_VALIDATION_POINTS)]

x_test = data_points[(NUMBER_OF_TRAIN_POINTS + NUMBER_OF_VALIDATION_POINTS):NUMBER_OF_DATA_POINTS]
y_test = volumes[(NUMBER_OF_TRAIN_POINTS + NUMBER_OF_VALIDATION_POINTS):NUMBER_OF_DATA_POINTS]

if '--load-model' in sys.argv:
    model = tf.keras.models.load_model(MODEL_FILE_NAME)
else:
    model = build_model(x_train, y_train, x_validate, y_validate)

model.summary()
model.evaluate(x_test, y_test, verbose=2)

# Some manual testing
test_data_points = generate_data_points(NUMBER_OF_TEST_CASES)
sum_relative_error = 0
for i in range(NUMBER_OF_TEST_CASES):
    test_data_point = test_data_points[i:(i+1)]

    volume = calculate_volume(test_data_point)
    predicted_volume = model.predict(test_data_point)

    relative_error = float(abs(volume - predicted_volume)/volume * 100)
    sum_relative_error += relative_error

    print("=== Tested Relative Error {:.2f}%".format(relative_error))
print("=== Mean Tested Relative Error {:.2f}%".format(sum_relative_error/NUMBER_OF_TEST_CASES))

if '--save-model' in sys.argv:
    model.save(MODEL_FILE_NAME)
