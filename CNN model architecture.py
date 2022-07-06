from keras.models import Sequential
from keras.optimizers import gradient_descent_v2
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.utils.vis_utils import plot_model


"""
Define CNN model
"""


def define_model():
    model = Sequential()
    model.add(Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation='relu',
        kernel_initializer='he_uniform',
        padding='same',
        input_shape=(200, 200, 3)))

    model.add(MaxPooling2D((2, 2)))

    # apply a dropout of 20% here
    model.add(Dropout(0.2))

    """
    add a second block with 64 filters.
    """

    model.add(Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation='relu',
        kernel_initializer='he_uniform',
        padding='same'))

    model.add(MaxPooling2D((2, 2)))

    # apply a dropout of 20% here
    model.add(Dropout(0.2))

    """
    add a THIRD block with 128 filters.
    """

    model.add(Conv2D(
        filters=128,
        kernel_size=(3, 3),
        activation='relu',
        kernel_initializer='he_uniform',
        padding='same'))

    model.add(MaxPooling2D((2, 2)))

    # apply a dropout of 20% here
    model.add(Dropout(0.2))

    """
    Continue
    """

    model.add(Flatten())

    model.add(Dense(128,
                    activation='relu',
                    kernel_initializer='he_uniform'))

    # apply a dropout of 50% here
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))

    """
    Compile the model
    """
    opt = gradient_descent_v2.SGD(
        learning_rate=0.001,
        momentum=0.9)
    model.compile(
        optimizer=opt,
        loss='binary_crossentropy',
        metrics=['accuracy'])

    return model


"""
Define model
"""
model = define_model()


"""
Summarize model
"""
model.summary()

"""
Plot the model
"""
plot_model(
    model,
    to_file='CNN model architecture_plot.png',
    show_shapes=True,
    show_layer_names=True)
