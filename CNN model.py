"""
3-block VGG baseline model with drop regularization.
1-epoch
"""
import sys
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.optimizers import gradient_descent_v2
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten


"""
Define train and test paths
"""
train_path = 'D:/DATASETS/tomato_data/train/'
test_path = 'D:/DATASETS/tomato_data/test/'


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
Plot diagnostic learning curves
"""


def summarize_diagnostics(history):
    """
    Plot loss
    """
    plt.subplot(211)
    plt.title('Cross Entropy loss')
    plt.plot(
        history.history['loss'],
        color='blue',
        label='train')
    plt.plot(
        history.history['val_loss'],
        color='orange',
        label='test')

    """
    Plot accuracy
    """
    plt.subplot(212)
    plt.title('Classification Accuracy')
    plt.plot(
        history.history['accuracy'],
        color='blue',
        label='train')
    plt.plot(
        history.history['val_accuracy'],
        color='orange',
        label='test')

    """
    Save plot to file
    """
    filename = sys.argv[0].split('/')[-1]
    plt.savefig(filename + '_plot.png')
    plt.close()


"""
Run the test harness for evaluating 
a model
"""


def run_test_harness():
    """
    define the model
    """
    model = define_model()

    """
    Create data generator 
    """
    datagen = ImageDataGenerator(rescale=1.0/255.0)

    """
    Prepare Iterators
    """
    train_it = datagen.flow_from_directory(
        train_path,
        class_mode='binary',
        batch_size=64,
        target_size=(200, 200))

    test_it = datagen.flow_from_directory(
        test_path,
        class_mode='binary',
        batch_size=64,
        target_size=(200, 200))

    """
    fit model
    """
    history = model.fit_generator(
        train_it,
        steps_per_epoch=len(train_it),
        validation_data=test_it,
        validation_steps=len(test_it),
        epochs=1,
        verbose=1)

    """
    Evaluate the model
    """
    _, acc = model.evaluate_generator(
        test_it,
        steps=len(test_it),
        verbose=1)
    print('> %.3f' % (acc * 100.0))

    """
    Learning curves
    """
    summarize_diagnostics(history)


"""
Entry point, run the test harness
"""
run_test_harness()
