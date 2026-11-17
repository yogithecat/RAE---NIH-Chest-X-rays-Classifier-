import numpy as np
import constants as c

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dropout,Dense, AveragePooling2D
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score

class Model:
    """
    CNN model built for classifying lung diseases.

    Predicts from six different classifications: No Finding, Infiltration,
    Atelectasis, Effusion, Nodule, and Effusion|Infiltration.

    Attributes:
        model (tensorflow.keras.models.Sequential) : Sequential model that is trained.
    """

    def __init__(self, image_size: tuple[int, int]=c.IMAGE_SIZE) -> None:
        """
        Creates the Sequential model and adds all layers.
        """
        image_shape = (image_size[0], image_size[1], 1,)
        model = Sequential()
        model.add(Conv2D(16, kernel_size=(2, 2), input_shape=image_shape, activation="sigmoid"))
        model.add(MaxPooling2D(pool_size=(1, 1)))
        model.add(Conv2D(8, kernel_size=(2, 2), input_shape=image_shape, activation="sigmoid"))
        model.add(MaxPooling2D(pool_size=(1, 1)))
        model.add(Flatten())

        # adding dropout
        model.add(Dropout(0.05))
        model.add(Dense(3000,  activation='relu'))
        model.add(Dense(c.NUM_CLASSES,  activation='softmax')) # total_num_classes, activation
        
        # visualize the model design
        model.summary()
        self.model = model

    def train_model(self, x_train: np.ndarray, y_train: np.ndarray):
        """
        Trains the model using given labeled data.

        Args:
            x_train : The features examples of the training data.
            y_train : The labels for the training data.
        """
        # training the model
        optim = Adam(learning_rate=0.001) #learning rate is the "eta" in our equation
        loss_fn = CategoricalCrossentropy()

        self.model.compile(loss=loss_fn,optimizer=optim,metrics=["accuracy"])

        batch_size = 128
        epochs = 15

        self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

    def predict_labels(self, x_test: np.ndarray, y_test: np.ndarray):
        """
        Uses the model to predict labels from x_test, then evaluates accuracy.

        Args:
            x_test : The features examplesof the test data.
            y_test : The labels for the test data.
        """
        # predictions
        predictions = self.model.predict(x_test)
        print (predictions)
        predicted_class_indices = predictions.argmax(axis=1)
        print (predicted_class_indices)
        acc = accuracy_score(y_test,predicted_class_indices)
        print (f"Testing accuracy = {acc}")