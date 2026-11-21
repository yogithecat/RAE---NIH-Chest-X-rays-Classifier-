import numpy as np
import constants as c

from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import AUC, BinaryAccuracy
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D


from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    hamming_loss,
    roc_auc_score,
)

from sklearn.preprocessing import MultiLabelBinarizer

class Model:
    """
    CNN model built for classifying lung diseases.

    Predicts from six different classifications: No Finding, Infiltration,
    Atelectasis, Effusion, Nodule, and Effusion|Infiltration.

    Attributes:
        model (tensorflow.keras.models.Sequential) : Sequential model that is trained.
    """

    def __init__(self, image_size: tuple[int, int]=c.IMAGE_SIZE, num_classes=len(c.CLASSES_LIST)) -> None:
        """
        Creates the Sequential model and adds all layers.
        """
        image_shape = (image_size[0], image_size[1], 1,)
        model = Sequential([
            Input(shape=image_shape),
            Conv2D(16, kernel_size=(2, 2),activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(8, kernel_size=(2, 2),activation="relu"),
            MaxPooling2D(pool_size=(1, 1)),
            Flatten(),
            Dropout(0.05),
            Dense(256,  activation='relu'),
            Dense(num_classes, activation='sigmoid')
        ])
        
        # visualize the model design
        model.summary()
        self.model = model

    def train_model(self, x_train: np.ndarray, y_train: np.ndarray):
        """
        Trains the model using multi-label binary indicator targets.

        Args:
            x_train : The features examples of the training data.
            y_train : Binary indicator matrix of shape (num_samples, num_classes).        
        """
        # training the model
        optim = Adam(learning_rate=0.001) #learning rate is the "eta" in our equation
        loss_fn = CategoricalCrossentropy()

        self.model.compile(
            loss=loss_fn,
            optimizer=optim,
            metrics=[
            BinaryAccuracy(name="binary_acc"),
            AUC(name="auc", multi_label=True, num_labels=y_train.shape[1]),
            ],
        )

        batch_size = c.MODEL_BATCH_SIZE
        epochs = c.MODEL_EPOCHS

        self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

    # def predict_labels(self, x_test: np.ndarray, y_test: np.ndarray):
    #     """
    #     Uses the model to predict labels from x_test, then evaluates accuracy.

    #     Args:
    #         x_test : The features examples of the test data.
    #         y_test : The labels for the test data.
    #     """
    #     # predictions
    #     predictions = self.model.predict(x_test)
    #     print (predictions)
    #     predicted_class_indices = predictions.argmax(axis=1)
    #     print (predicted_class_indices)
    #     acc = accuracy_score(y_test, predicted_class_indices)
    #     print (f"Testing accuracy = {acc}")


    def predict_multilabel(
        self,
        x_test: np.ndarray,
        y_test: np.ndarray,
        mlb: MultiLabelBinarizer,
        threshold: float = 0.5,
    ):
        """Predicts multi-label binary indicators and evaluates multi-label metrics."""
        y_test_lists = [s.split("|") if s else [] for s in y_test]
        y_true_bin = mlb.transform(y_test_lists)

        probs = self.model.predict(x_test)
        y_pred_bin = (probs >= threshold).astype(int)

        return {
            "exact_match_acc": accuracy_score(y_true_bin, y_pred_bin),
            "hamming_loss": hamming_loss(y_true_bin, y_pred_bin),
            "macro_f1": f1_score(y_true_bin, y_pred_bin, average="macro", zero_division=0),
            "micro_f1": f1_score(y_true_bin, y_pred_bin, average="micro", zero_division=0),
            "macro_roc_auc": roc_auc_score(y_true_bin, probs, average="macro"),
            "macro_pr_auc": average_precision_score(y_true_bin, probs, average="macro"),
        }