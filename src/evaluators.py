from sklearn.metrics import (confusion_matrix as conf_mx,
                             precision_score, recall_score, accuracy_score,
                             f1_score)
import pandas_ml as pdml
import numpy as np


class MetricsEvaluator:

    def __init__(self, test_x, test_y, trained_model, nb_classes):
        self.X = test_x
        self.Y = test_y
        self.model = trained_model
        self.nb_classes = nb_classes

    @property
    def y_pred(self):
        return self.model.predict(self.X)

    def confusion_matrix(self):
        #return conf_mx(self.Y, self.y_pred)
        cm = np.zeros((self.nb_classes, self.nb_classes))
        for r, p in zip(self.Y, self.y_pred):
            cm[r,p] += 1

        return cm

    def pdml_confusion_matrix(self):
        return pdml.ConfusionMatrix(self.Y, self.y_pred)

    def precision(self, average='weighted'):
        return precision_score(self.Y, self.y_pred, average=average)

    def recall(self, average='weighted'):
        return recall_score(self.Y, self.y_pred, average=average)

    def f_measure(self, average='weighted'):
        return f1_score(self.Y, self.y_pred, average=average)

    def accuracy(self):
        return accuracy_score(self.Y, self.y_pred)