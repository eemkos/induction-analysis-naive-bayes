import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder


class DataTransformer:

    def __init__(self, nb_target_values):
        self.possible_values = np.asarray(range(nb_target_values)).astype(int)

    def fit(self, fitting_data):
        raise NotImplementedError

    def transform(self, data):
        raise NotImplementedError


class DiscreteEncoder(DataTransformer):

    def __init__(self):
        super().__init__(0)
        self.label_encoder = LabelEncoder()

    def fit(self, fitting_data):
        self.label_encoder.fit(fitting_data)

    def transform(self, data):
        return self.label_encoder.transform(data)


class MultipleAttributeTransformer(DataTransformer):

    def __init__(self, attr_index_2_digitizer):
        super().__init__(0)
        self.digitizers_dict = attr_index_2_digitizer

    def fit(self, fitting_data):
        _, nb_attrs = fitting_data.shape

        for i in range(nb_attrs):
            if i in self.digitizers_dict and self.digitizers_dict[i] is not None:
                self.digitizers_dict[i].fit(fitting_data[:, i])

    def transform(self, data):
        _, nb_attrs = data.shape
        resulting_data = np.zeros(data.shape).astype(int)
        for i in range(nb_attrs):
            if i in self.digitizers_dict and self.digitizers_dict[i] is not None:
                resulting_data[:, i] = self.digitizers_dict[i].transform(data[:, i])
            else:
                resulting_data[:, i] = data[:, i].astype(int)
        return resulting_data


class KMeansDigitizer(DataTransformer):

    def __init__(self, nb_target_values):
        super().__init__(nb_target_values)
        self.kmeans = KMeans(nb_target_values)
        self.possible_values = np.asarray(range(nb_target_values))

    def fit(self, fitting_data):
        self.kmeans.fit(fitting_data.reshape(-1, 1))

    def transform(self, data):
        return self.kmeans.predict(data.reshape(-1, 1))


class RoundDigitizer(DataTransformer):

    def __init__(self, nb_target_values):
        super().__init__(nb_target_values)
        self.min = None
        self.max = None

    def fit(self, fitting_data):
        self.min = fitting_data.min()
        self.max = fitting_data.max()

    def transform(self, data):
        data_clipped = data.clip(self.min, self.max)
        data_shifted = data_clipped - self.min
        mx = self.max - self.min
        target_mx = self.possible_values.max()
        data_scaled = data_shifted * (target_mx/mx)
        data_rounded = np.round(data_scaled).astype(int)
        return data_rounded.astype(int)
