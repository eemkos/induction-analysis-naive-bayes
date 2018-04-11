import numpy as np


class BaseNaiveBayes:

    def train(self, x, y):
        raise NotImplementedError

    def transform(self, x):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

    def get_p_x(self, x):
        raise NotImplementedError

    def get_p_x_y(self, c, x):
        raise NotImplementedError

    def get_p_y(self, c):
        raise NotImplementedError


class NaiveBayes(BaseNaiveBayes):
    
    def __init__(self, zero_frequency_fill=False):
        self.P_y = {}
        self.P_x_y = {}
        self.P_x = {}
        self.possible_classes = []
        self.nb_attributes = 0
        self.zero_frequency_fill = zero_frequency_fill
        self.nb_training_samples = 0

    def train(self, x, y):
        assert len(x) == len(y)
        self.nb_training_samples, self.nb_attributes = x.shape
        self.possible_classes = list(set(y))

        for i in range(self.nb_attributes):
            self.P_x_y[i] = {}
            self.P_x[i] = {}
            for value in set(x[:, i]):
                where_is_value = x[:, i] == value
                self.P_x[i][value] = 1.*where_is_value.sum()/self.nb_training_samples
                self.P_x_y[i][value] = {}
                for c in self.possible_classes:
                    where_is_class = y == c
                    self.P_y[c] = 1.*where_is_class.sum()/self.nb_training_samples
                    where_is_class_and_value = where_is_class & where_is_value
                    value_count = where_is_class_and_value.sum()
                    if self.zero_frequency_fill:
                        value_count += 1
                    self.P_x_y[i][value][c] = 1. * value_count/self.nb_training_samples

    def transform(self, x):
        probabs = []
        for c in self.possible_classes:
            p_x_y = self.get_p_x_y(c, x)
            p_x = self.get_p_x(x)
            probabs.append(p_x_y * self.get_p_y(c) / p_x)
        return np.asarray(probabs).T

    def _probability(self, x_row):
        probabs = []
        for c in self.possible_classes:
            p_x_y = 1
            p_x = 1
            for i in range(self.nb_attributes):
                p_x_y *= self._get_p_x_y(c, i, x_row)
                p_x *= self._get_p_x(i, x_row)
            probabs.append(p_x_y * self.P_y[c] / p_x)
        return np.asarray(probabs)

    def predict(self, x):
        return self.transform(x).argmax(axis=1)

    def _get_p_x(self, i, x):
        try:
            return self.P_x[i][x[i]]
        except KeyError:
            return (1. if self.zero_frequency_fill else 1e-3) / self.nb_training_samples

    def _get_p_x_y(self, c, i, x):
        try:
            return self.P_x_y[i][x[i]][c]
        except KeyError:
            return (1. if self.zero_frequency_fill else 0) / self.nb_training_samples

    def get_p_x(self, x):
        res=[]
        for j in range(x.shape[0]):
            p_x = 1
            for i in range(x.shape[1]):
                p_x *= self._get_p_x(i,x[j, :])
            res.append(p_x)
        return np.asarray(res)

    def get_p_x_y(self, c, x):
        res = []
        for j in range(x.shape[0]):
            p_x_y = 1
            for i in range(x.shape[1]):
                p_x_y *= self._get_p_x_y(c, i, x[j, :])
            res.append(p_x_y)
        return np.asarray(res)

    def get_p_y(self, c):
        return self.P_y[c]


class GaussianNaiveBayes(BaseNaiveBayes):

    def __init__(self):
        self.P_x_y_means = {}
        self.P_x_y_stds = {}
        self.P_y = {}
        self.P_x_means = {}
        self.P_x_stds = {}
        self.possible_classes = []
        self.nb_attributes = 0

    def train(self, x, y):
        assert len(x) == len(y)
        nb_samples, self.nb_attributes = x.shape
        self.possible_classes = list(set(y))

        self.P_x_y_means = {c: [] for c in self.possible_classes}
        self.P_x_y_stds = {c: [] for c in self.possible_classes}

        for c in self.possible_classes:
            where_is_class = y == c
            self.P_y[c] = 1.*where_is_class.sum() / len(y)
            sub_x = x[where_is_class, :]
            for i in range(self.nb_attributes):
                self.P_x_y_means[c].append(sub_x[:, i].mean())
                std = sub_x[:, i].std()
                std += 1e-9 if std == 0 else 0
                self.P_x_y_stds[c].append(std)

        self.P_x_means = {i: x[:, i].mean() for i in range(self.nb_attributes)}
        stds = {i: x[:, i].std() for i in range(self.nb_attributes)}
        self.P_x_stds = {i: stds[i] if stds[i] != 0 else 1e-9 for i in range(self.nb_attributes)}

    def predict(self, x):
        return self.transform(x).argmax(axis=1)

    def transform(self, x):
        probabs = []
        for c in self.possible_classes:
            p_x_y = self.get_p_x_y(c, x)
            p_x = self.get_p_x(x)
            probabs.append(p_x_y * self.P_y[c] / p_x)
        return np.asarray(probabs).T

    def get_p_y(self, c):
        return self.P_y[c]

    def get_p_x_y(self, c, x):
        p_x_y = np.ones(x.shape[0])
        for i in range(x.shape[1]):
            p_x_y *= self._gaussian_probability(x[:, i],
                                                self.P_x_y_means[c][i],
                                                self.P_x_y_stds[c][i])
        return p_x_y

    def get_p_x(self, x):
        p_x = np.ones(x.shape[0])
        for i in range(x.shape[1]):
            p_x *= self._gaussian_probability(x[:, i],
                                              self.P_x_means[i],
                                              self.P_x_stds[i])

        p_x = p_x.clip(1e-9)
        return p_x

    @staticmethod
    def _gaussian_probability(x, mean, std):
        std_sq = std**2
        a = 1. / np.sqrt(2*np.pi*(std_sq))
        b = - (x - mean)**2
        c = 2*std_sq
        return a * np.exp(b/c)


class MixedNaiveBayes(BaseNaiveBayes):

    def __init__(self, gaussian_attributes_indices, zero_frequency_fill=False):
        self.gaussian_indices = list(set(gaussian_attributes_indices))
        self.nb_discrete = NaiveBayes(zero_frequency_fill)
        self.nb_gaussian = GaussianNaiveBayes()
        self.possible_classes = []

    def train(self, x, y):
        self.possible_classes = list(set(y))
        gm = self._gaussian_mask(x)

        self.nb_gaussian.train(x[:, gm],y)
        self.nb_discrete.train(x[:, ~gm], y)

    def transform(self, x):
        probabs = []
        for c in self.possible_classes:
            p_x_y = self.get_p_x_y(c, x)
            p_x = self.get_p_x(x)
            probabs.append(p_x_y * self.get_p_y(c) / p_x)
        return np.asarray(probabs).T

    def get_p_x(self, x):
        gm = self._gaussian_mask(x)
        return (self.nb_gaussian.get_p_x(x[:, gm])
                * self.nb_discrete.get_p_x(x[:, ~gm]))

    def get_p_y(self, c):
        if len(self.gaussian_indices) > 0:
            return self.nb_gaussian.get_p_y(c)
        else:
            return self.nb_discrete.get_p_y(c)

    def get_p_x_y(self, c, x):
        gm = self._gaussian_mask(x)
        return (self.nb_gaussian.get_p_x_y(c, x[:, gm])
                * self.nb_discrete.get_p_x_y(c, x[:, ~gm]))

    def predict(self, x):
        return self.transform(x).argmax(axis=1)

    def _gaussian_mask(self, x):
        gaussian_mask = np.zeros(x.shape[1]).astype(int)
        gaussian_mask[self.gaussian_indices] = 1
        return gaussian_mask.astype(bool)

