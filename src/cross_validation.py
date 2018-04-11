from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold
from src.evaluators import MetricsEvaluator


class CrossValidator:

    def __init__(self, x, y, model_initialiser, data_transformer_initialiser=None):
        self.model_initialiser = model_initialiser
        self.transformer_initialiser = data_transformer_initialiser
        self.X = x
        self.Y = y

    def kfold_cross_validation(self, nb_folds, shuffle=True,
                               return_train_evaluators=False):
        return self._base_cross_validation(KFold(nb_folds, shuffle=shuffle),
                                           return_train_evaluators=return_train_evaluators)

    def stratified_kfold_cross_validation(self, nb_folds, shuffle=True,
                                          return_train_evaluators=False):
        return self._base_cross_validation(StratifiedKFold(nb_folds, shuffle=shuffle),
                                           return_train_evaluators=return_train_evaluators)

    #def group_kfold_cross_validation(self, nb_folds, groups):
    #    return self._base_cross_validation(GroupKFold())

    def _base_cross_validation(self, kfold_object, return_train_evaluators=False):
        metr_evals = []
        trn_evals = []
        if self.transformer_initialiser is not None:
            transformer = self.transformer_initialiser()
            transformer.fit(self.X)

        for train_index, test_index in kfold_object.split(self.X, self.Y):
            x_tr, x_te = self.X[train_index], self.X[test_index]
            y_tr, y_te = self.Y[train_index], self.Y[test_index]

            if self.transformer_initialiser is not None:
                x_tr = transformer.transform(x_tr)
                x_te = transformer.transform(x_te)

            model = self.model_initialiser()
            model.train(x_tr, y_tr)
            metr_evals.append(MetricsEvaluator(x_te, y_te, model, len(set(self.Y))))
            trn_evals.append(MetricsEvaluator(x_tr, y_tr, model, len(set(self.Y))))

        if return_train_evaluators:
            return metr_evals, trn_evals
        else:
            return metr_evals
