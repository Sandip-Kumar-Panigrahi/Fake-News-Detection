from sklearn.base import BaseEstimator, ClassifierMixin


class WeightedHybridClassifier(BaseEstimator, ClassifierMixin):
    """
    Blend MultinomialNB and LogisticRegression probability outputs.
    Label mapping expects dataset labels: 0 => Fake, 1 => Real.
    """

    def __init__(self, nb_model, lr_model, nb_weight=0.45, lr_weight=0.55):
        self.nb_model = nb_model
        self.lr_model = lr_model
        self.nb_weight = float(nb_weight)
        self.lr_weight = float(lr_weight)
        self.classes_ = [0, 1]

    def fit(self, X, y):
        self.nb_model.fit(X, y)
        self.lr_model.fit(X, y)
        self.classes_ = self.lr_model.classes_
        return self

    def predict_proba(self, X):
        nb_probs = self.nb_model.predict_proba(X)
        lr_probs = self.lr_model.predict_proba(X)
        total = self.nb_weight + self.lr_weight
        return ((nb_probs * self.nb_weight) + (lr_probs * self.lr_weight)) / total

    def predict(self, X):
        probs = self.predict_proba(X)
        max_indices = probs.argmax(axis=1)
        return [self.classes_[i] for i in max_indices]
