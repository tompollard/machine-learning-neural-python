class Perceptron(object):

    def __init__(self, eta=0.01, epochs=50):
        self.eta = eta
        self.epochs = epochs

    def train(self, X, y):

        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.epochs):
            error = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                print(f"{_} target: {target}")
                print(f"{_} predict: {self.predict(xi)}")
                print(f"{_} error: {int(update != 0.0)}")
                # breakpoint()
                if target != int(self.predict(xi)):
                    print("BBOOOM")
                    breakpoint()
                    error = error + 1
                self.w_[1:] +=  update * xi
                self.w_[0] +=  update
            self.errors_.append(error)
            print(f"{_} ERRORS: {self.errors_}")
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)