import numpy as np
import matplotlib
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
matplotlib.use('TkAgg')


def data_rep(a, b, c):
    x = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10]
    p = []

    i = 0
    while i < len(a):
        p.append(a[i])
        p.append(b[i])
        p.append(c[i])
        i += 1

    model = make_pipeline(PolynomialFeatures(2), LinearRegression())
    model.fit(np.array(x).reshape(-1, 1), p)
    x_reg = np.arange(1, 11)
    y_reg = model.predict(x_reg.reshape(-1, 1))

    return x_reg, y_reg
