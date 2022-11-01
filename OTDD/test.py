import sys

sys.path.append("../OTDD")

from otdd import *

import numpy as np
import matplotlib.pylab as plt
from plot import *

from ot.datasets import make_data_classif

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def get_data(n):
    x = make_data_classif("3gauss", n)
    y = make_data_classif("3gauss2", n, classes=5)

    y_p = (y[0] @ np.array([[1, 0], [0, 1]])) + 5

    return [x[0], x[1] - 1], [y_p, y[1] - 1]


x, y = get_data(5000)

fig, ax = plt.subplots(1, 2, figsize=(16, 8))

ax[0].scatter(x[0][:, 0], x[0][:, 1], s=2.0, label="Dataset1")
ax[0].scatter(y[0][:, 0], y[0][:, 1], s=2.0, label="Dataset2")

colors = ["r", "b", "g"]
for c in range(3):
    ax[1].scatter(
        x[0][:, 0][x[1] == c],
        x[0][:, 1][x[1] == c],
        c=colors[c],
        s=2.0,
        label=c,
        alpha=0.5,
    )
    ax[1].scatter(
        y[0][:, 0][y[1] == c], y[0][:, 1][y[1] == c], c=colors[c], s=2.0, alpha=0.5
    )

ax[0].set_xticks([])
ax[0].set_yticks([])
ax[1].set_xticks([])
ax[1].set_yticks([])

ax[0].legend()
ax[1].legend()

ax[0].set_title("Unlabeled")
ax[1].set_title("Labeled")

plt.savefig("unlabeled.jpg")

# to ArrayDataset class
data_x = ArrayDataset(x[0], x[1])
data_y = ArrayDataset(y[0], y[1])

# Initial method class
distance_function = POTDistance(distance_metric="euclidean")
cost_function = SinkhornCost(distance_function, 0.02)
# calculate unlabeled OT by sinkhorn
cost, coupling, M_dist = cost_function.distance(data_x.features, data_y.features)
# visualize result
plot_coupling(
    coupling,
    M_dist,
    data_x.labels,
    data_y.labels,
    data_x.classes,
    data_y.classes,
    figsize=(5, 5),
    cmap="OrRd",
)
plot_coupling_network(
    data_x.features,
    data_y.features,
    data_x.labels,
    data_y.labels,
    coupling,
    plot_type="ds",
)
