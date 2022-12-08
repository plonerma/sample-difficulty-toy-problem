import matplotlib.pyplot as plt

from data import get_data


pos, label = get_data()

fig, ax = plt.subplots()
ax.scatter(*pos.T, c=label, cmap="tab10", vmax=9, marker=".")

ax.set_aspect(1)

plt.show()
