import torch

from tqdm import tqdm

from data import get_data
from model import Model

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from matplotlib import rcParams
# configure full path for ImageMagick
rcParams['animation.convert_path'] = r'/usr/bin/convert'


learning_rate = 0.001
n_epochs = 80

batch_size = 200

resolution = 40
interval = 100




#### training data
train_pos, train_label = get_data()

train_data = torch.utils.data.TensorDataset(torch.tensor(train_pos, dtype=torch.float32), torch.tensor(train_label))

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size,
    shuffle=True,
)

#### data for creating the image
x_min, x_max = -10, 10
y_min, y_max = -5, 5

w = int((x_max - x_min) * resolution)
h = int((y_max - y_min) * resolution)

grid_x, grid_y = torch.meshgrid(
    torch.linspace(x_min, x_max, w),
    torch.linspace(y_min, y_max, h),
    indexing="xy"
)

test_data = torch.utils.data.TensorDataset(torch.stack([
    grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1))

test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=batch_size, shuffle=False
)


model = Model()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), learning_rate)


ims = list()
fig, ax = plt.subplots()

im_kw = dict(
    origin="lower", extent=(x_min, x_max, y_min, y_max), cmap="RdBu",
    vmin=0, vmax=1
)

scatter_kw = dict(
    #marker=".",
    cmap="bwr"
)


def draw_image(epochs):
    # validate
    image = list()
    for pos, in test_loader:
        with torch.no_grad():
            y_pred = torch.softmax(model(pos), dim=-1)
            image.append(y_pred[:, 0])

    image = torch.cat(image)
    image = image.reshape(h, w)

    im = ax.imshow(image, animated=True, **im_kw)
    dots = ax.scatter(*train_pos.T, c=train_label, animated=True, **scatter_kw)

    ims.append([im, dots])

    if epoch == 0:
        ax.imshow(image, animated=False, **im_kw)
        ax.scatter(*train_pos.T, c=train_label, animated=False, **scatter_kw)

for epoch in tqdm(range(n_epochs)):
    if epoch == 0:
        draw_image(epochs=epoch)

    # train
    for sample, label in train_data:
        optimizer.zero_grad()
        pred = model(sample)
        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()

    draw_image(epochs=epoch + 1)


ax.set_aspect(1)
ani = animation.ArtistAnimation(
    fig, ims, interval=interval, blit=True, repeat_delay=0)

ani.save("training.gif", writer="imagemagick")
plt.show()
