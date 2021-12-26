
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def read_data():
    xy = []
    label = []
    with open('points-two-classes.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)
        for row in csv_reader:
            xy.append([float(row[0]), float(row[1])])
            label.append(int(row[2]))

    print(xy, label)
    return xy, label


def plot_data(xy, label):
    # set_trace()
    x = np.asarray(xy)[:, 0]
    y = np.asarray(xy)[:, 1]
    colors = ['red', 'green']

    fig = plt.figure(figsize=(8, 8))
    plt.title('The Dataset')
    plt.xlim(0., 1.)
    plt.ylim(0., 1.)
    plt.scatter(x, y, c=label,
                cmap=matplotlib.colors.ListedColormap(colors))

    # plt.show()
    plt.savefig('original-dataset.png')


if __name__ == '__main__':
    xy, label = read_data()
    plot_data(xy, label)
