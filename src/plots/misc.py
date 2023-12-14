from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def plot_hist(y, title):
    plt.hist(y, bins=[0, 0.5, 1.0], edgecolor='black')
    plt.xticks([0, 1])
    plt.xlabel('Value')
    plt.ylabel('Count')
    plt.title('Distribution ' + title)
    plt.show()


def plot_conf_mat(cm):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.subplots_adjust(bottom=.25, left=.25)
    plt.show()
