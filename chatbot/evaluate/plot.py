# coding:utf8
# @Time    : 18-5-21 下午3:03
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import matplotlib as mpl
# zhfont = mpl.font_manager.FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc')
def getChineseFont():
    try:
        return mpl.font_manager.FontProperties(fname='/usr/share/fonts/truetype/arphic/uming.ttc')
    except:
        return None

def plot_confusion_matrix(y_true, y_test, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = confusion_matrix(y_true, y_test)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_attention(sentences, attentions, labels, **kwargs):
    fig, ax = plt.subplots(**kwargs)
    im = ax.imshow(attentions, interpolation='nearest',
                   vmin=attentions.min(), vmax=attentions.max())
    plt.colorbar(im, shrink=0.5, ticks=[0, 1])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontproperties=getChineseFont())
    # Loop over data dimensions and create text annotations.
    for i in range(attentions.shape[0]):
        for j in range(attentions.shape[1]):
            text = ax.text(j, i, sentences[i][j],
                           ha="center", va="center", color="b", size=10,
                           fontproperties=getChineseFont())

    ax.set_title("Attention Visual")
    fig.tight_layout()
    plt.show()


