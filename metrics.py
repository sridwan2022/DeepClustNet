import numpy as np
from sklearn.metrics import normalized_mutual_info_score

nmi = normalized_mutual_info_score


def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    # from scipy.optimize import linear_sum_assignment as linear_assignment
    from utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def test_digits(data):    
    data=datasets.load_digits()
    target=data.target
    color=['red','black','green','pink','blue','orange','brown','purple','yellow','grey']
    t1=time.time()
    Y1=TSNE(n_components=2,learning_rate=0.08).fit_transform(data.data)
    t2=time.time()
    t=t2-t1
    print("Visualization cost time: %s"%str(round(t,2)))
    plt.figure(figsize=(8, 6))
    for i in range(10):
        xxx1=Y1[target==i,0]
        xxx2=Y1[target==i,1]
        plt.scatter(xxx1, xxx2, c=color[i], marker='o', edgecolor='none', alpha=1)
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.show()

# Function to plot images
def plot_images(images, labels, title):
    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(f"Label: {labels[i]}")
        plt.axis('off')
    plt.suptitle(title)
    plt.show()

