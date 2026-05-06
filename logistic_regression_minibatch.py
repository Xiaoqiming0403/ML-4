import numpy as np 
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer# 导入乳腺癌数据集
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

data = load_breast_cancer()
X = data.data # 特征矩阵，形状 (n_samples, n_features)
y = data.target # 标签向量，0/1

#数据处理（归一化，分为minibatch）


X_train_1 = X[:int(len(X)*0.8)]
y_train = y[:int(len(y)*0.8)]
X_test = X[int(len(X)*0.8):]
y_test = y[int(len(y)*0.8):]

X_train = (X_train_1 - X_train_1.mean(axis=0)) / X_train_1.std(axis=0) # 归一化
X_test = (X_test - X_train_1.mean(axis=0)) / X_train_1.std(axis=0) # 归一化

#编程基础课给出简洁的逻辑回归实现，直接套用公式X.T @(f(X @ w) - y)即可

def f(X,w):
    z = np.clip(X @ w, -500, 500) # 防止溢出
    return 1/(1+np.exp(-z))

def gradient(X,y,w):
    return X.T @ (f(X,w) - y)/len(y)

def train(X,y,lr=0.01,epochs=100,batch_size=32,seed = 40):
    n, d = X.shape
    rng = np.random.default_rng(seed) #这是初始化随机种子
    w = np.zeros(d)
    losses = []
    for epoch in range(epochs):
        # 打乱数据
        idx = rng.permutation(n) #permutation是排列的意思
        X_shuffled = X[idx]
        y_shuffled = y[idx]

        #分批处理
        for i in range(0, n ,batch_size):
            Xb = X_shuffled[i:i+batch_size]
            yb = y_shuffled[i:i+batch_size]
            g = gradient(Xb,yb,w)
            w -= lr * g

        losses.append(-np.mean(y * np.log(np.clip(f(X,w), 1e-15, 1-1e-15)) + (1-y) * np.log(np.clip(1-f(X,w), 1e-15, 1-1e-15))))
    return w, losses

w, losses = train(X_train,y_train)
#测试
y_pred = f(X_test,w) >= 0.5
accuracy = np.mean(y_pred == y_test)
print(f"Test Accuracy: {accuracy:.4f}")

plt.plot(losses)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid()
plt.savefig("training_loss.png")
plt.show()

#要画二维决策边界，我们需要选择两个特征进行可视化。这里我们选择前两个特征进行绘制。
def plot_decision_boundary(X, y, w):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Z = f(np.c_[xx.ravel(), yy.ravel()], w)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, levels=50, cmap='RdBu', alpha=0.6)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', s=100)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')
    plt.grid()
    plt.savefig("decision_boundary.png")
    plt.show()

plot_decision_boundary(X_train[:, :2], y_train, w[:2])

# t-SNE 可视化函数：先（可选）用 PCA 降维，再用 t-SNE 可视化并保存图片
def plot_tsne(X, y, title="t-SNE", filename="tsne.png", pca_dim=30, perplexity=30, max_iter=1000, random_state=42):
    # X assumed already normalized
    from inspect import signature

    X_emb = X
    if X.shape[1] > pca_dim:
        pca = PCA(n_components=pca_dim, random_state=random_state)
        X_emb = pca.fit_transform(X)

    # sklearn 版本差异：有的版本使用 max_iter，有的使用 n_iter
    sig_params = signature(TSNE.__init__).parameters
    tsne_kwargs = dict(n_components=2, perplexity=perplexity, random_state=random_state)
    if 'max_iter' in sig_params:
        tsne_kwargs['max_iter'] = max_iter
    elif 'n_iter' in sig_params:
        tsne_kwargs['n_iter'] = max_iter

    tsne = TSNE(**tsne_kwargs)
    X_tsne = tsne.fit_transform(X_emb)

    plt.figure(figsize=(8, 6))
    for label in np.unique(y):
        idx = (y == label)
        plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1], s=30, alpha=0.8, label=str(label), edgecolors='k')
    plt.title(title)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend(title='class')
    plt.grid(True)
    plt.savefig(filename)
    plt.show()

# 调用 t-SNE 可视化（训练集与测试集）
plot_tsne(X_train, y_train, title='t-SNE of Train Set', filename='tsne_train.png')
plot_tsne(X_test, y_test, title='t-SNE of Test Set', filename='tsne_test.png')
