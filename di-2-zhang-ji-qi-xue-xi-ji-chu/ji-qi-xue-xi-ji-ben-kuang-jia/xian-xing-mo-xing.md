# 线性回归

```python
import torch
import matplotlib.pyplot as plt
import numpy as np
from time import time
```

# data

线性回归模型 真实权重为$$w = [2,-3.4]^T$$ 偏差 $$b =4.2$$,  
$$y= Xw + b + \epsilon$$,  
$$ X \in R^{1000 x 2}$$,$$\epsilon \sim N(0,0.01)$$

```python
num_inputs = 2
num_examples = 1000
true_w = torch.Tensor([2, -3.4]).reshape(-1,1)
true_b = 4.2
X = torch.randn(num_examples, num_inputs)
noise = torch.normal(mean=torch.Tensor([[0.0]*num_examples]), std=0.001).reshape(-1,1)
y = X.mm(true_w)+true_b + noise
```

```python
def set_figsize(figsize=(5,5)):
    plt.rcParams['figure.figsize'] = figsize
```

```python
set_figsize()
plt.scatter(X[:,1].numpy(),y.numpy())
```

```
<matplotlib.collections.PathCollection at 0x51f9128>
```

![](/assets/output_5_1.png)

```python
import random
```

```python
def data_iter(batch_size, X, labels):
    num_examples = len(X)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0,num_examples,batch_size):
        current_batch_index = indices[i:min(i+batch_size,num_examples)]
        yield X[current_batch_index], labels[current_batch_index]
```

```python
batch_size = 10
for b_x,b_y in data_iter(batch_size, X, y):
    print(b_x,b_y)
    break
```

```
tensor([[-0.7391, -0.1767],
        [-0.0915,  1.0750],
        [ 0.0196, -0.0552],
        [ 1.1901,  1.6073],
        [ 0.9816, -1.5494],
        [-1.2341,  0.2992],
        [ 2.3567, -0.8632],
        [ 0.2445,  0.0408],
        [-0.1181,  0.1341],
        [-1.0128,  1.6987]]) tensor([[ 3.3213],
        [ 0.3609],
        [ 4.4268],
        [ 1.1146],
        [11.4291],
        [ 0.7153],
        [11.8483],
        [ 4.5495],
        [ 3.5074],
        [-3.5997]])
```

# 模型

$$y= Xw + b $$

```python
w = torch.randn(num_inputs,1,requires_grad=True)
b = torch.zeros(1,requires_grad=True)
```

```python
def linreg(X,w,b):
    return X.mm(w) + b
```

```python
y.shape
```

```
torch.Size([1000, 1])
```

# 损失函数

```python
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)).pow(2).sum()
```

# 优化算法

```python
def sgd(params, lr, batch_size):
    for param in params:
        param[:] = param - lr * param.grad/batch_size
        param.grad.zero_()
```

# 代码实现

```python
lr = 0.03
```

```python
num_epochs = 3
net = linreg
loss = squared_loss
```

```python
    # Manually update weights using gradient descent. Wrap in torch.no_grad()
    # because weights have requires_grad=True, but we don't need to track this
    # in autograd.
    # An alternative way is to operate on weight.data and weight.grad.data.
    # Recall that tensor.data gives a tensor that shares the storage with
    # tensor, but doesn't track history.
    # You can also use torch.optim.SGD to achieve this.
```

```python
for epoch in range(num_epochs):
    for b_x,b_y in data_iter(batch_size, X, y):
        l = loss(net(b_x,w,b), b_y)
        l.backward()
        with torch.no_grad():
            sgd([w,b], lr,batch_size)
    with torch.no_grad():
        train_l = loss(net(X,w,b), y)
        print("epoch {}, loss {}".format(epoch+1, train_l))
```

```
epoch 1, loss 0.12716761231422424
epoch 2, loss 0.0010516007896512747
epoch 3, loss 0.0010431952541694045
```

```python
w, b
```

```
(tensor([[ 2.0000],
         [-3.3999]], requires_grad=True), tensor([4.2000], requires_grad=True))
```

## 直接使用公式

$$ w = (X^TX)^{-1}X^Ty$$

```python
W =[[w],[b]]
```

```python
X_n = torch.cat((X, torch.ones(X.size()[0],1)), 1)
A = torch.inverse(torch.mm(X_n.t(), X_n))
W = A.mm(X_n.t()).mm(y)
```

```python
W
```

```
tensor([[ 2.0000],
        [-3.4000],
        [ 4.2000]])
```

# 使用pytorch 模型实现

```python
import torch.nn as nn
```

```python
class LinearRegressionModel(nn.Module):
    def __init__(self,inputdim, outputdim):
        super(LinearRegressionModel, self).__init__()
        self.lin = nn.Linear(inputdim,outputdim)

    def forward(self, x):
        return self.lin(x)
```

```python
net = LinearRegressionModel(2,1)
print(net)
```

```
LinearRegressionModel(
  (lin): Linear(in_features=2, out_features=1, bias=True)
)
```

```python
list(net.parameters())
```

```
[Parameter containing:
 tensor([[-0.5415, -0.5236]], requires_grad=True), Parameter containing:
 tensor([-0.3564], requires_grad=True)]
```

```python
optimizer = torch.optim.SGD(net.parameters(), lr=1e-4)
```

```python
print(X.shape)
print(y.shape)
```

```
torch.Size([1000, 2])
torch.Size([1000, 1])
```

```python
criterion = nn.MSELoss()
```

```python
import visdom
```

```python
vis = visdom.Visdom(env="Linear Regression")
```

```
WARNING:root:Setting up a new session...
```

```python
for epoch in range(150):
    for b_x,b_y in data_iter(batch_size, X, y):
        y_pred = net(b_x)
        loss = criterion(y_pred, b_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    vis.line(X=torch.FloatTensor([epoch]), Y = torch.FloatTensor([loss.item()]),win="loss1", update="append")
```

```python
list(net.parameters())
```

```
[Parameter containing:
 tensor([[ 1.9019, -3.2465]], requires_grad=True), Parameter containing:
 tensor([3.9708], requires_grad=True)]
```

```python

```

![loss](/assets/newplot.png)

