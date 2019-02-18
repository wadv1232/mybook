# 自定义网络

# 自定义激活函数

# 自定义损失函数

1. 自定义函数

不需要维护参数，所有操作都是通过tensor完成 可自动求导

```
def custom_MSELoss(y_pred, y):
    return torch.mean(torch.pow(y_pred - y), 2))
```

1. 继承于 nn.Module

需要维护参数

```
class C_MSELoss(nn.Module):
    def __init__(self,func=torch.pow,powerparams=2):
        self.func = func
        self.powerparam = powerparams
    def forward(self, y_pred, y):
        return torch.mean(self.func((y_pred-y),self.powerparam)
```

1. 继承于nn.autograd.function

不能使用torch操作，需要自定义实现前向反向传播

```
class MyReLU(torch.autograd.Function):
  """
  We can implement our own custom autograd Functions by subclassing
  torch.autograd.Function and implementing the forward and backward passes
  which operate on Tensors.
  """
  @staticmethod
  def forward(ctx, x):
    """
    In the forward pass we receive a context object and a Tensor containing the
    input; we must return a Tensor containing the output, and we can use the
    context object to cache objects for use in the backward pass.
    """
    ctx.save_for_backward(x)
    return x.clamp(min=0)

  @staticmethod
  def backward(ctx, grad_output):
    """
    In the backward pass we receive the context object and a Tensor containing
    the gradient of the loss with respect to the output produced during the
    forward pass. We can retrieve cached data from the context object, and must
    compute and return the gradient of the loss with respect to the input to the
    forward function.
    """
    x, = ctx.saved_tensors
    grad_x = grad_output.clone()
    grad_x[x < 0] = 0
    return grad_x
```

# 自定义初始化参数

# 不同层之间参数共享





