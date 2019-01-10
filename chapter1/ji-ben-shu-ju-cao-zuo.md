# pytorch 基本数据操作

## 创建

> In\[1\]:
>
> ```
> torch.arange(12)
> ```
>
> Out\[2\]:
>
> ```
> tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
> ```

## 运算

## 广播

## 索引

## 运算内存开销

torch.tensor\(\)  always copies  data  . If you have a Tensor  data  and just want to change its  requires\_grad  flag, use  requires\_grad\_\(\)  or  detach\(\)  to avoid a copy.If you have a numpy array and want to avoid a copy, use[`torch.as_tensor()`](https://pytorch.org/docs/stable/torch.html#torch.as_tensor)



## tensor和numpy 相互转化

```
p=np.ones((2,3))
```

```
d=torch.tensor(p)
```

```
d
```

Out\[39\]:

```
tensor([[1., 1., 1.],
        [1., 1., 1.]], dtype=torch.float64)
```

In \[40\]:

```
d.numpy()
```

Out\[40\]:

```
array([[1., 1., 1.],
       [1., 1., 1.]])
```



