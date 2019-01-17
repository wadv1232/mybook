# LeNet

![](/assets/lenet.png)

LeNet PyTorch实现

```
class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(50 * 4 * 4, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 50 * 4 * 4)
        x = self.classifier(x)
        x = F.softmax(x, 1)
        return x
```

`model = LeNet(10)`

`ts.summary(model, input_size=(1,28,28))`

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 20, 24, 24]             520
         MaxPool2d-2           [-1, 20, 12, 12]               0
            Conv2d-3             [-1, 50, 8, 8]          25,050
         MaxPool2d-4             [-1, 50, 4, 4]               0
            Linear-5                  [-1, 500]         400,500
              ReLU-6                  [-1, 500]               0
            Linear-7                   [-1, 10]           5,010
================================================================
Total params: 431,080
Trainable params: 431,080
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.15
Params size (MB): 1.64
Estimated Total Size (MB): 1.80
----------------------------------------------------------------
```

# AlexNet



