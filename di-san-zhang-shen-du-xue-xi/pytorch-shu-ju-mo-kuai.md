`torch.utils.data.Dataset`

抽象类，派生类需要重载` __len__ `和` __getitem__ `

```
def __len__(self):
    return ***

def __getitem__(self. idx):
    return ***
```

```
torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, collate_fn=<function default_collate>, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None)
```



