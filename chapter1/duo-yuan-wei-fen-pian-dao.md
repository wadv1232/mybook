# 导数

Tensor and Function are interconnected and build up an acyclic graph, that encodes a complete history of computation. Each tensor has a .grad\_fn attribute that references a Function that has created the Tensor \(except for Tensors created by the user - their grad\_fn is None\).



