# Fusing preceding batch normalization and succeeding convolution
PyTorch script to fuse BatchNorm layers into succeeding Conv or Linear layers in FX graph mode

```sh

> python ./fuse_bn_conv_relu.py
Model(
  (bn): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv): Conv2d(8, 16, kernel_size=(1, 1), stride=(1, 1))
  (relu): ReLU()
)
GraphModule(
  (bn): ConvReLU2d(
    (0): Conv2d(8, 16, kernel_size=(1, 1), stride=(1, 1))
    (1): ReLU()
  )
)
tensor([[[ 2.9802e-08,  0.0000e+00,  0.0000e+00,  5.9605e-08,  0.0000e+00],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00],
         [-5.9605e-08,  0.0000e+00,  0.0000e+00, -1.7881e-07,  0.0000e+00],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  5.9605e-08,  0.0000e+00],
         [ 0.0000e+00,  2.9802e-08,  7.4506e-08,  0.0000e+00,  5.9605e-08]],

        ......

        [[ 5.9605e-08,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00],
         [ 0.0000e+00,  0.0000e+00, -1.1921e-07, -5.9605e-08,  0.0000e+00],
         [ 0.0000e+00,  0.0000e+00, -5.9605e-08,  0.0000e+00,  0.0000e+00],
         [-1.1921e-07,  0.0000e+00,  0.0000e+00,  5.9605e-08,  0.0000e+00],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00, -8.9407e-08]]],
       grad_fn=<SelectBackward0>)
```
