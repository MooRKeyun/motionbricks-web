"""Smoke test: Warp CUDA kernel + TensorRT import on this machine."""

import warp as wp

wp.init()
wp.set_device("cuda:0")


@wp.kernel
def scale_kernel(a: wp.array(dtype=float)):
    i = wp.tid()
    a[i] = a[i] * 2.0 + 1.0


a = wp.array([1.0, 2.0, 3.0], dtype=float, device="cuda:0")
wp.launch(scale_kernel, dim=3, inputs=[a])
wp.synchronize()
print("kernel result:", a.numpy())

dp = wp.get_device("cuda:0")
print("device:", dp.name, "arch:", dp.arch, "toolkit:", dp.cuda_toolkit_version)

import tensorrt as trt

print("tensorrt:", trt.__version__)
print("trt builder available:", trt.Builder is not None)
