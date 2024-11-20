# Simple multithreaded llama.cpp worker

This project shows creating multiple instances of llama.cpp models to accelerate work processing.
It uses worker pattern, where each thread is run with its own model.
Two shared queues are connected to workers, where one queue is to send works to the worker thread, and the other is to receive results after each work is done.
This may increase the throughput by nearly the number of instances, although some saturation in performance is observed above some point.

## Known Issues

llama.cpp with CUDA backend works seamlessly, but Vulkan backend yields vk::DeviceLostError with more than 1 threads.
