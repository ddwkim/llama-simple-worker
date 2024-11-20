# Simple multithreaded llama.cpp worker

This project shows creating multiple instances of llama.cpp models to accelerate work processing.
It uses worker pattern, where each thread is run with its own model.
Two shared queues are connected to workers, where one queue is to send works to the worker thread, and the other is to receive results after each work is done.
This may increase the throughput by nearly the number of instances, although some saturation in performance is observed above some point.

## How to build

For CUDA,
```
cmake -B build -DGGML_CUDA=ON -DLLAMA_BUILD_COMMON=ON -DCMAKE_BUILD_TYPE=DEBUG
cmake --build build -j ${num_workers}
```

For Vulkan,
```
cmake -B build -DGGML_VULKAN=ON -DLLAMA_BUILD_COMMON=ON -DCMAKE_BUILD_TYPE=DEBUG
cmake --build build -j ${num_workers}
```

## How to run

Make sure to download gguf model from the huggingface repository. In this case, Llama-3.2-1B-Instruct is used. 
```
./build/bin/llama_simple_worker -m ./Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-f16.gguf -n 256 -b 2500 -ub 512 --temp 0.5 -t 1 -tb 1 -ns 32 -ngl 100
```


## Known Issues

llama.cpp with CUDA backend works seamlessly, but Vulkan backend yields vk::DeviceLostError with more than 1 threads.
