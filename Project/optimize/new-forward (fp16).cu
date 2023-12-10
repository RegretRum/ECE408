#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include "cuda_fp16.h"

#define TILE_WIDTH 16
__global__ void conv_forward_kernel(half *output, const half *input, const half *mask, const int B, const int M, const int C, const int H, const int W, const int K,const int S)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    S - stride step length
    */

    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
    // (void)H_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)W_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    #define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;

    int W_tile = ceil(1.0 * W_out / TILE_WIDTH);
    int H_tile = ceil(1.0 * H_out / TILE_WIDTH);
    
    int h = (bz / W_tile) * TILE_WIDTH + ty;
    int w = (bz % W_tile) * TILE_WIDTH + tx;

    half acc = 0.0;

    for (int c = 0; c < C; c++) {
        for (int p = 0; p < K; p++) {
            for (int q = 0; q < K; q++) {
                if (h * S + p < H && w * S + q < W) {
                    acc = __hadd(__hmul(in_4d(bx, c, h*S+p, w*S+q), mask_4d(by, c, p, q)), acc);
                }
            }
        }
    }
    if (h < H_out && w < W_out) {
        out_4d(bx, by, h, w) = acc;
    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d
}

__global__ void float2halfArray(const float *input, half *output, const int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = idx; i < size; i += blockDim.x * gridDim.x) {
        output[i] = __float2half(input[i]);
    }
}

__global__ void half2floatArray(half *input, float *output, const int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = idx; i < size; i += blockDim.x * gridDim.x) {
        output[i] = __half2float(input[i]);
    }
}
	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Allocate memory for device_output, device_input, and device_mask
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
    cudaMalloc((void**) device_output_ptr, B * M * H_out * W_out * sizeof(float));
    cudaMalloc((void**) device_input_ptr, B * C * H * W * sizeof(float));
    cudaMalloc((void**) device_mask_ptr, M * C * K * K * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(*device_input_ptr, host_input, B * C * H * W * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, M * C * K * K * sizeof(float), cudaMemcpyHostToDevice);
    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }
   
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Set the kernel dimensions and call the kernel
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
    int W_tile = ceil(1.0 * W_out / TILE_WIDTH);
    int H_tile = ceil(1.0 * H_out / TILE_WIDTH);
    int Z = W_tile * H_tile;


    half *device_input_half;
    half *device_mask_half;
    half *device_output_half;
    cudaMalloc((void**)&device_input_half, sizeof(half) * B * C * H * W);
    cudaMalloc((void**)&device_mask_half, sizeof(half) * M * C * K * K);
    cudaMalloc((void**)&device_output_half, sizeof(half) * B * M * H_out * W_out);

    dim3 dimGrid(16, 1, 1);
    dim3 dimBlock(512, 1, 1);
    float2halfArray<<<dimGrid, dimBlock>>>(device_input, device_input_half, B * C * H * W);
    cudaDeviceSynchronize();
    float2halfArray<<<dimGrid, dimBlock>>>(device_mask, device_mask_half, M * C * K * K);
    cudaDeviceSynchronize();

    dim3 dimGrid2(B, M, Z);
    dim3 dimBlock2(TILE_WIDTH, TILE_WIDTH, 1);
    conv_forward_kernel<<<dimGrid2, dimBlock2>>>(device_output_half, device_input_half, device_mask_half, B, M, C, H, W, K, S);
    cudaDeviceSynchronize();

    dim3 dimGrid3(16, 1, 1);
    dim3 dimBlock3(1024, 1, 1);
    half2floatArray<<<dimGrid3, dimBlock3>>>(device_output_half, device_output, B * M * H_out * W_out);
    cudaDeviceSynchronize();

    cudaFree(device_input_half);
    cudaFree(device_mask_half);
    cudaFree(device_output_half);

}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Copy the output back to host
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
    cudaMemcpy(host_output, device_output, B * M * W_out * H_out * sizeof(float), cudaMemcpyDeviceToHost);
   
    // Free device memory
    cudaFree(device_input);
    cudaFree(device_output);
    cudaFree(device_mask);

}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
