#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#define TILE_WIDTH 16

__constant__ float Mask[6000];


__global__ void conv_forward_kernel(float * __restrict__ output, const float * __restrict__ input, const float *  __restrict__ mask, const int B, const int M, const int C, const int H, const int W, const int K,const int S)
{
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;


    #define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    //#define mask_4d(i3, i2, i1, i0) mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    #define mask_4d(i3, i2, i1, i0) Mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0] // using constant memory

    // Insert your GPU convolution kernel code here
    int W_tile = ceil(1.0 * W_out / TILE_WIDTH);
    int H_tile = ceil(1.0 * H_out / TILE_WIDTH);

    int bxi = blockIdx.x;
    int byi = blockIdx.y;
    int h = (blockIdx.z / W_tile) * TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.z % W_tile) * TILE_WIDTH + threadIdx.x;
    float acc = 0.0f;

    for (int c = 0; c < C; c++) {
        if (K <= 3){
            acc += in_4d(bxi, c, h*S+0, w*S+0) * mask_4d(byi, c, 0, 0);
            acc += in_4d(bxi, c, h*S+0, w*S+1) * mask_4d(byi, c, 0, 1);
            acc += in_4d(bxi, c, h*S+0, w*S+2) * mask_4d(byi, c, 0, 2);

            acc += in_4d(bxi, c, h*S+1, w*S+0) * mask_4d(byi, c, 1, 0);
            acc += in_4d(bxi, c, h*S+1, w*S+1) * mask_4d(byi, c, 1, 1);
            acc += in_4d(bxi, c, h*S+1, w*S+2) * mask_4d(byi, c, 1, 2);

            acc += in_4d(bxi, c, h*S+2, w*S+0) * mask_4d(byi, c, 2, 0);
            acc += in_4d(bxi, c, h*S+2, w*S+1) * mask_4d(byi, c, 2, 1);
            acc += in_4d(bxi, c, h*S+2, w*S+2) * mask_4d(byi, c, 2, 2);
        }
        
        if (K > 3){
            acc += in_4d(bxi, c, h*S+0, w*S+0) * mask_4d(byi, c, 0, 0);
            acc += in_4d(bxi, c, h*S+0, w*S+1) * mask_4d(byi, c, 0, 1);
            acc += in_4d(bxi, c, h*S+0, w*S+2) * mask_4d(byi, c, 0, 2);
            acc += in_4d(bxi, c, h*S+0, w*S+3) * mask_4d(byi, c, 0, 3);
            acc += in_4d(bxi, c, h*S+0, w*S+4) * mask_4d(byi, c, 0, 4);
            acc += in_4d(bxi, c, h*S+0, w*S+5) * mask_4d(byi, c, 0, 5);
            acc += in_4d(bxi, c, h*S+0, w*S+6) * mask_4d(byi, c, 0, 6);

            acc += in_4d(bxi, c, h*S+1, w*S+0) * mask_4d(byi, c, 1, 0);
            acc += in_4d(bxi, c, h*S+1, w*S+1) * mask_4d(byi, c, 1, 1);
            acc += in_4d(bxi, c, h*S+1, w*S+2) * mask_4d(byi, c, 1, 2);
            acc += in_4d(bxi, c, h*S+1, w*S+3) * mask_4d(byi, c, 1, 3);
            acc += in_4d(bxi, c, h*S+1, w*S+4) * mask_4d(byi, c, 1, 4);
            acc += in_4d(bxi, c, h*S+1, w*S+5) * mask_4d(byi, c, 1, 5);
            acc += in_4d(bxi, c, h*S+1, w*S+6) * mask_4d(byi, c, 1, 6);

            acc += in_4d(bxi, c, h*S+2, w*S+0) * mask_4d(byi, c, 2, 0);
            acc += in_4d(bxi, c, h*S+2, w*S+1) * mask_4d(byi, c, 2, 1);
            acc += in_4d(bxi, c, h*S+2, w*S+2) * mask_4d(byi, c, 2, 2);
            acc += in_4d(bxi, c, h*S+2, w*S+3) * mask_4d(byi, c, 2, 3);
            acc += in_4d(bxi, c, h*S+2, w*S+4) * mask_4d(byi, c, 2, 4);
            acc += in_4d(bxi, c, h*S+2, w*S+5) * mask_4d(byi, c, 2, 5);
            acc += in_4d(bxi, c, h*S+2, w*S+6) * mask_4d(byi, c, 2, 6);

            acc += in_4d(bxi, c, h*S+3, w*S+0) * mask_4d(byi, c, 3, 0);
            acc += in_4d(bxi, c, h*S+3, w*S+1) * mask_4d(byi, c, 3, 1);
            acc += in_4d(bxi, c, h*S+3, w*S+2) * mask_4d(byi, c, 3, 2);
            acc += in_4d(bxi, c, h*S+3, w*S+3) * mask_4d(byi, c, 3, 3);
            acc += in_4d(bxi, c, h*S+3, w*S+4) * mask_4d(byi, c, 3, 4);
            acc += in_4d(bxi, c, h*S+3, w*S+5) * mask_4d(byi, c, 3, 5);
            acc += in_4d(bxi, c, h*S+3, w*S+6) * mask_4d(byi, c, 3, 6);

            acc += in_4d(bxi, c, h*S+4, w*S+0) * mask_4d(byi, c, 4, 0);
            acc += in_4d(bxi, c, h*S+4, w*S+1) * mask_4d(byi, c, 4, 1);
            acc += in_4d(bxi, c, h*S+4, w*S+2) * mask_4d(byi, c, 4, 2);
            acc += in_4d(bxi, c, h*S+4, w*S+3) * mask_4d(byi, c, 4, 3);
            acc += in_4d(bxi, c, h*S+4, w*S+4) * mask_4d(byi, c, 4, 4);
            acc += in_4d(bxi, c, h*S+4, w*S+5) * mask_4d(byi, c, 4, 5);
            acc += in_4d(bxi, c, h*S+4, w*S+6) * mask_4d(byi, c, 4, 6);

            acc += in_4d(bxi, c, h*S+5, w*S+0) * mask_4d(byi, c, 5, 0);
            acc += in_4d(bxi, c, h*S+5, w*S+1) * mask_4d(byi, c, 5, 1);
            acc += in_4d(bxi, c, h*S+5, w*S+2) * mask_4d(byi, c, 5, 2);
            acc += in_4d(bxi, c, h*S+5, w*S+3) * mask_4d(byi, c, 5, 3);
            acc += in_4d(bxi, c, h*S+5, w*S+4) * mask_4d(byi, c, 5, 4);
            acc += in_4d(bxi, c, h*S+5, w*S+5) * mask_4d(byi, c, 5, 5);
            acc += in_4d(bxi, c, h*S+5, w*S+6) * mask_4d(byi, c, 5, 6);

            acc += in_4d(bxi, c, h*S+6, w*S+0) * mask_4d(byi, c, 6, 0);
            acc += in_4d(bxi, c, h*S+6, w*S+1) * mask_4d(byi, c, 6, 1);
            acc += in_4d(bxi, c, h*S+6, w*S+2) * mask_4d(byi, c, 6, 2);
            acc += in_4d(bxi, c, h*S+6, w*S+3) * mask_4d(byi, c, 6, 3);
            acc += in_4d(bxi, c, h*S+6, w*S+4) * mask_4d(byi, c, 6, 4);
            acc += in_4d(bxi, c, h*S+6, w*S+5) * mask_4d(byi, c, 6, 5);
            acc += in_4d(bxi, c, h*S+6, w*S+6) * mask_4d(byi, c, 6, 6);
        }
    }
    if (h < H_out && w < W_out) {
        out_4d(bxi, byi, h, w) = acc;
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
    cudaMemcpyToSymbol(Mask, host_mask, M * C * K * K * sizeof(float));
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
    int tilesize = (TILE_WIDTH - 1) * S + K;
    size_t share = (C * tilesize * tilesize * sizeof(float)); //Tile

    //dim3 dimgrid(M, B, Z); // FP16-1
    //dim3 dimgrid(M, Z, B); // Tiled
    dim3 dimgrid(B, M, Z); // Original
    dim3 dimblock(TILE_WIDTH, TILE_WIDTH, 1);
    conv_forward_kernel <<<dimgrid, dimblock, 1>>> (device_output, device_input, device_mask, B, M, C, H, W, K, S);
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
