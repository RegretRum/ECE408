#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#define TILE_WIDTH 16

//__constant__ float Mask[6000];
__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    extern __shared__ float input_tile[];

    const int H_out = (H - K) / S + 1;
    const int W_out = (W - K) / S + 1;

    int blockWidth = (TILE_WIDTH - 1) * S + K;

    #define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    #define tile_3d(i2, i1, i0) input_tile[(i2) * (blockWidth * blockWidth) + (i1) * (blockWidth) + i0]

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;
    int Width_grid = ceil(1.0 * W_out / TILE_WIDTH);
    int h = (by / Width_grid) * TILE_WIDTH + ty;
    int w = (by % Width_grid) * TILE_WIDTH + tx;

    int h_start = h  - ty;
    int w_start = w  - tx;

    // pre-load the input array into shared memory
    for (int c = 0; c < C; c++)
        for (int p = ty; p < blockWidth; p += TILE_WIDTH)
            for (int q = tx; q < blockWidth; q += TILE_WIDTH)
                tile_3d(c, p, q) = in_4d(bz, c, h_start * S + p, w_start * S + q);

    __syncthreads();
    if (h < H_out && w < W_out) {
        float res = 0.0;
        for (int c = 0; c < C; c++)
            for (int p = 0; p < K; p++)
                for (int q = 0; q < K; q++)
                    if(ty * S + p < blockWidth && tx * S + q < blockWidth)
                        res += tile_3d(c, ty * S + p, tx * S + q) * mask_4d(bx, c, p, q);
        out_4d(bz, bx, h, w) = res;
        //atomicAdd((&out_4d(bz,bx,h,w)),res);
    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d
    #undef tile_3d
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
    //cudaMemcpyToSymbol(Mask, host_mask, M * C * K * K * sizeof(float));
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
    int blockWidth = (TILE_WIDTH - 1) * S + K;
    
    size_t sharedX_size = C * blockWidth * blockWidth * sizeof(float);
    dim3 dimgrid(M, Z, B);
    dim3 dimblock(TILE_WIDTH, TILE_WIDTH, 1);
    conv_forward_kernel <<<dimgrid, dimblock,sharedX_size>>> (device_output, device_input, device_mask, B, M, C, H, W, K, S);
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