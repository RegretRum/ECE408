// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void load_auxiliary(float *output, float* auxiliary, int len) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i * 2 * BLOCK_SIZE - 1 < len && 0 <= i * 2 * BLOCK_SIZE - 1) {
    auxiliary[i] = output[i * 2 * BLOCK_SIZE - 1];
  } else {
    auxiliary[i] = 0;
  }
  __syncthreads();
}

__global__ void add(float* deviceOutput, float* device_auxiliaryOutput, int len) {
  int i = blockIdx.x * BLOCK_SIZE * 2 + threadIdx.x;
  if (i < len) {
    deviceOutput[i] += device_auxiliaryOutput[i / (2 * BLOCK_SIZE)];
  }
  if (i + BLOCK_SIZE < len) {
    deviceOutput[i + BLOCK_SIZE] += device_auxiliaryOutput[i / (2 * BLOCK_SIZE)];
  }
}

__global__ void scan(float *input, float *output, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  __shared__ float xy[BLOCK_SIZE * 2];
  int tx = threadIdx.x;
  int start = 2 * blockIdx.x * blockDim.x;

  xy[tx] = (start + tx < len)?input[start + tx]:0.0;
  xy[tx + blockDim.x] = (start + tx + blockDim.x < len) ? input[start + tx + blockDim.x] : 0.0;

  for(int stride = 1;stride <= blockDim.x;stride *= 2){
    __syncthreads();
    int index = (tx + 1) * 2 * stride - 1;
    if(index < 2 * blockDim.x){
      xy[index] += xy[index - stride];
    }
  }

  for(int stride = blockDim.x / 2; stride > 0; stride /= 2){
    __syncthreads();
    int index = (tx + 1) * 2 * stride - 1;
    if(index + stride < 2 * blockDim.x){
      xy[index + stride] += xy[index];
    }
  }
  __syncthreads();

  if(start + tx < len){
    output[start + tx] = xy[tx];
  }

  if(start + tx + blockDim.x < len){
    output[start + tx + blockDim.x] = xy[tx + blockDim.x];
  }
}


int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  float *device_auxiliaryInput;
  float *device_auxiliaryOutput;
  int numElements; // number of elements in the list


  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  int block_dim = ceil(numElements / (2.0 * BLOCK_SIZE));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  cudaMalloc((void **)&device_auxiliaryInput, block_dim * sizeof(float));
  cudaMalloc((void **)&device_auxiliaryOutput, block_dim * sizeof(float));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 GridDim(ceil(numElements / (2.0 * BLOCK_SIZE)), 1, 1);
  dim3 BlockDim(BLOCK_SIZE, 1, 1);
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce

  scan<<<GridDim, BlockDim>>>(deviceInput, deviceOutput, numElements);
  cudaDeviceSynchronize();
  load_auxiliary<<<dim3(block_dim, 1, 1), dim3(1, 1, 1) >>>(deviceOutput, device_auxiliaryInput, numElements);
  cudaDeviceSynchronize();
  scan<<<dim3(ceil(block_dim / (2.0 * BLOCK_SIZE)),1,1), BlockDim>>>(device_auxiliaryInput, device_auxiliaryOutput, block_dim);
  cudaDeviceSynchronize();
  add<<<GridDim, BlockDim>>>(deviceOutput, device_auxiliaryOutput, numElements);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(device_auxiliaryInput);
  cudaFree(device_auxiliaryOutput);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}
