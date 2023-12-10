// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256
//@@ insert code here

__global__ void floatToUnsignedChar(float *input, unsigned char *output, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = (unsigned char) (255*input[idx]); 
  }
}

__global__ void RGBtoGray(unsigned char *input, unsigned char *output, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = (unsigned char) (0.21*input[3*idx] + 0.71*input[3*idx+1] + 0.07*input[3*idx+2]);
  }
}

__global__ void computeHistogram(unsigned char *input, unsigned int *output, int len) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  __shared__ unsigned int histo[HISTOGRAM_LENGTH];

  if (threadIdx.x < HISTOGRAM_LENGTH) {
    histo[threadIdx.x] = 0;
  }
  __syncthreads();

  if (idx < len) {
    int pos = input[idx];
    atomicAdd(&(histo[pos]), 1);
  }
  __syncthreads();

  if (threadIdx.x < HISTOGRAM_LENGTH) {
    atomicAdd(&(output[threadIdx.x]), histo[threadIdx.x]);
  }
}


__global__ void scan(unsigned int *histogram, float *cdf, int size) {
  __shared__ float xy[HISTOGRAM_LENGTH];

  int tx = threadIdx.x;
  xy[tx] = (tx < HISTOGRAM_LENGTH)?histogram[tx]:0.0;
  xy[tx + blockDim.x] = (tx + blockDim.x)?histogram[tx + blockDim.x]:0.0;
  
  for (int stride = 1; stride <= blockDim.x; stride *= 2) {
    __syncthreads();
    int index = (tx+1) * 2 * stride - 1;
    if (index < HISTOGRAM_LENGTH) {
      xy[index] += xy[index - stride];
    }
  }

  for (int stride = ceil(HISTOGRAM_LENGTH/4.0); stride > 0; stride /= 2) {
    __syncthreads();
    int index = (tx+1)*stride*2 - 1;
    if(index + stride < HISTOGRAM_LENGTH) {
      xy[index + stride] += xy[index];
    }
  }
  __syncthreads();
  if (tx < HISTOGRAM_LENGTH){
    cdf[tx] = ((float) (xy[tx]*1.0)/size);
  }

  if (tx + blockDim.x < HISTOGRAM_LENGTH) {
    cdf[tx+blockDim.x] = ((float) (xy[tx+blockDim.x]*1.0)/size);
  }
}


__global__ void histogramEqualize(unsigned char *inout, float *cdf, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    float equalized = 255.0*(cdf[inout[idx]]-cdf[0])/(1.0-cdf[0]);
    inout[idx] = (unsigned char) (min(max(equalized, 0.0), 255.0));
  }
}

__global__ void unsignedCharToFloat(unsigned char *input, float *output, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = (float) (input[idx]/255.0);
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here
  float   *deviceImageFloat;
  unsigned char *deviceImageChar;
  unsigned char *deviceImageCharGrayScale;
  unsigned int  *deviceImageHistogram;
  float   *deviceImageCDF;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  hostInputImageData = wbImage_getData(inputImage);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostOutputImageData = wbImage_getData(outputImage);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  // Allocating GPU memory
  cudaMalloc((void **)&deviceImageFloat, imageWidth*imageHeight*imageChannels*sizeof(float));
  cudaMalloc((void **)&deviceImageChar, imageWidth*imageHeight*imageChannels*sizeof(unsigned char));
  cudaMalloc((void **)&deviceImageCharGrayScale, imageWidth*imageHeight*sizeof(unsigned char));
  cudaMalloc((void **)&deviceImageHistogram, HISTOGRAM_LENGTH*sizeof(unsigned int));
  //cudaMemset((void *) deviceImageHistogram, 0, HISTOGRAM_LENGTH *sizeof(unsigned int));
  cudaMalloc((void **)&deviceImageCDF, HISTOGRAM_LENGTH*sizeof(float));
  
  // Copy data to GPU
  cudaMemcpy(deviceImageFloat, hostInputImageData, 
             imageWidth*imageHeight*imageChannels*sizeof(float),cudaMemcpyHostToDevice);
  

  // convert to unsigned char
  dim3 dimGrid1(ceil(imageWidth*imageHeight*imageChannels/256.0), 1, 1);
  dim3 dimBlock1(256,1,1);
  floatToUnsignedChar<<<dimGrid1,dimBlock1>>>(deviceImageFloat, 
                                         deviceImageChar, imageWidth*imageHeight*imageChannels);
  cudaDeviceSynchronize();
  // convert to gray
  dim3 dimGrid2(ceil(imageWidth*imageHeight/256.0), 1, 1);
  dim3 dimBlock2(256,1,1);
  RGBtoGray<<<dimGrid2,dimBlock2>>>(deviceImageChar, 
                                 deviceImageCharGrayScale, imageWidth*imageHeight);
  cudaDeviceSynchronize();
  // compute histogram
  dim3 dimGrid3(ceil(imageWidth*imageHeight/256.0), 1, 1);
  dim3 dimBlock3(256,1,1);
  computeHistogram<<<dimGrid3,dimBlock3>>>(deviceImageCharGrayScale, 
                                 deviceImageHistogram, imageWidth*imageHeight);
  cudaDeviceSynchronize();
  // scan
  dim3 dimGrid4(1, 1, 1);
  dim3 dimBloc4(128,1,1);
  scan<<<dimGrid4, dimBloc4>>>(deviceImageHistogram, deviceImageCDF, imageWidth*imageHeight);
  cudaDeviceSynchronize();
  // histogram equalization function
  dim3 dimGrid5(ceil(imageWidth*imageHeight*imageChannels/256.0), 1, 1);
  dim3 dimBlock5(256,1,1);
  histogramEqualize<<<dimGrid5,dimBlock5>>>(deviceImageChar, 
                                  deviceImageCDF, imageWidth*imageHeight*imageChannels);
  cudaDeviceSynchronize();
  // cast to float
  dim3 dimGrid6(ceil(imageWidth*imageHeight*imageChannels/256.0), 1, 1);
  dim3 dimBlock6(256,1,1);
  unsignedCharToFloat<<<dimGrid6,dimBlock6>>>(deviceImageChar, 
                                 deviceImageFloat, imageWidth*imageHeight*imageChannels);
  cudaDeviceSynchronize();
  cudaMemcpy(hostOutputImageData, deviceImageFloat,
             imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyDeviceToHost);

  // Check Solution 
  wbImage_setData(outputImage, hostOutputImageData);
  wbSolution(args, outputImage);

  // Free GPU Memory 
  cudaFree(deviceImageFloat);
  cudaFree(deviceImageChar);
  cudaFree(deviceImageCharGrayScale);
  cudaFree(deviceImageHistogram);
  cudaFree(deviceImageCDF);
  // Free CPU Memory
  free(hostInputImageData);
  free(hostOutputImageData);
  return 0;
}
