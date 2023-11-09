// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE 256

//@@ insert code here

__global__ void convert_to_uint(float *input, unsigned int *output, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    output[i] = (unsigned int)(input[i] * 255);
  }
}

__global__ void convert_to_grey(unsigned int *input, unsigned int *output, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int index = i * 3;
  if (i < size) {
    output[i] = (unsigned int)(0.21 * input[index] + 0.71 * input[index + 1] + 0.07 * input[index + 2]);
  }
}

__global__ void compute_histogram(unsigned int *input, unsigned int *output, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    atomicAdd(&(output[input[i]]), 1);
  }
}


__global__ void scan(unsigned int *input, float *output, float* blockSum, int len, int total) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  
  // use the Brent-Kung Tree algorithm
  __shared__ float XY[2 * BLOCK_SIZE];
  int tx = threadIdx.x;

  
  // first load the elements into XY, each thread load two elements
  int index = 2 * blockIdx.x * blockDim.x + tx;
  if (index < len) {
    XY[tx] = (float)(input[index] / (float)total);
  } else {
    XY[tx] = 0.0;
  }
  if (index + blockDim.x < len) {
    XY[tx + blockDim.x] = (float)(input[index + blockDim.x] / (float)total);
  } else {
    XY[tx + blockDim.x] = 0.0;
  }
  int stride = 1;
  // reduction phase
  while (stride < 2 * BLOCK_SIZE) {
    __syncthreads();
    int index = (tx + 1) * stride * 2 - 1;
    if (index < 2 * BLOCK_SIZE) {
      XY[index] += XY[index - stride];
    }
    stride *= 2;
  }
  stride = BLOCK_SIZE / 2;
  // post reduction phase
  while (stride > 0) {
    __syncthreads();
    int index = (tx + 1) * stride * 2 - 1;
    if (index + stride < 2 * BLOCK_SIZE) {
      XY[index + stride] += XY[index];
    }
    stride /= 2;
  }

  // write the result to output
  __syncthreads();
  if (index < len) {
    output[index] = XY[tx];
  }
  if (index + blockDim.x < len) {
    output[index + blockDim.x] = XY[tx + blockDim.x];
  }
  // write the last element of each block to blockSum
  if (tx == 0) {
    blockSum[blockIdx.x] = XY[2 * BLOCK_SIZE - 1];
  }
}

__global__ void add(float *output, float* blockSum, int len) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float sum;
  sum = 0.0;
  if (blockIdx.x > 0 && threadIdx.x == 0) {  // Skip the first block and let only the first thread in other blocks calculate the block sum

      for (int i = 0; i < blockIdx.x; i++) {
          sum += blockSum[i];
      }
  }

  __syncthreads();

  if (blockIdx.x > 0 && index < len) {
      output[index] += sum;
  }
}

__global__ void correct_color(unsigned int *input, float *cdf, int size) {
  float cdfmin = cdf[0];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int temp;
  if (i < size) {
    temp = (unsigned int)((cdf[input[i]] - cdfmin) / (1 - cdfmin) * 255);
    input[i] = min(max(temp, 0), 255);
  }
}

__global__ void convert_to_float(unsigned int *input, float *output, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    output[i] = (float)(input[i] / 255.0);
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
  float *deviceInputImageData;
  float *deviceOutputImageData;
  unsigned int *deviceInputImageData_uint;
  unsigned int *devicegreyImage;
  unsigned int *devicehistogram;
  float *devicecdf;
  float *deviceblockSum;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  // malloc device space here
  cudaMalloc((void**)&deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc((void**)&deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc((void**)&deviceInputImageData_uint, imageWidth * imageHeight * imageChannels * sizeof(unsigned int));
  cudaMalloc((void**)&devicegreyImage, imageWidth * imageHeight * sizeof(unsigned int));
  cudaMalloc((void**)&devicehistogram, HISTOGRAM_LENGTH * sizeof(float));
  cudaMalloc((void**)&devicecdf, HISTOGRAM_LENGTH * sizeof(float));
  cudaMalloc((void**)&deviceblockSum, ceil(HISTOGRAM_LENGTH / (float)BLOCK_SIZE) * sizeof(unsigned int));

  // copy host memory to device here
  cudaMemcpy(deviceInputImageData, hostInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyHostToDevice);

  // invoke kernels
  dim3 dimGrid(ceil(imageWidth * imageHeight * imageChannels / (float)BLOCK_SIZE), 1, 1);
  dim3 dimBlock(BLOCK_SIZE, 1, 1);
  // convert to unsigned int
  convert_to_uint<<<dimGrid, dimBlock>>>(deviceInputImageData, deviceInputImageData_uint, imageWidth * imageHeight * imageChannels);

  // convert to grey image
  dim3 dimGrid2(ceil(imageWidth * imageHeight / (float)BLOCK_SIZE), 1, 1);
  dim3 dimBlock2(BLOCK_SIZE, 1, 1);
  convert_to_grey<<<dimGrid2, dimBlock2>>>(deviceInputImageData_uint, devicegreyImage, imageWidth * imageHeight);

  // compute histogram
  dim3 dimGrid3(ceil(imageWidth * imageHeight / (float)BLOCK_SIZE), 1, 1);
  dim3 dimBlock3(BLOCK_SIZE, 1, 1);
  cudaMemset(devicehistogram, 0, HISTOGRAM_LENGTH * sizeof(unsigned int));
  compute_histogram<<<dimGrid3, dimBlock3>>>(devicegreyImage, devicehistogram, imageWidth * imageHeight);

  // compute cdf
  dim3 dimGrid4(ceil(HISTOGRAM_LENGTH / (float)BLOCK_SIZE), 1, 1);
  dim3 dimBlock4(BLOCK_SIZE, 1, 1);
  cudaMemset(devicecdf, 0, HISTOGRAM_LENGTH * sizeof(float));
  scan<<<dimGrid4, dimBlock4>>>(devicehistogram, devicecdf, deviceblockSum, HISTOGRAM_LENGTH, imageHeight * imageWidth);
  cudaDeviceSynchronize();

  // compute correct color
  dim3 dimGrid5(ceil(imageChannels * imageWidth * imageHeight / (float)BLOCK_SIZE), 1, 1);
  dim3 dimBlock5(BLOCK_SIZE, 1, 1);
  correct_color<<<dimGrid5, dimBlock5>>>(deviceInputImageData_uint, devicecdf, imageWidth * imageHeight * imageChannels);

  // cast back to float
  dim3 dimGrid6(ceil(imageWidth * imageHeight * imageChannels / (float)BLOCK_SIZE), 1, 1);
  dim3 dimBlock6(BLOCK_SIZE, 1, 1);
  convert_to_float<<<dimGrid6, dimBlock6>>>(deviceInputImageData_uint, deviceOutputImageData, imageWidth * imageHeight * imageChannels);

  // copy device memory back to host here
  cudaMemcpy(hostOutputImageData, deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyDeviceToHost);

  // free device memory here
  cudaFree(deviceInputImageData);
  cudaFree(deviceOutputImageData);
  cudaFree(deviceInputImageData_uint);
  cudaFree(devicegreyImage);
  cudaFree(devicehistogram);
  cudaFree(devicecdf);
  cudaFree(deviceblockSum);
  
  wbSolution(args, outputImage);


  return 0;
}
