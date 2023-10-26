// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>
#include <stdio.h>

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

__global__ void scan(float *input, float *output, float* blockSum, int len) {
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
    XY[tx] = input[index];
  } else {
    XY[tx] = 0.0;
  }
  if (index + blockDim.x < len) {
    XY[tx + blockDim.x] = input[index + blockDim.x];
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


int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  float *blockSum;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&blockSum, ceil(numElements / ((float) BLOCK_SIZE * 2)) * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 DimGrid(ceil(numElements / (float)BLOCK_SIZE / 2), 1, 1);
  dim3 DimBlock(BLOCK_SIZE, 1, 1);
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  scan<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, blockSum, numElements);
  cudaDeviceSynchronize();
  // get the last element of each block, and put them into an array

  // add scanned block sum i to all values of scanned block i+1
  dim3 DimGrid2(ceil(numElements / (float)BLOCK_SIZE / 2), 1, 1);
  dim3 DimBlock2(BLOCK_SIZE * 2, 1, 1);
  add<<<DimGrid2, DimBlock2>>>(deviceOutput, blockSum, numElements);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}
