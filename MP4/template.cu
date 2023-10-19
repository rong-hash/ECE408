#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
// define the constant memory of kernel


#define TILE_SIZE 4
#define MASK_WIDTH 3
__constant__ float kernel[3][3][3];
__constant__ int maskwidth = 3;


//@@ Define constant memory for device kernel here

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;
  int z = threadIdx.z + blockDim.z * blockIdx.z;
  int index = x + y * x_size + z * x_size * y_size;
  float result = 0.0;
  __shared__ float N_ds[TILE_SIZE][TILE_SIZE][TILE_SIZE];
  N_ds[threadIdx.z][threadIdx.y][threadIdx.x] = input[index];
  __syncthreads();
  int radius = maskwidth/2;
  int this_tile_start_x = blockIdx.x * TILE_SIZE;
  int this_tile_start_y = blockIdx.y * TILE_SIZE;
  int this_tile_start_z = blockIdx.z * TILE_SIZE;
  int next_tile_start_x = this_tile_start_x + TILE_SIZE;
  int next_tile_start_y = this_tile_start_y + TILE_SIZE;
  int next_tile_start_z = this_tile_start_z + TILE_SIZE;
  int N_start_x = x - radius;
  int N_start_y = y - radius;
  int N_start_z = z - radius;
  for (int i = 0; i < MASK_WIDTH; i++){
    for (int j = 0; j < MASK_WIDTH; j++){
      for (int k = 0; k < MASK_WIDTH; k++){
        int N_index_x = N_start_x + k;
        int N_index_y = N_start_y + j;
        int N_index_z = N_start_z + i;
        if (N_index_x >= 0 && N_index_x < x_size && N_index_y >= 0 && N_index_y < y_size && N_index_z >= 0 && N_index_z < z_size){
          if (N_index_x >= this_tile_start_x && N_index_x < next_tile_start_x && N_index_y >= this_tile_start_y && N_index_y < next_tile_start_y && N_index_z >= this_tile_start_z && N_index_z < next_tile_start_z){
            result += N_ds[N_index_z - this_tile_start_z][N_index_y - this_tile_start_y][N_index_x - this_tile_start_x] * kernel[i][j][k];
          }
          else{
            result += input[N_index_x + N_index_y * x_size + N_index_z * x_size * y_size] * kernel[i][j][k];
          }
        }
      }
    }
  }
  if (x < x_size && y < y_size && z < z_size)
    output[index] = result;
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  wbTime_stop(GPU, "Doing GPU memory allocation");
  cudaMalloc((void **)&deviceInput, (inputLength - 3) * sizeof(float));
  cudaMalloc((void **)&deviceOutput, (inputLength - 3) * sizeof(float));
  cudaMemcpyToSymbol(kernel, hostKernel, kernelLength * sizeof(float));
  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  wbTime_stop(Copy, "Copying data to the GPU");
  cudaMemcpy(deviceInput, hostInput + 3, (inputLength - 3) * sizeof(float), cudaMemcpyHostToDevice);
  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here
  dim3 DimGrid(ceil(x_size/4.0), ceil(y_size/4.0), ceil(z_size/4.0));
  dim3 DimBlock(4, 4, 4);
  //@@ Launch the GPU kernel here
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");
  conv3d<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, z_size, y_size, x_size);
  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  wbTime_stop(Copy, "Copying data from the GPU");
  cudaMemcpy(hostOutput + 3, deviceOutput, (inputLength - 3) * sizeof(float), cudaMemcpyDeviceToHost);
  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}