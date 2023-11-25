#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 20

__constant__ float MASK[7000];

// Compute C = A * B
__global__ void matrixMultiplyShared(const float *A, const float *B, float *C,
    int numARows, int numAColumns,
    int numBRows, int numBColumns,
    int numCRows, int numCColumns) {
    //@@ Insert code to implement matrix multiplication here
    //@@ You have to use shared memory for this MP
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];
    int imageidx = blockIdx.x;
    int bx = blockIdx.y;
    int by = blockIdx.z;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;
    float Cvalue = 0.0;
    for (int m = 0; m < (numAColumns - 1) / TILE_WIDTH + 1; ++m) {
        if (m * TILE_WIDTH + ty < numBRows && Col < numBColumns) {
            ds_B[ty][tx] = B[imageidx * (numBRows * numBColumns) + (m * TILE_WIDTH + ty) * numBColumns + Col];
        } else {
            ds_B[ty][tx] = 0.0;
        }
        __syncthreads();
        for (int k = 0; k < TILE_WIDTH; ++k) {
            Cvalue += MASK[Row * numAColumns + m * TILE_WIDTH + k] * ds_B[k][tx];
        }
        __syncthreads();
    }
    if (Row < numCRows && Col < numCColumns) {
        C[imageidx * numCRows * numCColumns + Row * numCColumns + Col] = Cvalue;
    }
}


void unroll(int B, int C, int H, int W, int K, int S, const float* input, float* unrolled_input) {
    #define unrolled_3d(i2, i1, i0) unrolled_input[(i2) * (C * K * K * ((H - K)/S + 1) * ((W - K)/S + 1)) + (i1) * (((H - K)/S + 1) * ((W - K)/S + 1)) + i0]
    #define input_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    int H_out = (H - K) / S + 1;
    int W_out = (W - K) / S + 1;
    for (int b = 0; b < B; b++) {
        for (int c = 0; c < C; c++) {
            int w_base = c * K * K;
            for (int p = 0; p < K; p++) {
                for (int q = 0; q < K; q++) {
                    for (int h = 0; h < H_out; h++) {
                        for (int w = 0; w < W_out; w++) {
                            int h_unroll = w_base + p * K + q;
                            int w_unroll = h * W_out + w;
                            unrolled_3d(b, h_unroll, w_unroll) = input_4d(b, c, h * S + p, w * S + q);
                        }
                    }
                }
            }
        }
    }
    #undef unrolled_3d
    #undef input_4d
}



	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }
    float* unrolled_host_input = (float*)malloc(B * C * K * K * ((H - K)/S + 1) * ((W - K)/S + 1) * sizeof(float));
    unroll(B, C, H, W, K, S, host_input, unrolled_host_input);
    
    cudaMalloc((void**)device_output_ptr, B * M * ((H - K)/S + 1) * ((W - K)/S + 1) * sizeof(float));
    cudaMalloc((void**)device_input_ptr, B * C * K * K * ((H - K)/S + 1) * ((W - K)/S + 1) * sizeof(float));
    cudaMemcpy(*device_input_ptr, unrolled_host_input, B * C * K * K * ((H - K)/S + 1) * ((W - K)/S + 1) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(MASK, host_mask, M * C * K * K * sizeof(float));
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Set the kernel dimensions and call the kernel
    dim3 dimGrid(B, ceil(((H - K)/S + 1) * ((W - K)/S + 1) / (float) TILE_WIDTH), ceil(M / (float) TILE_WIDTH));
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    matrixMultiplyShared<<<dimGrid, dimBlock>>>(device_mask, device_input, device_output, M, C * K * K, C * K * K, ((H - K)/S + 1) * ((W - K)/S + 1), M, ((H - K)/S + 1) * ((W - K)/S + 1));

}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Copy the output back to host
    cudaMemcpy(host_output, device_output, B * M * ((W - K)/S + 1) * ((W - K)/S + 1) * sizeof(float), cudaMemcpyDeviceToHost);
   
    // Free device memory
    cudaFree(device_output);
    cudaFree(device_input);

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
