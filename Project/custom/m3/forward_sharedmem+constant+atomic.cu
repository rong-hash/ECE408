#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16

__constant__ float MASK[7000];

__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int B, const int M, const int C, const int H, const int W, const int K,const int S)
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

    int INPUT_TILE_WIDTH = S * TILE_WIDTH + K - S;

    extern __shared__ float sm[];

    #define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define mask_4d(i3, i2, i1, i0) MASK[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    #define sm_3d(i2, i1, i0) sm[(i2) * (INPUT_TILE_WIDTH * INPUT_TILE_WIDTH) + (i1) * (INPUT_TILE_WIDTH) + i0]
    #define sm_2d(i1, i0) sm[(i1) * (TILE_WIDTH) + i0]


    // Insert your GPU convolution kernel code here
    int W_grid = ceil((float)((W - K)/S + 1)/TILE_WIDTH);
    int H_grid = ceil((float)((H - K)/S + 1)/TILE_WIDTH);
    int b = blockIdx.x;
    int m = blockIdx.y;
    int h = (blockIdx.z / W_grid) * TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.z % W_grid) * TILE_WIDTH + threadIdx.x;
    int c = threadIdx.z;
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    float acc = 0.0f;

    int left_up_corner_h = ((blockIdx.z / W_grid) * TILE_WIDTH) * S;
    int left_up_corner_w = ((blockIdx.z % W_grid) * TILE_WIDTH) * S;
    // load the input tile
    for (int p = ty; p < INPUT_TILE_WIDTH; p += TILE_WIDTH) {
        for (int q = tx; q < INPUT_TILE_WIDTH; q += TILE_WIDTH) {
            int in_h = left_up_corner_h + p;
            int in_w = left_up_corner_w + q;
            if (in_h < H && in_w < W) {
                sm_3d(c, p, q) = in_4d(b, c, in_h, in_w);
            } else {
                sm_3d(c, p, q) = 0.0f;
            }
        }
    }

    __syncthreads();

    if (h < H_out && w < W_out){
        for (int p = 0; p < K; p++) {
            for (int q = 0; q < K; q++) {
                acc += mask_4d(m, c, p, q) * sm_3d(c, ty * S + p, tx * S + q);
            }
        }
        atomicAdd(&out_4d(b, m, h, w), acc);
    }


    #undef out_4d
    #undef in_4d
    #undef mask_4d
    #undef sm_3d
    #undef sm_2d
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

    cudaMalloc((void**)device_output_ptr, B * M * ((H - K)/S + 1) * ((W - K)/S + 1) * sizeof(float));
    cudaMalloc((void**)device_input_ptr, B * C * H * W * sizeof(float));
    cudaMalloc((void**)device_mask_ptr, M * C * K * K * sizeof(float));
    cudaMemcpy(*device_input_ptr, host_input, B * C * H * W * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(*device_mask_ptr, host_mask, M * C * K * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(MASK, host_mask, M * C * K * K * sizeof(float));

   
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Set the kernel dimensions and call the kernel
    int W_grid = ceil((float)((W - K)/S + 1)/TILE_WIDTH);
    int H_grid = ceil((float)((H - K)/S + 1)/TILE_WIDTH);
    int Y = W_grid * H_grid;
    int shared_mem_size = C * (S * TILE_WIDTH + K - S) * (S * TILE_WIDTH + K - S) * sizeof(float);
    dim3 dimgrid(B, M, Y);
    dim3 dimblock(TILE_WIDTH, TILE_WIDTH, C);
    conv_forward_kernel<<<dimgrid, dimblock, shared_mem_size>>>(device_output, device_input, device_mask, B, M, C, H, W, K, S);

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
