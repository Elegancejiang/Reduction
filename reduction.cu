#include <stdio.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <cub/cub.cuh>
#include <sys/time.h>

double reduction1_time = 0;
struct timeval reduction1_begin;
struct timeval reduction1_end;

double reduction2_time = 0;
struct timeval reduction2_begin;
struct timeval reduction2_end;

double reduction3_time = 0;
struct timeval reduction3_begin;
struct timeval reduction3_end;

double reduction4_time = 0;
struct timeval reduction4_begin;
struct timeval reduction4_end;

double reduction5_time = 0;
struct timeval reduction5_begin;
struct timeval reduction5_end;

double reduction6_time = 0;
double reduction6_time_128 = 0;
double reduction6_time_256 = 0;
double reduction6_time_512 = 0;
double reduction6_time_1024 = 0;
struct timeval reduction6_begin;
struct timeval reduction6_end;

double reduction_trust_time = 0;
struct timeval reduction_trust_begin;
struct timeval reduction_trust_end;

double reduction_cub_time = 0;
struct timeval reduction_cub_begin;
struct timeval reduction_cub_end;

int length;

const int blockSize = 128;

__global__ void init_num(int *num, int length)
{
    int ii  = blockIdx.x * blockDim.x + threadIdx.x;

    if(ii < length) num[ii] = 1;
}

__global__ void reduction1(int *num, int length)
{
    int ii  = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    extern __shared__ int share_num[];

    if(ii < length) share_num[tid] = num[ii];
    else share_num[tid] = 0;

     __syncthreads();

    if(ii < length)
    {
        for(int i = 1;i < blockDim.x;i *= 2)
        {
            if(tid % (2 * i) == 0)
                share_num[tid] += share_num[tid + i];
            
            __syncthreads();
        }

        if(tid == 0) num[blockIdx.x] = share_num[0];
    }
}

__global__ void reduction2(int *num, int length)
{
    int ii  = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    extern __shared__ int share_num[];

    if(ii < length) share_num[tid] = num[ii];
    else share_num[tid] = 0;

     __syncthreads();

    if(ii < length)
    {
        for(int i = 1;i < blockDim.x;i *= 2)
        {
            int index = 2 * i * tid;

            if(index < blockDim.x)
                share_num[index] += share_num[index + i];
            
            __syncthreads();
        }

        if(tid == 0) num[blockIdx.x] = share_num[0];
    }
}

__global__ void reduction3(int *num, int length)
{
    int ii  = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    extern __shared__ int share_num[];

    if(ii < length) share_num[tid] = num[ii];
    else share_num[tid] = 0;

     __syncthreads();

    if(ii < length)
    {
        for(int i = blockDim.x / 2;i > 0;i >>= 1)
        {
            if(tid < i)
                share_num[tid] += share_num[tid + i];
            
            __syncthreads();
        }

        if(tid == 0) num[blockIdx.x] = share_num[0];
    }
}

__global__ void reduction4(int *num, int length)
{
    int ii  = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    int tid = threadIdx.x;
    
    extern __shared__ int share_num[];

    if(ii + blockDim.x < length) share_num[tid] = num[ii] + num[ii + blockDim.x];
    else if(ii < length) share_num[tid] = num[ii];
    else share_num[tid] = 0;

     __syncthreads();

    if(ii < length)
    {
        for(int i = blockDim.x / 2;i > 0;i >>= 1)
        {
            if(tid < i)
                share_num[tid] += share_num[tid + i];
            
            __syncthreads();
        }

        if(tid == 0) num[blockIdx.x] = share_num[0];
    }
}

__device__ void warpReduction5(volatile int *share_num, int tid)
{
    share_num[tid] += share_num[tid + 32];
    share_num[tid] += share_num[tid + 16];
    share_num[tid] += share_num[tid + 8];
    share_num[tid] += share_num[tid + 4];
    share_num[tid] += share_num[tid + 2];
    share_num[tid] += share_num[tid + 1];
}

__global__ void reduction5(int *num, int length)
{
    int ii  = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    int tid = threadIdx.x;
    
    extern __shared__ int share_num[];

    if(ii + blockDim.x < length) share_num[tid] = num[ii] + num[ii + blockDim.x];
    else if(ii < length) share_num[tid] = num[ii];
    else share_num[tid] = 0;

     __syncthreads();

    if(ii < length)
    {
        for(int i = blockDim.x / 2;i > 32;i >>= 1)
        {
            if(tid < i)
                share_num[tid] += share_num[tid + i];
            
            __syncthreads();
        }

        if(tid < 32) warpReduction5(share_num,tid);

        if(tid == 0) num[blockIdx.x] = share_num[0];
    }
}

__device__ void warpReduction6(volatile int *share_num, int tid ,int blocksize)
{
    if(blocksize >= 64) share_num[tid] += share_num[tid + 32];
    if(blocksize >= 32) share_num[tid] += share_num[tid + 16];
    if(blocksize >= 16) share_num[tid] += share_num[tid + 8];
    if(blocksize >= 8) share_num[tid] += share_num[tid + 4];
    if(blocksize >= 4) share_num[tid] += share_num[tid + 2];
    if(blocksize >= 2) share_num[tid] += share_num[tid + 1];
}

__global__ void reduction6(int *num, int length)
{
    int ii  = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    int tid = threadIdx.x;
    
    extern __shared__ int share_num[];

    if(ii + blockDim.x < length) share_num[tid] = num[ii] + num[ii + blockDim.x];
    else if(ii < length) share_num[tid] = num[ii];
    else share_num[tid] = 0;

     __syncthreads();

    if(ii < length)
    {
        if(blockDim.x >= 512) 
        {
            if(tid < 256) share_num[tid] += share_num[tid + 256];
            __syncthreads();
        }
        if(blockDim.x >= 256) 
        {
            if(tid < 128) share_num[tid] += share_num[tid + 128];
            __syncthreads();
        }
        if(blockDim.x >= 128) 
        {
            if(tid < 64) share_num[tid] += share_num[tid + 64];
            __syncthreads();
        }

        if(tid < 32) warpReduction6(share_num, tid, blockDim.x);

        if(tid == 0) num[blockIdx.x] = share_num[0];
    }
}

void print_num(int *num, int length)
{
    // for(int i = 0;i < 2048;i++)
    // {
    //     printf("%d ",num[i]);
    //     if(i % 32 == 31) printf("\n");
    // }
    printf("%d\n",num[0]);
}

int main(int argc, char **argv)
{
    cudaSetDevice(0);

    length = atoi(argv[1]);

    int *num, *cuda_num;

    num = (int *)malloc(length * sizeof(int));

    cudaMalloc((void**)&cuda_num,length * sizeof(int));

    //reduction1
    init_num<<<(length + 128 - 1) / 128,128>>>(cuda_num,length);

    cudaDeviceSynchronize();
    gettimeofday(&reduction1_begin,NULL);

    for(int l = length;l != 1;l = (l + 128 - 1) / 128)
        reduction1<<<(l + 128 - 1) / 128,128>>>(cuda_num,l);

    cudaDeviceSynchronize();
    gettimeofday(&reduction1_end,NULL);
    reduction1_time += (reduction1_end.tv_sec - reduction1_begin.tv_sec) * 1000 + (reduction1_end.tv_usec - reduction1_begin.tv_usec) / 1000.0;

    // exam
    cudaMemcpy(&num[0], &cuda_num[0],sizeof(int),cudaMemcpyDeviceToHost);
    printf("reduction1    :");
    print_num(num, length);

    //reduction2
    init_num<<<(length + 128 - 1) / 128,128>>>(cuda_num,length);

    cudaDeviceSynchronize();
    gettimeofday(&reduction2_begin,NULL);

    for(int l = length;l != 1;l = (l + 128 - 1) / 128)
        reduction2<<<(l + 128 - 1) / 128,128>>>(cuda_num,l);

    cudaDeviceSynchronize();
    gettimeofday(&reduction2_end,NULL);
    reduction2_time += (reduction2_end.tv_sec - reduction2_begin.tv_sec) * 1000 + (reduction2_end.tv_usec - reduction2_begin.tv_usec) / 1000.0;

    // exam
    cudaMemcpy(&num[0], &cuda_num[0],sizeof(int),cudaMemcpyDeviceToHost);
    printf("reduction2    :");
    print_num(num, length);

    //reduction3
    init_num<<<(length + 128 - 1) / 128,128>>>(cuda_num,length);

    cudaDeviceSynchronize();
    gettimeofday(&reduction3_begin,NULL);

    for(int l = length;l != 1;l = (l + 128 - 1) / 128)
        reduction3<<<(l + 128 - 1) / 128,128>>>(cuda_num,l);

    cudaDeviceSynchronize();
    gettimeofday(&reduction3_end,NULL);
    reduction3_time += (reduction3_end.tv_sec - reduction3_begin.tv_sec) * 1000 + (reduction3_end.tv_usec - reduction3_begin.tv_usec) / 1000.0;

    // exam
    cudaMemcpy(&num[0], &cuda_num[0],sizeof(int),cudaMemcpyDeviceToHost);
    printf("reduction3    :");
    print_num(num, length);

    //reduction4
    init_num<<<(length + 128 - 1) / 128,128>>>(cuda_num,length);

    cudaDeviceSynchronize();
    gettimeofday(&reduction4_begin,NULL);

    for(int l = length;l != 1;l = (l + 256 - 1) / 256)
        reduction4<<<(l + 256 - 1) / 256,128>>>(cuda_num,l);

    cudaDeviceSynchronize();
    gettimeofday(&reduction4_end,NULL);
    reduction4_time += (reduction4_end.tv_sec - reduction4_begin.tv_sec) * 1000 + (reduction4_end.tv_usec - reduction4_begin.tv_usec) / 1000.0;

    // exam
    cudaMemcpy(&num[0], &cuda_num[0],sizeof(int),cudaMemcpyDeviceToHost);
    printf("reduction4    :");
    print_num(num, length);

    //reduction5
    init_num<<<(length + 128 - 1) / 128,128>>>(cuda_num,length);

    cudaDeviceSynchronize();
    gettimeofday(&reduction5_begin,NULL);

    for(int l = length;l != 1;l = (l + 256 - 1) / 256)
        reduction5<<<(l + 256 - 1) / 256,128>>>(cuda_num,l);

    cudaDeviceSynchronize();
    gettimeofday(&reduction5_end,NULL);
    reduction5_time += (reduction5_end.tv_sec - reduction5_begin.tv_sec) * 1000 + (reduction5_end.tv_usec - reduction5_begin.tv_usec) / 1000.0;

    // exam
    cudaMemcpy(&num[0], &cuda_num[0],sizeof(int),cudaMemcpyDeviceToHost);
    printf("reduction5    :");
    print_num(num, length);

    //reduction6
    init_num<<<(length + 128 - 1) / 128,128>>>(cuda_num,length);

    cudaDeviceSynchronize();
    gettimeofday(&reduction6_begin,NULL);

    for(int l = length;l != 1;l = (l + 256 - 1) / 256)
        reduction6<<<(l + 256 - 1) / 256,128>>>(cuda_num,l);

    cudaDeviceSynchronize();
    gettimeofday(&reduction6_end,NULL);
    reduction6_time_128 += (reduction6_end.tv_sec - reduction6_begin.tv_sec) * 1000 + (reduction6_end.tv_usec - reduction6_begin.tv_usec) / 1000.0;

    // exam
    num[0] = 0;
    cudaMemcpy(&num[0], &cuda_num[0],sizeof(int),cudaMemcpyDeviceToHost);
    printf("reduction6_128:");
    print_num(num, length);

    init_num<<<(length + 128 - 1) / 128,128>>>(cuda_num,length);

    cudaDeviceSynchronize();
    gettimeofday(&reduction6_begin,NULL);

    for(int l = length;l != 1;l = (l + 512 - 1) / 512)
        reduction6<<<(l + 512 - 1) / 512,256>>>(cuda_num,l);

    cudaDeviceSynchronize();
    gettimeofday(&reduction6_end,NULL);
    reduction6_time_256 += (reduction6_end.tv_sec - reduction6_begin.tv_sec) * 1000 + (reduction6_end.tv_usec - reduction6_begin.tv_usec) / 1000.0;

    // exam
    num[0] = 0;
    cudaMemcpy(&num[0], &cuda_num[0],sizeof(int),cudaMemcpyDeviceToHost);
    printf("reduction6_256:");
    print_num(num, length);

    //thrust
    init_num<<<(length + 128 - 1) / 128,128>>>(cuda_num,length);

    cudaDeviceSynchronize();
    gettimeofday(&reduction_trust_begin,NULL);

    int thrust_sum = thrust::reduce(thrust::device, cuda_num, cuda_num + length);

    cudaDeviceSynchronize();
    gettimeofday(&reduction_trust_end,NULL);
    reduction_trust_time += (reduction_trust_end.tv_sec - reduction_trust_begin.tv_sec) * 1000 + (reduction_trust_end.tv_usec - reduction_trust_begin.tv_usec) / 1000.0;

    // exam
    printf("thrust        :");
    printf("%d\n",thrust_sum);

    //cub
    int *cuda_num_out;
    cudaMalloc((void**)&cuda_num_out,length * sizeof(int));

    init_num<<<(length + 128 - 1) / 128,128>>>(cuda_num,length);

    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;

    cudaDeviceSynchronize();
    gettimeofday(&reduction_cub_begin,NULL);

    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, cuda_num, cuda_num_out, length);
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // Run sum-reduction
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, cuda_num, cuda_num_out, length);

    cudaDeviceSynchronize();
    gettimeofday(&reduction_cub_end,NULL);
    reduction_cub_time += (reduction_cub_end.tv_sec - reduction_cub_begin.tv_sec) * 1000 + (reduction_cub_end.tv_usec - reduction_cub_begin.tv_usec) / 1000.0;

    // exam
    cudaMemcpy(&num[0],&cuda_num_out[0],sizeof(int),cudaMemcpyDeviceToHost);
    printf("cub           :");
    print_num(num, length);

    printf("\nThe Time:\n");
    printf("    the reduction1 time:      %.3lf ms\n", reduction1_time);
    printf("    the reduction2 time:      %.3lf ms\n", reduction2_time);
    printf("    the reduction3 time:      %.3lf ms\n", reduction3_time);
    printf("    the reduction4 time:      %.3lf ms\n", reduction4_time);
    printf("    the reduction5 time:      %.3lf ms\n", reduction5_time);
    printf("    the reduction6 time 128:  %.3lf ms\n", reduction6_time_128);
    printf("    the reduction6 time 256:  %.3lf ms\n", reduction6_time_256);

    printf("    the trust time:           %.3lf ms\n", reduction_trust_time);
    printf("    the cub time:             %.3lf ms\n", reduction_cub_time);

    cudaFree(cuda_num);
    cudaFree(cuda_num_out);
    free(num);

    return 0;
}