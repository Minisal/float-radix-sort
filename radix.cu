#include <iostream>
#include <cuda_runtime.h>
#include <fstream>
#include <algorithm>
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <sys/time.h>
#include"device_launch_parameters.h" 

#define MAX_NUM_LISTS 256

using namespace std;
int num_data = 10000; // the number of the data
int num_lists = 128; // the number of parallel threads

unsigned GetTickCount()
{
    struct timeval tv;
    if(gettimeofday(&tv,NULL)!=0) return 0;
    return tv.tv_sec*1000+tv.tv_usec/1000;
}

__device__ void radix_sort(float* const data_0, float* const data_1, \
    int num_lists, int num_data, int tid); 
__device__ void merge_list(const float* src_data, float* const dst_list, \
    int num_lists, int num_data, int tid); 
__device__ void preprocess_float(float* const data, int num_lists, int num_data, int tid);
__device__ void Aeprocess_float(float* const data, int num_lists, int num_data, int tid);


// CPU implementation of Radix Sort
void radix_sort(float *a, int n) {
    float *t = (float *)malloc(sizeof(float)*n);
    int bitLength = sizeof(float) * 8;
    int count[2];
    int pref[2];
    int mask = 1;
    int negatives = 0, positives = 0;
    unsigned int b;
    union ufi {
     float f;
     int i;
    } u; 

    for (int shift = 0; shift < bitLength; shift++) {
    
        for (int i = 0; i < 2; i++)
            count[i] = 0;

        
        for (int i = 0; i < n; i++) {
         
            u.f = a[i];
            b = (u.i >> shift) & mask; 

            count[b]++;


            if (shift == 0 && a[i] < 0)
                negatives++;
        }
        if (shift == 0) positives = n - negatives;

        
        pref[0] = 0;
        for (int i = 1; i < 2; i++)
            pref[i] = pref[i - 1] + count[i - 1];

       
        for (int i = 0; i < n; i++) {
            
            u.f = a[i];
            b = (u.i >> shift) & mask;
         
            int index = pref[b]++;

            
            if (shift == bitLength - 1)
            {
                if (a[i] < 0)
                    index = positives - (index - negatives) - 1;
                else
                    index += negatives;
            }
            t[index] = a[i];
        }
        memcpy( a, t, sizeof(float) * n );
    }
    free(t);
}

// GPU implementation of Radix Sort
__global__ void GPU_radix_sort(float* const src_data, float* const dst_data, \
    int num_lists, int num_data) 
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    // special preprocessing of IEEE floating-point numbers before applying radix sort
    preprocess_float(src_data, num_lists, num_data, tid); 
    __syncthreads();
    // no shared memory
    radix_sort(src_data, dst_data, num_lists, num_data, tid);
    __syncthreads();
    merge_list(src_data, dst_data, num_lists, num_data, tid);    
    __syncthreads();
    Aeprocess_float(dst_data, num_lists, num_data, tid);
    __syncthreads();
}

/* 
 * Preprocessing floating point numbers
 *  In addition to the highest sign bit, the weight of IEEE float increases from 
 * lowest to highest, and can use the same sort method as the unsigned integer.
 * 
 * Preprocess for symbol bit
 *  - For positive floating point numbers : *data_temp>>31 & 0x1  ==  0
 *          Reverse the sign bit of the highest bit (0=>1) 
 *          (*data_temp) | 0x8000000
 *  - For negative floating point numbers : *data_temp>>31 & 0x1  ==  1
 *          Reverse all bits
 *          ~(*data_temp)
 * 
 *  After processing, the radix sorting method od integer can be applied.
*/
__device__ void preprocess_float(float* const src_data, int num_lists, int num_data, int tid)
{
    for(int i = tid;i<num_data;i+=num_lists)
    {
        unsigned int *data_temp = (unsigned int *)(&src_data[i]);    
        *data_temp = (*data_temp>>31 & 0x1)? ~(*data_temp): (*data_temp) | 0x80000000; 
    }
}

/*
 * Restore the sign bit of a floating-point number 
 */
__device__ void Aeprocess_float(float* const data, int num_lists, int num_data, int tid)
{
    for(int i = tid;i<num_data;i+=num_lists)
    {
        unsigned int* data_temp = (unsigned int *)(&data[i]);
        *data_temp = (*data_temp >> 31 & 0x1)? (*data_temp) & 0x7fffffff: ~(*data_temp);
    }
}


__device__ void radix_sort(float* const data_0, float* const data_1, \
    int num_lists, int num_data, int tid) 
{
    /*
        A loop of 32 bits in the outer layer. 
        Bacause the result of the previous iteration is the input of the next iteration,
        it cannot b parallelized.
    */
    for(int bit=0;bit<32;bit++)
    {
        int bit_mask = (1 << bit);
        int count_0 = 0;
        int count_1 = 0;
        for(int i=tid; i<num_data;i+=num_lists) 
        /* 
            The inner layer loops through all the data.
            Assuming that the number of threads is 128, let each thread sort the data of 
            its correspoding part i*num_lists.
            The data is divided into steps of num_lists, and an ordered lists of the number
            of threads is obtained.
        */
        {
            unsigned int *temp =(unsigned int *) &data_0[i];
            if(*temp & bit_mask) // The scanning bit is 1, which is divided into 1 list
            {
                // BUG : mandatory type conversion is performed
                data_1[tid+count_1*num_lists] = data_0[i];
                count_1 += 1;
            }
            else{ // The scanning bit is 0, which is divided into 0 list
                data_0[tid+count_0*num_lists] = data_0[i];
                count_0 += 1;
            }
        }
        for(int j=0;j<count_1;j++) // Put the 1 list after the 0 list
        {
            data_0[tid + count_0*num_lists + j*num_lists] = data_1[tid + j*num_lists]; 
        }
    }
}

/*  Parallel merge list
        After  parallel radix sorting, 128 sequential tables are obtained.
        First of all, find the minimum value in each list by parallel reduction.

    Parallel Reduction
        Using a thread with half the number of threads, the smaller value is transferred
        to the previous thread by comparing the corresponding element of the current thread
        with that of another thread.
        After each comparison, the number of active threads is reduced by half until only
        one thread is left.
        The element corresponding to the thread is the minimum value (tree structure 
        compares from bottom to top, conveying smaller values to root.)

*/
__device__ void merge_list(const float* src_data, float* const dst_list, \
    int num_lists, int num_data, int tid) 
{
    // The amount of data processed by each thread = the number of data in the ordered list
    int num_per_list = ceil((float)num_data/num_lists); 
    /* 
        To facilitate the communication between threads, shared memory is used.
        Shared memory records the minimun value of the current pointer position and pointer 
        of each other.
        Synchronize threads through syn()
        Finally, thread 0 writes the minimun value to the new list.

        MAX_NUM_LIST=256 : 128 threads are used to compare 256 data
    */
    // The current pointer position of each list
    __shared__ int list_index[MAX_NUM_LISTS]; 
    // The minimun value corresponding to the pointer
    __shared__ float record_val[MAX_NUM_LISTS]; 

   // ID of the thread corresponding to the minimun value source
    __shared__ int record_tid[MAX_NUM_LISTS]; 
    list_index[tid] = 0;
    record_val[tid] = 0;
    // The ID of the thread corresponding to the minimun source is initialized as the current thread
    record_tid[tid] = tid;
    __syncthreads();
    for(int i=0;i<num_data;i++)
    {
        record_val[tid] = 0;
        record_tid[tid] = tid; // BUG2 #TODO Initialization is required every time

        if(list_index[tid] < num_per_list) 
        { /* The pointer position in the list corresponding to the current thread
             is in the range (less than ther number of list data).
          */
            int src_index = tid + list_index[tid]*num_lists; // ？？tid*num_lists+list_index[tid]
            if(src_index < num_data)
            {
                // Read in the values in the corresponding list
                record_val[tid] = src_data[src_index]; // 
            }else{ 
                /*
                    The number used to fill in the last position of a list.
                    Set this number to the maximum so as not to affect 
                    finding the minimun value.
                */ 
                unsigned int *temp = (unsigned int *)&record_val[tid];
                *temp = 0xffffffff;
            }
        }else{ // All numbers in the current list have been written to the new list.
                unsigned int *temp = (unsigned int *)&record_val[tid];
                *temp = 0xffffffff;
        }
        __syncthreads();
        // Use half of the threads for comparison
        int tid_max = num_lists >> 1; 
        while(tid_max != 0)
        {/*
            For each list, compare the smallest value in each list.
            The number of threads used each time is halved until the lowest of the 
            minimum values in each list is found.
        */
            if(tid < tid_max)
            {/*
                With step_size of tid_max, compare the corresponding two data.
            */
                unsigned int *temp1 = (unsigned int*)&record_val[tid];
                unsigned int *temp2 = (unsigned int*)&record_val[tid+tid_max];
                // When the smaller value is put in front, that is, the position of temp1
                if(*temp2 < *temp1) // When temp2 is lower
                { 
                    // Update the value of the minimum value corresponding to the thread
                    record_val[tid] = record_val[tid + tid_max];
                    // Update the thread source of the minimum value corresponding to the thread
                    record_tid[tid] = record_tid[tid + tid_max];
                }
            }
            // After comparison, the number of threads used is halved.
            tid_max = tid_max >> 1;
            // Synchronization pretens confusion when reading data. 
            __syncthreads();  
        }
        if(tid == 0) 
        { // Find the minimun value and output it by thread 0
            // Pointer to the list +1
            list_index[record_tid[0]]++;
            // Writes the minimun value to the new list
            dst_list[i] = record_val[0]; 
        }
        __syncthreads();
    }
}


int main(void)
{
    float *data = new float[num_data], *result_CPU = new float[num_data], *result_GPU = new float[num_data];
    float  *src_data, *dst_data;
    for(int i =0;i<num_data;i++)
    {
        data[i] = (float)rand()/double(RAND_MAX);
    } 
    ofstream outfile("./data.bin", ios::out | ios::binary);
    outfile.write((char *)data, sizeof(float)*num_data);
    outfile.close();
    ifstream infile("./data.bin",ios::in | ios::binary);
    infile.read((char *)data, sizeof(float)*num_data);


    cout<<"number of data:"<<num_data<<endl;

    // test GPU sorting
    
    float elapsed = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMalloc((void**)&src_data, sizeof(float)*num_data);
    cudaMalloc((void**)&dst_data, sizeof(float)*num_data);
    cudaMemcpy(src_data, data, sizeof(float)*num_data, cudaMemcpyHostToDevice); 

    cudaEventRecord(start);
    GPU_radix_sort<<<1,num_lists>>>(src_data, dst_data, num_lists, num_data); 
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed,start,stop);

    cudaMemcpy(result_GPU, dst_data, sizeof(float)*num_data, cudaMemcpyDeviceToHost);
    

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cout<<"GPU SORT:"<<(unsigned)elapsed<<"ms"<<endl;
    //cout<<"data"<<endl;
    //for(int i=0; i<num_data; i++) cout<<data[i]<<" "; cout<<endl;
    //cout<<"result_GPU"<<endl;
    //for(int i=0; i<num_data; i++) cout<<result_GPU[i]<<" "; cout<<endl;


    //test CPU sorting
    unsigned start_t, end_t;

    start_t = GetTickCount();
    radix_sort(data,num_data);
    end_t = GetTickCount();
    cout<<"CPU SORT:"<<end_t-start_t<<"ms"<<endl;
    //cout<<"result_CPU"<<endl;
    //for(int i=0; i<num_data; i++) cout<<data[i]<<" "; cout<<endl;

    return 0;
}
