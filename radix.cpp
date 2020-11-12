#include<iostream>
#include<math.h>
#include<cuda_runtime.h>
#include"device_launch_parameters.h" 
#include<fstream>
#define MAX_NUM_LISTS 256

using namespace std;
int num_data = 1000; // the number of the data
int num_lists = 128; // the number of parallel threads

__device__ void radix_sort(float* const data_0, float* const data_1, \
    int num_lists, int num_data, int tid); 
__device__ void merge_list(const float* src_data, float* const dest_list, \
    int num_lists, int num_data, int tid); 
__device__ void preprocess_float(float* const data, int num_lists, int num_data, int tid);
__device__ void Aeprocess_float(float* const data, int num_lists, int num_data, int tid);


//基数排序的GPU实现
__global__ void GPU_radix_sort(float* const src_data, float* const dest_data, \
    int num_lists, int num_data) 
{
    // temp_data:temporarily store the data
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    // special preprocessing of IEEE floating-point numbers before applying radix sort
    preprocess_float(src_data, num_lists, num_data, tid); 
    __syncthreads();
    // no shared memory？？？
    radix_sort(src_data, dest_data, num_lists, num_data, tid);
    __syncthreads();
    merge_list(src_data, dest_data, num_lists, num_data, tid);    
    __syncthreads();
    Aeprocess_float(dest_data, num_lists, num_data, tid);
    __syncthreads();
}

/* 对浮点数进行预处理
IEEE float 除了最高的符号位，其他位置从小到大对数值的权重依次增加，可与无符号整数使用相同的排序方法
针对符号位可做预处理
    对于正浮点数 *data_temp >> 31 & 0x1 == 0
        将最高位的符号位取反（0=>1)
            (*data_temp) | 0x80000000
    对于负浮点数 *data_temp>>31 & 0x1 == 1
        全部位均取反
            ~(*data_temp)
处理后可应用整数的基数排序方法
*/
__device__ void preprocess_float(float* const src_data, int num_lists, int num_data, int tid)
{
    for(int i = tid;i<num_data;i+=num_lists)
    {
        unsigned int *data_temp = (unsigned int *)(&src_data[i]);    
        *data_temp = (*data_temp>>31 & 0x1)? ~(*data_temp): (*data_temp) | 0x80000000; 
    }
}

/* 恢复浮点数的符号位
*/
__device__ void Aeprocess_float(float* const data, int num_lists, int num_data, int tid)
{
    for(int i = tid;i<num_data;i+=num_lists)
    {
        unsigned int* data_temp = (unsigned int *)(&data[i]);
        *data_temp = (*data_temp >> 31 & 0x1)? (*data_temp) & 0x7fffffff: ~(*data_temp);
    }
}

/*

*/

__device__ void radix_sort(float* const data_0, float* const data_1, \
    int num_lists, int num_data, int tid) 
{
    for(int bit=0;bit<32;bit++) // 外层32个比特位的循环。因为上一迭代的结果为下一迭代的输入，不能并行。
    {
        int bit_mask = (1 << bit);
        int count_0 = 0;
        int count_1 = 0;
        for(int i=tid; i<num_data;i+=num_lists) // 内层遍历所有数据的循环。
            // 假设线程数量为128，让每一个线程对自身对应部分的数据i*num_lists进行基数排序
            // 数据以num_lists的步长进行划分，最后得到线程数量个有序的列表
        {
            unsigned int *temp =(unsigned int *) &data_0[i];
            if(*temp & bit_mask)//扫描比特位为1，分入1列表
            {
                data_1[tid+count_1*num_lists] = data_0[i]; //BUG 等于时会做强制类型转化
                count_1 += 1;
            }
            else{//扫描比特位为0，分入0列表
                data_0[tid+count_0*num_lists] = data_0[i];
                count_0 += 1;
            }
        }
        for(int j=0;j<count_1;j++) // 将1列表置于0列表后
        {
            data_0[tid + count_0*num_lists + j*num_lists] = data_1[tid + j*num_lists]; 
        }
    }
}

/* 并行合并列表
经过并行基数排序，得到线程数目（128）个有序列表
首先通过并行归约？？的方式找到各列表中的最小值
并行归约
    使用线程数量一半的线程，通过比较当前线程对应元素与另一线程对应元素，将较小值转移到前面的线程
    每次比较后活跃的线程数量减少一半，直到只剩一个线程为止
    该线程对应的元素即为最小值
    （树状结构自底向上进行比较，输送较小值至根部）
*/
__device__ void merge_list(const float* src_data, float* const dest_list, \
    int num_lists, int num_data, int tid) 
{
    int num_per_list = ceil((float)num_data/num_lists); // 每个线程处理的数据量=有序列表中数据的数量
    /* 
    为了方便各线程通信，使用共享内存
    共享内存记录各线程当前的指针位置与指针对应的最小值
    通过sys同步各个线程
    最后由0号线程向新列表中写入最小值
    */
    //MAX_NUM_LIST=256,比较256个数使用128个线程
    __shared__ int list_index[MAX_NUM_LISTS]; // 各个列表当前指针所处的位置
    __shared__ float record_val[MAX_NUM_LISTS]; // 指针对应的最小值
    __shared__ int record_tid[MAX_NUM_LISTS]; //最小值来源对应的线程的id
    list_index[tid] = 0;
    record_val[tid] = 0;
    record_tid[tid] = tid; // 最小值来源对应的线程的id，初始化为当前线程
    __syncthreads();
    for(int i=0;i<num_data;i++)
    {
        record_val[tid] = 0;
        record_tid[tid] = tid; // bug2 每次都要进行初始化？？？
        if(list_index[tid] < num_per_list) // 当前线程对应的列表中的指针位置在范围内（小于列表数据数量）
        {
            int src_index = tid + list_index[tid]*num_lists; // ？？tid*num_lists+list_index[tid]
            if(src_index < num_data)
            {
                record_val[tid] = src_data[src_index]; // 读入对应列表中的值
            }else{ // 最后一个列表中填位的数，将数置为最大不影响找最小值
                unsigned int *temp = (unsigned int *)&record_val[tid];
                *temp = 0xffffffff;
            }
        }else{ //当前列表中的所有数都已经被写入新列表
                unsigned int *temp = (unsigned int *)&record_val[tid];
                *temp = 0xffffffff;
        }
        __syncthreads();
        int tid_max = num_lists >> 1; // 使用一半的线程进行比较
        while(tid_max != 0) // 对于每个列表，比较各列表中最小的值。每次使用的线程减半，直到找到各列表最小值中的最小值
        {
            if(tid < tid_max)
            {
                // 以tid_max的步长，比较对应的两个数据
                unsigned int *temp1 = (unsigned int*)&record_val[tid];
                unsigned int *temp2 = (unsigned int*)&record_val[tid+tid_max];
                // 当较小值放在前面，即temp1的位置
                if(*temp2 < *temp1) //当temp2的值更小的时候
                { 
                    record_val[tid] = record_val[tid + tid_max];//更新线程对应的最小值的值
                    record_tid[tid] = record_tid[tid + tid_max];//更新线程对应的最小值的线程来源
                }
            }
            tid_max = tid_max >> 1; // 比较后，使用的线程数量减半
            __syncthreads(); // 同步防止读取数据时混乱
        }
        if(tid == 0) // 找到最小值，由0号线程进行输出
        {
            list_index[record_tid[0]]++;// 最小值来源的列表的指针+1
            dest_list[i] = record_val[0]; // 向新列表中写入最小值
        }
        __syncthreads();
    }
}


int main(void)
{
    float *data = new float[num_data];
    float  *src_data, *dest_data;
    for(int i =0;i<num_data;i++)
    {
        data[i] = (float)rand()/double(RAND_MAX);
    } 
    ofstream outfile("./data.bin", ios::out | ios::binary);
    outfile.write((char *)data, sizeof(float)*num_data);
    outfile.close();
    ifstream infile("./data.bin",ios::in | ios::binary);
    infile.read((char *)data, sizeof(float)*num_data);

    cudaMalloc((void**)&src_data, sizeof(float)*num_data);
    cudaMalloc((void**)&dest_data, sizeof(float)*num_data);
    cudaMemcpy(src_data, data, sizeof(float)*num_data, cudaMemcpyHostToDevice);

    GPU_radix_sort<<<1,num_lists>>>(src_data, dest_data, num_lists, num_data);
    cudaMemcpy(data, dest_data, sizeof(float)*num_data, cudaMemcpyDeviceToHost);
    
    for(int i =0;i<num_data;i++)
    {
        cout<<data[i]<<" ";
    }
    return 0;
}
