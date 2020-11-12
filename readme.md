# parallel radix sort with CUDA

Radix sort is an algorithm with a fixed iteration time that sorts the values by comparing the lowest bit to the highest bit. The paralle implementation of radix sort mainly includes three algotithms: parallel radix sort, parallel merge, and parallel reduction.

## Radix Sort

The basic idea of radix sort is to start from the lowest bit, sort according to the value of the current bit to get a new sequence, and then reorder the new sequence according to the value of the higher bit, thus looping to the highest bit. 

## Radix Sort of Float

For IEEE 754, the storage format is as shown in the figure below. The highest bit represents the sign of the number, 8 bits represent the exponent, and 23 bits represent the mantissa.

```
single precision : 
					|31|30 ··· 23|22 ··· 0|
					|s |   exp   |  frac  |
```

IEEE float has a feature that expect fot the highest sign bit, the weight of the value increases from 0 to 30. These bits are sorted in the same way as 32-bit unsigned integers after preprocessing. 
The preprocessing of the sign bit is as follows: 
 - Positive float: The highest sign bit is inverted (from 0 to 1).
 - Negative float: All bits are inverted, so the radix sort of integers can be applied.
 After converting the floating-point number to an integers format, performing bit operations on it for sorting, and then converting after the sorting is completed.


```
unsigned int *data_temp = (unsigned int *)(&src_data[i]);    
*data_temp = (*data_temp >> 31 & 0x1)? ~(*data_temp): (*data_temp) | 0x80000000; 
```
## Parallel Radix Sort

Radix sort scans all the data to be sorted in each iteration, and divides the it into a 0-list and a 1-list according to the position of the scan bit, which are 0 or 1, respectively. For simplicity, 2N memory array is used, one stores the 0-list, and the other one stores the 1-list. After put the 1-list after 0-list, starts the next iteration.


## Parallel Merge Lists
After parallel radix sort, we get ordered lists with the number of threads (128). For the serial merge method, we select the smallest number from each ordered list and insert it into a new list each time. Therefore, we also need an array list_index to record the current pointer position of each list.

Compared with serial merging, parallel merging first finds the minimum value in each list through parallel reduction. Parallel reduction uses half of the number of threads to compare the element of the current thread with element of corresponding thread, and transfer the smaller value to the previous thread. After each comparison, the number of active threads is reduced by half until only one thread remains, and the element corresponding to this thread is the minimun value.
In order to facilitate communication between threads, shared memory is used to record the current pointer position of each thread and the minimum value corresponding to the pointer. Finally, the thread with thread number 0 writes the minimum value to the new list.


# Project

> Compiling method : 
> nvcc  radix.cu  -o  radix.exe

```
__device__  radix_sort()           // Parallel radix sort
__device__  merge_list()           // Parallel merge lists
__device__  preprocess_float()     // Preprocessing of the sign bit of float
__device__  Aeprocess_float()      // Restore the sign bit of float
__global__  GPU_radix_sort()       // GPU implementation of radix sort
```



