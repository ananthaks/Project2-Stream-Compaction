#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

		// Global variables
		int *device_iData;
		int *device_oData;

#define blockSize 128

		/**
		 * Kernel to perform a Naive scan on a integer array
		 */
		__global__ void kernScan(int n, int power, int* outputData, int* inputData)
        {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;

			// Fetch it once
			const int currData = inputData[index];

			if(index >= power)
			{
				outputData[index] = inputData[index - power] + currData;
			}
			else
			{
				outputData[index] = currData;
			}
        }

		/**
		 * Shifts the whole array to the right by one in parallel
		 */
		__global__ void kernMakeExclusive(int n, int* outputData, int* inputData)
		{
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;

			outputData[index] = index != 0 ? inputData[index - 1] : 0;
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) 
    	{
			// 1. Allocate the memory in device
			cudaMalloc((void**)&device_iData, n * (sizeof(int)));
			cudaMalloc((void**)&device_oData, n * (sizeof(int)));
			cudaMemcpy(device_iData, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
			cudaDeviceSynchronize();

        	timer().startGpuTimer();

			// 2. Compute Block count
			dim3 numBlocks((n + blockSize - 1) / blockSize);
			
			// 3. Call the kernel
			const int logn = ilog2ceil(n);
			for (int i = 1; i <= logn; ++i)
			{
				const int power = 1 << (i - 1);
				kernScan << < numBlocks, blockSize >> > (n, power, device_oData, device_iData);
				
				// Swap
				int* temp = device_iData;
				device_iData = device_oData;
				device_oData = temp;
			}

			// Make it exclusive as we need that for stream compaction later on
			kernMakeExclusive <<< numBlocks, blockSize >> > (n, device_oData, device_iData);

            timer().endGpuTimer();

			cudaDeviceSynchronize();
			cudaMemcpy(odata, device_oData, sizeof(int) * n, cudaMemcpyDeviceToHost);

			// 4. Free up any gpu memory
			cudaFree(device_iData);
			cudaFree(device_oData);
        }
    }
}
