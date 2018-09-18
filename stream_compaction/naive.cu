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

#define blockSize 512

		void printArray(int n, int *a, bool abridged = false) {
			printf("    [ ");
			for (int i = 0; i < n; i++) {
				if (abridged && i + 2 == 15 && n > 16) {
					i = n - 2;
					printf("... ");
				}
				printf("%3d ", a[i]);
			}
			printf("]\n");
		}

		/**
		 * Kernel to perform a Naive scan on a integer array
		 */
		__global__ void kernScan(int n, int power, int* outputData, int* inputData)
        {
	        const int index = (blockIdx.x * blockDim.x) + threadIdx.x;

			// Fetch it once
			const auto curr_data = inputData[index];

			if(index >= power)
			{
				outputData[index] = inputData[index - power] + curr_data;
			}
			else
			{
				outputData[index] = curr_data;
			}
        }

		/**
		 * Shifts the whole array to the right by one in parallel
		 */
		__global__ void kernMakeExclusive(int n, int* outputData, int* inputData)
		{
			const int index = (blockIdx.x * blockDim.x) + threadIdx.x;

			outputData[index] = index != 0 ? inputData[index - 1] : 0;
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) 
    	{
			// 1. Allocate the memory in device
			cudaMalloc(reinterpret_cast<void**>(&device_iData), n * (sizeof(int)));
			cudaMalloc(reinterpret_cast<void**>(&device_oData), n * (sizeof(int)));
			cudaMemcpy(device_iData, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
			cudaDeviceSynchronize();

        	timer().startGpuTimer();

			// 2. Compute Block count
			dim3 num_blocks((n + blockSize - 1) / blockSize);
			
			// 3. Call the kernel
			const auto logn = ilog2ceil(n);
			for (auto i = 1; i <= logn; ++i)
			{
				const auto power = 1 << (i - 1);
				kernScan << < num_blocks, blockSize >> > (n, power, device_oData, device_iData);
				
				// Swap
				const auto temp = device_iData;
				device_iData = device_oData;
				device_oData = temp;
			}

			// Make it exclusive as we need that for stream compaction later on
			kernMakeExclusive <<< num_blocks, blockSize >> > (n, device_oData, device_iData);

			cudaDeviceSynchronize();
			timer().endGpuTimer();

			cudaMemcpy(odata, device_oData, sizeof(int) * n, cudaMemcpyDeviceToHost);

			// 4. Free up any gpu memory
			cudaFree(device_iData);
			cudaFree(device_oData);
        }
    }
}
