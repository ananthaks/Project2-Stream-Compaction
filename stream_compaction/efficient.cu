#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

		// Global variables
		int* device_iData;
		int* device_oData;
		int* device_bools;
		int* device_sortedbools;

#define blockSize 128

		void printArray(int n, const int *a, bool abridged = false) {
			
			if("    [ ");
			for (int i = 0; i < n; i++) {
				if (abridged && i + 2 == 15 && n > 16) {
					i = n - 2;
					printf("... ");
				}
				printf("%3d ", a[i]);
			}
			printf("]\n");
		}


		void PrintCUDAArray(int n, const int* cudaArrayPtr)
		{
			cudaDeviceSynchronize();

			int* temp = new int[n];
			cudaMemcpy(temp, cudaArrayPtr, n * sizeof(n), cudaMemcpyDeviceToHost);

			printArray(n, temp, true);

			delete[] temp;
		}



		/**
		* Kernel to perform a Work efficient scan on a integer array
		*/
		__global__ void kernUpSweep(int n, int two_d, int* outputData)
		{
			const int index = (blockIdx.x * blockDim.x) + threadIdx.x;

			const int two_d_1 = two_d * 2;

			if(index % two_d_1 != 0)
			{
				return;
			}
						
			const int oldIndex = index + two_d - 1;
			const int newIndex = index + two_d_1 - 1;

			const int currData = outputData[newIndex];

			outputData[newIndex] = newIndex != (n - 1) ? currData + outputData[oldIndex] : 0;
		}

		/**
		* Kernel to perform a Work efficient scan on a integer array
		*/
		__global__ void kernDownSweep(int n, int two_d, int* outputData)
		{
			const int index = (blockIdx.x * blockDim.x) + threadIdx.x;

			const int two_d_1 = two_d * 2;

			if (index % two_d_1 != 0)
			{
				return;
			}

			const int oldIndex = index + two_d - 1;
			const int newIndex = index + two_d_1 - 1;

			const int dataAtNewIndex = outputData[newIndex];

			const int t = outputData[oldIndex];
			outputData[oldIndex] = dataAtNewIndex;
			outputData[newIndex] = t + outputData[newIndex];
		}
		
		/**
		* Shifts the whole array to the right by one in parallel
		*/
		__global__ void kernMakeExclusive(int n, int* outputData, int* inputData)
		{
			const int index = (blockIdx.x * blockDim.x) + threadIdx.x;

			outputData[index] = index != 0 ? inputData[index - 1] : 0;
		}


		inline int RoundToPower2(int n)
		{
			int start = 2;

			while(start <= n)
			{
				start *= 2;
			}
			return start;
		}


        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) 
    	{
			const int numElements = RoundToPower2(n);
			const int numTotalBytes = numElements * sizeof(int);
			const int numActualBytes = n * sizeof(int);

			// 1. Allocate the memory in device
			cudaMalloc(reinterpret_cast<void**>(&device_oData), numTotalBytes);
			cudaMemcpy(device_oData, idata, numActualBytes, cudaMemcpyHostToDevice);

			cudaDeviceSynchronize();

			timer().startGpuTimer();

			// 2. Compute Block count
			dim3 num_blocks((numElements + blockSize - 1) / blockSize);

			// 3. Call the kernel
			const int log_n = ilog2ceil(numElements);
			// 3a. UpSweep
			int power_2 = 1;
			for(int d = 0; d < log_n; ++d)
			{
				power_2 = (1 << d);
				kernUpSweep << < num_blocks, blockSize >> > (numElements, power_2, device_oData);
			}
			
			// 3b. DownSweep
			for (int d = log_n - 1; d >= 0; --d)
			{
				power_2 = (1 << d);
				kernDownSweep << < num_blocks, blockSize >> > (numElements, power_2, device_oData);
			}

			timer().endGpuTimer();

			cudaDeviceSynchronize();
			cudaMemcpy(odata, device_oData, numActualBytes, cudaMemcpyDeviceToHost);

			// 4. Free up any gpu memory
			cudaFree(device_iData);
			cudaFree(device_oData);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) 
    	{
			const int numElements = RoundToPower2(n);
			const int numTotalBytes = numElements * sizeof(int);
			const int numActualBytes = n * sizeof(int);

			int* tempHolder = new int[n];

			printArray(n, idata, true);

			// 1. Allocate the memory in device
			cudaMalloc(reinterpret_cast<void**>(&device_iData), numTotalBytes);
			cudaMalloc(reinterpret_cast<void**>(&device_oData), numTotalBytes);
			cudaMalloc(reinterpret_cast<void**>(&device_bools), numTotalBytes);
			cudaMalloc(reinterpret_cast<void**>(&device_sortedbools), numTotalBytes);

			cudaMemcpy(device_iData, idata, numActualBytes, cudaMemcpyHostToDevice);

			cudaDeviceSynchronize();

			timer().startGpuTimer();

			// 2. Compute Block count
			dim3 num_blocks((numElements + blockSize - 1) / blockSize);

			// 3. Call the kernel

			// 3a. Map to bools
			StreamCompaction::Common::kernMapToBoolean << <num_blocks, blockSize >> > (numElements, device_bools, device_iData);
			cudaMemcpy(device_sortedbools, device_bools, numActualBytes, cudaMemcpyHostToHost);

			// 3b. UpSweep
			const int log_n = ilog2ceil(numElements);
			int power_2 = 1;
			for (int d = 0; d < log_n; ++d)
			{
				power_2 = (1 << d);
				kernUpSweep << < num_blocks, blockSize >> > (numElements, power_2, device_sortedbools);
			}
			
			// 3c. DownSweep
			for (int d = log_n - 1; d >= 0; --d)
			{
				power_2 = (1 << d);
				kernDownSweep << < num_blocks, blockSize >> > (numElements, power_2, device_sortedbools);
			}

			// 3d. Compact
			StreamCompaction::Common::kernScatter << <num_blocks, blockSize >> > (numElements, device_oData, device_iData, device_bools, device_sortedbools);

			// 4. Manually copy from the GPU the bools and check the number of valida values to return
			cudaDeviceSynchronize();
			cudaMemcpy(tempHolder, device_bools, numActualBytes, cudaMemcpyDeviceToHost);
			int count = 0;
			for(int i = 0; i < n; ++i)
			{
				if(tempHolder[i] != 0)
				{
					count++;
				}
			}

			timer().endGpuTimer();

			// Copy over the results
			cudaDeviceSynchronize();
			cudaMemcpy(odata, device_oData, numActualBytes, cudaMemcpyDeviceToHost);

			// 5. Free up any gpu memory
			cudaFree(device_iData);
			cudaFree(device_oData);
			cudaFree(device_bools);
			cudaFree(device_sortedbools);
			delete[] tempHolder;

            return (count);
        }
    }
}
