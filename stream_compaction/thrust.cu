#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "common.h"
#include "thrust.h"

namespace StreamCompaction {
    namespace Thrust {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

		thrust::device_ptr<int> dev_thrustInputData;
		thrust::device_ptr<int> dev_thrustOutputData;
		int* device_iData;
		int* device_oData;

#define blockSize 512

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) 
    	{
			const int numTotalBytes = n * sizeof(int);

			cudaMalloc(reinterpret_cast<void**>(&device_iData), numTotalBytes);
			cudaMalloc(reinterpret_cast<void**>(&device_oData), numTotalBytes);

			cudaMemcpy(device_iData, idata, numTotalBytes, cudaMemcpyHostToDevice);
			cudaDeviceSynchronize();

			dev_thrustInputData = thrust::device_ptr<int>(device_iData);
			dev_thrustOutputData = thrust::device_ptr<int>(device_oData);

			timer().startGpuTimer();
			
			thrust::exclusive_scan(dev_thrustInputData, n + dev_thrustInputData, dev_thrustOutputData);

			timer().endGpuTimer();

			cudaDeviceSynchronize();
			cudaMemcpy(odata, device_oData, numTotalBytes, cudaMemcpyDeviceToHost);

			cudaFree(device_iData);
			cudaFree(device_oData);
        }
    }
}
