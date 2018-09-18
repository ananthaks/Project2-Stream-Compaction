#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {

        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
	        static PerformanceTimer timer;
	        return timer;
        }

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
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();

			odata[0] = 0;

			for(int i = 1; i < n; ++i)
			{
				odata[i] = odata[i - 1] + idata[i - 1];
			}

	        timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();

			int outIndex = 0;
			for (int i = 0; i < n; ++i)
			{
				if (idata[i] != 0)
				{
					odata[outIndex++] = idata[i];
				}
			}

	        timer().endCpuTimer();
            return (outIndex);
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {

			// Keeping allocations outside of timer
			auto *tempValidator = new int[n];
			auto *scanArray = new int[n];

	        timer().startCpuTimer();

			// 1. Compute temporary array
			for (int i = 0; i < n; ++i)
			{
				tempValidator[i] = (idata[i] != 0 ? 1 : 0);
			}

			// 2. Perform exclusive scan
			scanArray[0] = 0;
			for (int i = 1; i < n; ++i)
			{
				scanArray[i] = scanArray[i - 1] + tempValidator[i - 1];
			}


			// 3. Scatter
			int outIndex = 0;
			for(int i = 0; i < n; ++i)
			{
				if(tempValidator[i] != 0)
				{
					outIndex = scanArray[i];
					odata[outIndex] = idata[i];
				}
			}

	        timer().endCpuTimer();

			delete[] tempValidator;
			delete[] scanArray;

            return (outIndex + 1);
        }
    }
}
