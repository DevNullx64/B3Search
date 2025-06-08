using ILGPU;
using ILGPU.IR;
using ILGPU.Runtime;
using System.Runtime.InteropServices.Marshalling;
using static System.Runtime.InteropServices.JavaScript.JSType;

namespace B3Search
{
    /// <summary>
    /// Implementation of Bounded Branchless Binary Search (B3Search) algorithm.
    /// </summary>
    public static class B3Search
    {
        /// <summary>
        /// Returns 1 if <paramref name="this"/> is less than <paramref name="other"/>, otherwise 0. Branchless.
        /// </summary>
        public static int IsLessThan(uint @this, uint other)
            => (int)((@this - other) >> 31);

        /// <summary>
        /// Returns the smaller of two signed integers, branchless.
        /// </summary>
        public static int Min(int a, int b)
        {
            // diff's MSB is 1 if a < b (underflow)
            int diff = a - b;
            // mask: 0xFFFFFFFF if a < b, else 0
            int mask = (diff >> 31);
            // returns a if a < b, else b
            return (a & mask) | (b & ~mask);
        }

        public static long Min(long a, long b)
        {
            // diff's MSB is 1 if a < b (underflow)
            long diff = a - b;
            // mask: 0xFFFFFFFFFFFFFFFF if a < b, else 0
            long mask = (diff >> 63);
            // returns a if a < b, else b
            return (a & mask) | (b & ~mask);
        }

        /// <summary>
        /// Branchless binary search: finds the first index where array[index] >= value.
        /// Returns last index if not found.
        /// </summary>
        public static int Search(uint[] array, uint value, byte iteration)
        {
            int begin = 0; // Start of the search range
            int end = array.Length; // End (exclusive) of the search range
            int lastIndex = end - 1; // Last valid index in the array

            for (byte i = 0; i < iteration; i++)
            {
                // Compute the middle index, clamped to lastIndex
                int mid = Min((begin + end) >> 1, lastIndex);

                // Check if array[mid] < value (1 if true, 0 if false)
                int isLess = IsLessThan(array[mid], value);

                // If array[mid] < value, move begin to mid + 1; else, keep begin
                begin += isLess * (mid + 1 - begin);

                // If array[mid] >= value, move end to mid; else, keep end
                end -= (1 - isLess) * (end - mid);
            }

            // Return the first index where array[index] >= value, or lastIndex if not found
            return Min(begin, lastIndex);
        }

        public static int Search(uint[] array, uint value)
        {
            if (array.Length == 0)
                throw new ArgumentException("Array must not be empty.");
            // Calculate the number of iterations needed based on the array length
            byte iteration = (byte)Math.Ceiling(Math.Log2(array.Length));
            return Search(array, value, iteration);
        }

        /// <summary>
        /// Branchless binary search: finds the first index where array[index] >= value.
        /// Returns last index if not found.
        /// </summary>
        /// <param name="idx">The index of the current thread. Or the index of the value to search from <paramref name="values"/>.</param>
        /// <param name="array">The array to search.</param>
        /// <param name="begin">The starting index of the search range.</param>
        /// <param name="end">The ending index (exclusive) of the search range.</param>
        /// <param name="values">The values to search for, one per thread.</param>
        /// <param name="iteration">The number of iterations to perform.</param>
        /// <returns>The first index where array[index] >= value, or lastIndex if not found.</returns>
        /// <remarks>
        /// This method uses <see cref="int"/> for indices and is suitable for arrays with a length up to 2^31-1.
        /// For larger arrays, use the overload that accepts <see cref="long"/> indices.
        /// </remarks>
        public static void SearchKernel(
            Index1D idx,
            ArrayView1D<uint, Stride1D.Dense> array,
            ArrayView1D<int, Stride1D.Dense> beginEnd,
            ArrayView1D<uint, Stride1D.Dense> values,
            SpecializedValue<byte> iteration,
            ArrayView1D<int, Stride1D.Dense> indices)
        {
            if(idx >= values.Length)
                return; // Out of bounds check

            int b = beginEnd[0]; // Start of the search range
            int e = beginEnd[1]; // End (exclusive) of the search range
            int lastIndex = e - 1; // Last valid index in the array
            uint value = values[idx]; // Value to search for

            for (byte i = 0; i < iteration; i++)
            {
                // Compute the middle index, clamped to lastIndex
                int mid = IntrinsicMath.Min((b + e) >> 1, lastIndex);
                // Check if array[mid] < value (1 if true, 0 if false)
                int isLess = IsLessThan(array[mid], value);
                // If array[mid] < value, move begin to mid + 1; else, keep begin
                b += isLess * (mid + 1 - b);
                // If array[mid] >= value, move end to mid; else, keep end
                e -= (1 - isLess) * (e - mid);
            }
            // Return the first index where array[index] >= value, or lastIndex if not found
            indices[idx] = IntrinsicMath.Min(b, lastIndex);
        }

        public static int[] GpuSearch(this Accelerator accelerator, uint[] array, uint[] values, out long internalTicks)
        {
            if (array.Length == 0 || values.Length == 0)
                throw new ArgumentException("Array and values must not be empty.");
            if (array.Length > int.MaxValue || values.Length > int.MaxValue)
                throw new ArgumentException("Array and values must not exceed int.MaxValue length.");

            // Allocate memory buffers on the GPU
            using MemoryBuffer1D<uint, Stride1D.Dense> arrayBuffer = accelerator.Allocate1D(array);
            using MemoryBuffer1D<int, Stride1D.Dense> beginEnd = accelerator.Allocate1D([0, array.Length]);
            using MemoryBuffer1D<uint, Stride1D.Dense> valuesBuffer = accelerator.Allocate1D(values);
            byte iteration = (byte)Math.Ceiling(Math.Log2(array.Length));
            using MemoryBuffer1D<int, Stride1D.Dense> indices = accelerator.Allocate1D<int>(values.Length);

            // Initialize begin and end buffers
            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<uint, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>,
                ArrayView1D<uint, Stride1D.Dense>,
                SpecializedValue<byte>,
                ArrayView1D<int, Stride1D.Dense>>(SearchKernel);

            // Launch the kernel to perform the search
            long startTicks = DateTimeOffset.UtcNow.Ticks;
            kernel(
                values.Length,
                arrayBuffer.View,
                beginEnd.View,
                valuesBuffer.View,
                new SpecializedValue<byte>(iteration),
                indices.View);
            accelerator.Synchronize();
            internalTicks = DateTimeOffset.UtcNow.Ticks - startTicks;

            // Copy the results from the GPU to the CPU
            int[] result = indices.GetAsArray1D();
            return result;
        }

        /// <summary>
        /// Branchless binary search: finds the first index where array[index] >= value.
        /// Returns last index if not found.
        /// </summary>
        /// <param name="array">The array to search.</param>
        /// <param name="begin">The starting index of the search range.</param>
        /// <param name="end">The ending index (exclusive) of the search range.</param>
        /// <param name="value">The value to search for.</param>
        /// <param name="iteration">The number of iterations to perform.</param>
        /// <returns>The first index where array[index] >= value, or lastIndex if not found.</returns>
        /// <remarks>
        /// This method uses <see cref="long"/> for indices and is suitable for arrays with a length up to 2^63-1.
        /// For smaller arrays, use the overload that accepts <see cref="int"/> indices.
        /// </remarks>
        public static long Search(ArrayView1D<uint, Stride1D.Dense> array, long begin, long end, uint value, SpecializedValue<byte> iteration)
        {
            long lastIndex = end - 1; // Last valid index in the array
            for (byte i = 0; i < iteration; i++)
            {
                // Compute the middle index, clamped to lastIndex
                long mid = Min((begin + end) >> 1, lastIndex);
                // Check if array[mid] < value (1 if true, 0 if false)
                int isLess = IsLessThan(array[mid], value);
                // If array[mid] < value, move begin to mid + 1; else, keep begin
                begin += isLess * (mid + 1 - begin);
                // If array[mid] >= value, move end to mid; else, keep end
                end -= (1 - isLess) * (end - mid);
            }
            // Return the first index where array[index] >= value, or lastIndex if not found
            return Min(begin, lastIndex);
        }
    }
}
