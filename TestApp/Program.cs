using B3Search;
using ILGPU;
using ILGPU.Runtime;
using System.Globalization;

internal class Program
{
    private static readonly NumberFormatInfo nfi = new() { NumberGroupSeparator = "'" };

    private static void Main(string[] args)
    {
        Random rnd = new();
        
        // Build a sorted big memory buffer
        uint[] array = new uint[16 * 1024 * 1024];
        Console.WriteLine($"Building a sorted {array.Length / 1024 / 1024}MB memory buffer...");
        uint baseValue = 0;
        for (int i = 0; i < array.Length; i++)
        {
            baseValue += (uint)rnd.Next(0, int.MaxValue / array.Length);
            array[i] = baseValue;
        }

        Context devices = Context.CreateDefault();
        Device device = devices.GetPreferredDevice(false);
        using Accelerator accelerator = device.CreateAccelerator(devices);
        Console.WriteLine($"Using device: {device.Name}");

        // Prepare the values to search for
        uint[] values = new uint[device.MaxNumThreads];
        values[0] = array[0];
        values[1] = array[^1];
        uint min = values[0];
        uint max = values[1] + 1;
        for (int i = 2; i < values.Length; i++)
        {
            // Generate random values within the range of the array
            values[i] = (uint)rnd.Next((int)min, (int)max);
        }

        // Perform a standard BSearch using the GPU
        Console.Write("\n1. First BSearch run on GPU (need compilation)...");
        GpuBSearchTest(accelerator, device, array, values);
        Console.Write("2. Second BSearch run on GPU (should be faster)...");
        GpuBSearchTest(accelerator, device, array, values);
        Console.Write("3. Third B3Search run on GPU (should be faster)...");
        GpuBSearchTest(accelerator, device, array, values);

        // Perform the search using the GPU
        Console.Write("\n1. First B3Search run on GPU (need compilation)...");
        GpuTest(accelerator, device, array, values);
        Console.Write("2. Second B3Search run on GPU (should be faster)...");
        GpuTest(accelerator, device, array, values);
        Console.Write("3. Third B3Search run on GPU (should be faster)...");
        GpuTest(accelerator, device, array, values);

        // Perform the search using the CPU
        Console.Write("\n1. First B3Search run on CPU...");
        CpuTest(array, values);
        Console.Write("2. Second B3Search run on CPU...");
        CpuTest(array, values);
        Console.Write("3. Third B3Search run on CPU...");
        CpuTest(array, values);

        // Perform the search using the native binary search
        Console.Write("\n1. First BSearch run using native binary search...");
        NativeBSearch(array, values);
        Console.Write("2. Second BSearch run using native binary search...");
        NativeBSearch(array, values);
        Console.Write("3. Third BSearch run using native binary search...");
        NativeBSearch(array, values);
    }

    public static void GpuTest(Accelerator accelerator, Device device, uint[] array, uint[] values)
    {
        // Perform the search on the GPU
        Console.WriteLine($"\nB3Search: Searching {values.Length} elements using the {device.Name}...");
        long startTime = DateTimeOffset.UtcNow.Ticks;
        int[] indices = accelerator.GpuSearch(array, values, out long internalTicks);
        long endTime = DateTimeOffset.UtcNow.Ticks;
        long elapsedMicroseconds = (endTime - startTime) / TimeSpan.TicksPerMicrosecond;
        internalTicks /= TimeSpan.TicksPerMicrosecond; // Convert internal ticks to microseconds
        Console.WriteLine($"Search completed in (total) {elapsedMicroseconds.ToString("N0", nfi)} µs.");
        Console.WriteLine($"Search completed in (internal) {internalTicks.ToString("N0", nfi)} µs.");
    }

    public static void GpuBSearchTest(Accelerator accelerator, Device device, uint[] array, uint[] values)
    {
        // Perform the search on the GPU using the B3Search kernel
        Console.WriteLine($"\nB3Search: Searching {values.Length} elements using the {device.Name} with B3Search kernel...");
        long startTime = DateTimeOffset.UtcNow.Ticks;
        int[] indices = GpuBSearch(accelerator, array, values, out long internalTicks);
        long endTime = DateTimeOffset.UtcNow.Ticks;
        long elapsedMicroseconds = (endTime - startTime) / TimeSpan.TicksPerMicrosecond;
        internalTicks /= TimeSpan.TicksPerMicrosecond; // Convert internal ticks to microseconds
        Console.WriteLine($"Search completed in (total) {elapsedMicroseconds.ToString("N0", nfi)} µs.");
        Console.WriteLine($"Search completed in (internal) {internalTicks.ToString("N0", nfi)} µs.");
    }


    public static void CpuTest(uint[] array, uint[] values)
    {
        Console.WriteLine($"\nB3Search: Searching {values.Length} elements using the CPU...");
        long startTime = DateTimeOffset.UtcNow.Ticks;
        int[] indices = new int[values.Length];
        for (int i = 0; i < values.Length; i++)
        {
            indices[i] = B3Search.B3Search.Search(array, values[i]);
        }
        long endTime = DateTimeOffset.UtcNow.Ticks;
        long elapsedMicroseconds = (endTime - startTime) / TimeSpan.TicksPerMicrosecond;
        Console.WriteLine($"Search completed in {elapsedMicroseconds.ToString("N0", nfi)} µs.");
    }

    public static void NativeBSearch(uint[] array, uint[] values)
    {
        Console.WriteLine($"\nB3Search: Searching {values.Length} elements using the native binary search...");
        long startTime = DateTimeOffset.UtcNow.Ticks;
        int[] indices = new int[values.Length];
        for (int i = 0; i < values.Length; i++)
        {
            indices[i] = Array.BinarySearch(array, values[i]);
        }
        long endTime = DateTimeOffset.UtcNow.Ticks;
        long elapsedMicroseconds = (endTime - startTime) / TimeSpan.TicksPerMicrosecond;
        Console.WriteLine($"Search completed in {elapsedMicroseconds.ToString("N0", nfi)} µs.");
    }

    /// <summary>
    /// A standar BSearch kernel for ILGPU that performs a binary search on a sorted array.
    /// </summary>
    /// <param name="idx">The index of the current thread.</param>
    /// <param name="array">The sorted array to search in.</param>
    /// <param name="beginEnd">An array containing the begin and end indices of the search range.</param>
    /// <param name="values">The values to search for in the array.</param>
    /// <param name="iteration">The iteration count, used to determine the number of search iterations.</param>
    /// <param name="indices">An array to store the indices of the found values.</param>
    public static void BSearchKernel(
    Index1D idx,
    ArrayView1D<uint, Stride1D.Dense> array,
    ArrayView1D<int, Stride1D.Dense> beginEnd,
    ArrayView1D<uint, Stride1D.Dense> values,
    SpecializedValue<byte> iteration,
    ArrayView1D<int, Stride1D.Dense> indices)
    {
        if (idx >= values.Length)
            return; // Out of bounds check

        int b = beginEnd[0]; // Start of the search range
        int e = beginEnd[1]; // End (exclusive) of the search range
        int lastIndex = e - 1; // Last valid index in the array
        uint value = values[idx]; // Value to search for

        // Perform binary search iterations
        for (byte i = 0; i < iteration.Value; i++)
        {
            if (b > lastIndex)
            {
                indices[idx] = lastIndex; // Value not found
                return;
            }
            int m = b + (e - b) / 2; // Calculate the middle index
            uint midValue = array[m]; // Get the value at the middle index
            if (midValue == value)
            {
                indices[idx] = m; // Value found, store the index
                return;
            }
            else if (midValue < value)
            {
                b = m + 1; // Search in the right half
            }
            else
            {
                e = m; // Search in the left half
            }
        }
        indices[idx] = IntrinsicMath.Min(b, lastIndex); // If not found, return the last valid index
    }

    public static int[] GpuBSearch(Accelerator accelerator, uint[] array, uint[] values, out long internalTicks)
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
            ArrayView1D<int, Stride1D.Dense>>(BSearchKernel);

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

}