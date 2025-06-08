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
        uint[] array = new uint[1 * 1024 * 1024];
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
        uint[] values = new uint[1280];
        values[0] = array[0];
        values[1] = array[^1];
        uint min = values[0];
        uint max = values[1] + 1;
        for (int i = 2; i < values.Length; i++)
        {
            // Generate random values within the range of the array
            values[i] = (uint)rnd.Next((int)min, (int)max);
        }

        // Perform the search using the GPU
        Console.Write("\n1. First run on GPU (need compilation)...");
        GpuTest(accelerator, device, array, values);

        Console.Write("2. Second run on GPU (should be faster)...");
        GpuTest(accelerator, device, array, values);

        // Perform the search using the CPU
        Console.Write("\n1. First run on CPU...");
        CpuTest(array, values);
        Console.Write("2. Second run on CPU...");
        CpuTest(array, values);

        // Perform the search using the native binary search
        Console.Write("\n1. First run using native binary search...");
        NativeBSearch(array, values);
        Console.Write("2. Second run using native binary search...");
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
}