using Microsoft.ML.OnnxRuntime.Tensors;
using System.Runtime.InteropServices;

namespace ClipSharp;

public static class SingleExtensions
{
    private const ushort MidValue = ushort.MaxValue / 2;

    public static Float16 ToFloat16(this float value)
    {
        var half = (Half)value;
        Span<Half> halfs = stackalloc Half[] { half };
        Span<ushort> shorts = MemoryMarshal.Cast<Half, ushort>(halfs);
        return new Float16(shorts[0]);
    }
}
