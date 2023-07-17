using Microsoft.ML.OnnxRuntime.Tensors;
using System.Runtime.InteropServices;

namespace ClipSharp;

public static class Float16Extensions
{

    public static float ToSingle(this Float16 value)
    {
        var uint16 = (ushort)value;
        Span<ushort> shorts = stackalloc ushort[] { uint16 };
        Span<Half> halfs = MemoryMarshal.Cast<ushort, Half>(shorts);
        return (float)halfs[0];
    }
}
