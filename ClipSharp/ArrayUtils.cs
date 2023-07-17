using Microsoft.ML.OnnxRuntime.Tensors;

namespace ClipSharp;

public static class ArrayUtils
{
    public static Float16[] ToFlatArray(this float[,,] input)
    {
        var I = input.GetLength(0);
        var J = input.GetLength(1);
        var K = input.GetLength(2);

        var z = new Float16[I * J * K];
        var x = 0;
        for (var i = 0; i < I; i++)
        {
            for (var j = 0; j < J; j++)
            {
                for (var k = 0; k < K; k++)
                {
                    z[x++] = input[i, j, k].ToFloat16();
                }
            }
        }

        return z;
    }

    public static T[] ToFlatArray<T>(this T[,,] input)
    {
        var I = input.GetLength(0);
        var J = input.GetLength(1);
        var K = input.GetLength(2);

        var z = new T[I * J * K];
        var x = 0;
        for (var i = 0; i < I; i++)
        {
            for (var j = 0; j < J; j++)
            {
                for (var k = 0; k < K; k++)
                {
                    z[x++] = input[i, j, k];
                }
            }
        }

        return z;
    }
}
