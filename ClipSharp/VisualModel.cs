using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace ClipSharp;

public class VisualModel
{
    private readonly InferenceSession _session;
    private readonly int _inputSize;
    private readonly string _inputName;
    private readonly string _outputName;

    public static VisualModel Load(string modelPath)
    {
        var options = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
            // EnableMemoryPattern = false,
            // LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING,

        };

        options.AppendExecutionProvider_CUDA();
        options.AppendExecutionProvider_CPU();
        // options.RegisterOrtExtensions();

        var session = new InferenceSession(modelPath, options);

        return new VisualModel(session);
    }

    public VisualModel(InferenceSession session)
    {
        _session = session;

        var input = _session.InputMetadata.First();
        if (input.Value.Dimensions.Length != 4 || input.Value.Dimensions[2] != input.Value.Dimensions[3])
        {
            throw new ArgumentException($"Unexpected input dimensions (expected height and witdth to be equal)");
        }

        _inputSize = input.Value.Dimensions[2];
        _inputName = input.Key;

        var output = _session.OutputMetadata.First();
        _outputName = output.Key;
    }


    public IReadOnlyCollection<float[]> Encode(string[] images)
    {
        var imgFs = images.Select(img => ImageToVector(img)).ToArray();

        Memory<Float16> tokens = imgFs.SelectMany(l => l.ToFlatArray()).ToArray();
        var inputTensor = new DenseTensor<Float16>(tokens, new[] { images.Length, 3, 224, 224 });

        using var results = _session.Run(new[] { NamedOnnxValue.CreateFromTensor(_inputName, inputTensor) });

        using var result = results.First();
        var embeddings = (DenseTensor<Float16>)result.Value;

        var output = new float[embeddings.Dimensions[0]][];
        for (int i = 0; i < embeddings.Dimensions[0]; i++)
        {
            output[i] = new float[embeddings.Dimensions[1]];
            for (int j = 0; j < embeddings.Dimensions[1]; j++)
            {
                output[i][j] = embeddings[i, j].ToSingle();
            }
        }

        return output;
    }

    private static float[,,] ImageToVector(string imagePath)
    {
        var image = Image.Load<Rgba32>(imagePath);
        var orgWidth = image.Width;
        var orgHeight = image.Height;

        // resize
        // center crop
        image.Mutate(x =>
        {
            var min = Math.Min(orgWidth, orgHeight);
            x.Crop(new Rectangle((orgWidth - min) / 2, (orgHeight - min) / 2, min, min));
            x.Resize(224, 224);
        });

        // convert rgb
        // to float
        return ToNormalizedColorHeightWidthArray(image);
    }

    public static float[,,] ToNormalizedColorHeightWidthArray(Image<Rgba32> image)
    {
        //var img = new NDArray<float>(new Shape(3, image.Height, image.Width));
        var img = new float[3, image.Height, image.Width];

        for (var y = 0; y < image.Height; y++)
        {
            for (var x = 0; x < image.Width; x++)
            {
                var p = image[x, y];
                //img[0, y, x] = (p.R - 127) / 128f;
                //img[1, y, x] = (p.G - 127) / 128f;
                //img[2, y, x] = (p.B - 127) / 128f;

                img[0, y, x] = (p.R / 255f - 0.48145466f) / 0.26862954f;
                img[1, y, x] = (p.G / 255f - 0.4578275f) / 0.26130258f;
                img[2, y, x] = (p.B / 255f - 0.40821073f) / 0.27577711f;
            }
        }

        return img;
    }
}
