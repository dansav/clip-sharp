// See https://aka.ms/new-console-template for more information
using ClipSharp;
using System.CommandLine;

Console.WriteLine("Hello, World!");

var rootCommand = new RootCommand("C# CLIP example application.");

var textOption = new Option<FileInfo?>("--text", "The path to a text file where every line is a 'prompt'");
var imagesOption = new Option<DirectoryInfo?>("--images", "The path to a directory where images, that should be described, are store");

rootCommand.AddOption(textOption);
rootCommand.AddOption(imagesOption);
rootCommand.SetHandler(Start, textOption, imagesOption);

rootCommand.Invoke(args);

void Start(FileInfo? file, DirectoryInfo? imageDir)
{
    if (file is null || !file.Exists) file = new FileInfo("text.txt");

    var texts = File.ReadAllLines(file.FullName);

    string[] images;
    if (imageDir?.Exists == true)
    {
        images = Directory.GetFiles(imageDir.FullName, "*", SearchOption.AllDirectories);
    }
    else
    {
        images = new[]
        {
            "images/flowers.jpg",
            "images/heavy-industry.jpg",
        };
    }

    var textTokenizer = SimpleTextTokenizer.Load();

    var textModel = TextualModel.Load("clip-vit-base-patch16-textual-float16.onnx", textTokenizer);

    var textEmbeddings = textModel.Encode(texts);

    Console.WriteLine(texts.Length);
    Console.WriteLine(textEmbeddings.Count);

    var visualModel = VisualModel.Load("clip-vit-base-patch16-visual-float16.onnx");

    var imageEmbeddings = visualModel.Encode(images);
    Console.WriteLine($"Embeddings shape: {imageEmbeddings.Count}");

    Console.WriteLine();
    foreach (var (image, ie) in images.Zip(imageEmbeddings))
    {
        var similarities = new List<(string Text, float Similarity)>();

        foreach (var (text, te) in texts.Zip(textEmbeddings))
        {
            var similarity = CosineSimilarity(ie, te);
            similarities.Add((text, similarity));
        }

        var sorted = similarities.OrderByDescending(s => s.Similarity);
        Console.WriteLine($"Image: {image}");
        foreach (var similarity in sorted)
        {
            Console.WriteLine($"{similarity.Similarity:0.0000}\t{similarity.Text}");
        }

        Console.WriteLine();
    }
}

// def cosine_similarity(a, b):
//     return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
float CosineSimilarity(float[] a, float[] b)
{
    float dotProduct = 0;
    float normA = 0;
    float normB = 0;

    for (int i = 0; i < a.Length; i++)
    {
        dotProduct += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }

    var cosineSimilarity = dotProduct / (MathF.Sqrt(normA) * MathF.Sqrt(normB));
    return cosineSimilarity;
}