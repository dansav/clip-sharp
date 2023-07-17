// See https://aka.ms/new-console-template for more information
using ClipSharp;

Console.WriteLine("Hello, World!");

var textTokenizer = SimpleTextTokenizer.Load();

var textModel = TextualModel.Load("clip-vit-base-patch16-textual-float16.onnx", textTokenizer);

var texts = new[] {
    "a close up photo of a cherry blossom",
    "cherry blossom",
    "flowers",
    "plant",
    "processing plant",
    "a large industrial plant with many pipes, walkways and railings",
    "ruhrgebiet",
    "industry",
    "a photo taken on a bright and sunny day",
    "a photo taken on a dark and cloudy day",
    "a photo taken at midnight",
    "bees",
    "cars",
    "dogs and cats",
};

var textEmbeddings = textModel.Encode(texts);

Console.WriteLine(texts.Length);
Console.WriteLine(textEmbeddings.Count);

var images = new[] {
    "flowers.jpg",
    "heavy-industry.jpg",
};


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

    var cosineSimilarity = dotProduct / (float)(Math.Sqrt(normA) * Math.Sqrt(normB));
    

    return cosineSimilarity;
}