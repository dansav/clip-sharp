using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace ClipSharp.Example.Wpf
{
    public partial class MainViewModel : ObservableObject
    {
        private static readonly HashSet<string> _knownImageExtensions = new HashSet<string>() { ".jpg", ".jpeg", ".png" };

        [ObservableProperty]
        private IReadOnlyCollection<string>? _imageFiles;

        [ObservableProperty]
        private string? _selectedImage;

        [ObservableProperty]
        private string? _descriptions;

        [ObservableProperty]
        private string? _matches;

        private string? _textFile;

        private TextualModel? _textModel;
        private VisualModel? _visualModel;

        private (string First, float[] Second)[]? _descriptionEmbeddings;

        public void Initialize()
        {
            var textTokenizer = SimpleTextTokenizer.Load();
            _textModel = TextualModel.Load("clip-vit-base-patch16-textual-float16.onnx", textTokenizer);
            _visualModel = VisualModel.Load("clip-vit-base-patch16-visual-float16.onnx");
        }

        public void HandleFileDrop(string[] files)
        {
            var dir = new DirectoryInfo(files[0]);

            if (dir.Exists)
            {
                ImageFiles = Directory.GetFiles(dir.FullName, "*", SearchOption.AllDirectories)
                    .Where(p => _knownImageExtensions.Contains(Path.GetExtension(p).ToLowerInvariant()))
                    .ToArray();

                return;
            }

            var textFile = new FileInfo(files[0]);
            if (textFile.Exists)
            {
                _textFile = textFile.FullName;
                LoadText();
            }
        }

        partial void OnDescriptionsChanged(string? oldValue, string? newValue)
        {
            if (newValue is null) return;

            var descriptions = newValue.Split(Environment.NewLine)
                .Select(line => line.Trim())
                .Where(line => line.Length > 0)
                .ToArray();

            var textTokenizer = SimpleTextTokenizer.Load();
            var textModel = TextualModel.Load("clip-vit-base-patch16-textual-float16.onnx", textTokenizer);
            var embeddings = textModel.Encode(descriptions);
            _descriptionEmbeddings = descriptions
                .Zip(embeddings)
                .ToArray();

            Debug.WriteLine(_descriptionEmbeddings.Length);
        }

        partial void OnSelectedImageChanged(string? oldValue, string? newValue)
        {
            if (_visualModel is null) return;
            if (_descriptionEmbeddings is null) return;
            if (newValue is null) return;

            var imageEmbeddings = _visualModel.Encode(new[] { newValue }).First();

            var similarities = new List<(string Text, float Similarity)>();

            foreach (var (text, te) in _descriptionEmbeddings)
            {
                var similarity = CosineSimilarity(imageEmbeddings, te);
                similarities.Add((text, similarity));
            }

            var sorted = similarities
                .OrderByDescending(s => s.Similarity)
                .Select(s => $"{s.Similarity:0.0000}\t{s.Text}");

            Matches = string.Join(Environment.NewLine, sorted);
        }

        [RelayCommand]
        private void Run()
        {

        }

        private void LoadText()
        {
            if (_textFile != null)
            {
                var text = File.ReadAllText(_textFile);
                Descriptions = text;
            }
        }

        private float CosineSimilarity(float[] a, float[] b)
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
    }
}
