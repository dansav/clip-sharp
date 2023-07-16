namespace ClipSharp;

public interface ITextTokenizer
{
    int SotToken { get; }

    int EotToken { get; }

    IReadOnlyCollection<int> Encode(string input);

    string Decode(IEnumerable<int> tokens);
}
