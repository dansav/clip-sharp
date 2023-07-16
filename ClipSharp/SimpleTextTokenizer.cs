using System.IO.Compression;
using System.Text;
using System.Text.RegularExpressions;

namespace ClipSharp;

public partial class SimpleTextTokenizer : ITextTokenizer
{
    public const string StartOfText = "<|startoftext|>";
    public const string EndOfText = "<|endoftext|>";

    [GeneratedRegex("""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", RegexOptions.IgnoreCase | RegexOptions.Compiled)]
    private static partial Regex BpePattern();

    [GeneratedRegex("""\s+""", RegexOptions.Compiled)]
    private static partial Regex Whitespace();


    private readonly Dictionary<byte, char> _byteEncoder;
    private readonly Dictionary<char, byte> _byteDecoder;
    private readonly Dictionary<string, float> _bpeRanks;
    private readonly Dictionary<string, int> _vocabEncoder;
    private readonly Dictionary<int, string> _vocabDecoder;

    private Dictionary<string, string> _bpeCache;

    public static SimpleTextTokenizer Load()
    {
        // generate vocab encoder
        var merges = new List<string>();
        using (var fileStream = new FileStream("./bpe_simple_vocab_16e6.txt.gz", FileMode.Open))
        using (var textStream = new GZipStream(fileStream, CompressionMode.Decompress))
        {
            var reader = new StreamReader(textStream);

            reader.ReadLine(); // ignore first line
            string? line;
            while ((line = reader.ReadLine()) is not null && merges.Count < 49152 - 256 - 2) // not sure why the file contains extra data...
            {
                if (line.Length == 0) continue;

                var bpe = line.Split(' ');
                if (bpe.Length != 2) continue;

                merges.Add(line);
            }
        }

        // generate byte encoder
        var (byteEncoder, byteDecoder) = BytesToUnicode();
        var vocab = byteEncoder.Values.Select(c => new string(c, 1)).ToList();
        vocab.AddRange(byteEncoder.Values.Select(c => $"{c}</w>"));

        int index = 0;
        var bpeRanks = new Dictionary<string, float>();

        // append merges
        foreach (var merge in merges)
        {            
            vocab.Add(merge.Replace(" ", ""));

            bpeRanks.Add(merge, index);
            index++;
        }
        vocab.AddRange(new[] { StartOfText, EndOfText });

        var vocabEncoder = new Dictionary<string, int>();
        var vocabDecoder = new Dictionary<int, string>();
        index = 0;
        foreach (var entry in vocab)
        {
            vocabEncoder.Add(entry, index);
            vocabDecoder.Add(index, entry);
            index++;
        }

        SimpleTextTokenizer tokenizer = new SimpleTextTokenizer(
            byteEncoder,
            byteDecoder,
            bpeRanks,
            vocabEncoder,
            vocabDecoder);

        return tokenizer;
    }

    private static (Dictionary<byte, char>, Dictionary<char, byte>) BytesToUnicode()
    {
        var bs = new List<byte>();
        var cs = new List<char>();

        // '!' to '~' is the range 33 to 126
        for (int i = '!'; i <= '~'; i++)
        {
            bs.Add((byte)i);
            cs.Add((char)i);
        }

        // '¡' (inverted excalamtion) to '¬' (logical not sign) is the range 161 to 172
        for (int i = '¡'; i <= '¬'; i++)
        {
            bs.Add((byte)i);
            cs.Add((char)i);
        }

        // '®' (registered trademark) to 'ÿ' (y umlaut) is the range 174 to 255
        for (int i = '®'; i <= 'ÿ'; i++)
        {
            bs.Add((byte)i);
            cs.Add((char)i);
        }

        int n = 0;
        for (int i = 0; i < 256; i++)
        {
            byte b = (byte)i;
            if (!bs.Contains(b))
            {
                bs.Add(b);
                cs.Add((char)(256 + n));
                n++;
            }
        }

        var encoder = new Dictionary<byte, char>();
        var decoder = new Dictionary<char, byte>();
        for (int i = 0; i < bs.Count; i++)
        {
            encoder[bs[i]] = cs[i];
            decoder[cs[i]] = bs[i];
        }

        return (encoder, decoder);
    }

    private static IList<(string, string)> GetPairs(string[] word)
    {
        var pairs = new List<(string, string)>();
        var prevChar = word[0];
        foreach (var @char in word.Skip(1))
        {
            pairs.Add((prevChar, @char));
            prevChar = @char;
        }

        return pairs;
    }

    private static (string, string) GetBySmallestRank(IEnumerable<(string, string)> pairs, IReadOnlyDictionary<string, float> bpeRanks)
    {
        return pairs.MinBy(p => bpeRanks.GetValueOrDefault($"{p.Item1} {p.Item2}", float.PositiveInfinity));
    }

    public SimpleTextTokenizer(
        Dictionary<byte, char> byteEncoder,
        Dictionary<char, byte> byteDecoder,
        Dictionary<string, float> bpeRanks,
        Dictionary<string, int> vocabEncoder,
        Dictionary<int, string> vocabDecoder)
    {
        _byteEncoder = byteEncoder;
        _byteDecoder = byteDecoder;

        _bpeRanks = bpeRanks;

        _vocabEncoder = vocabEncoder;
        _vocabDecoder = vocabDecoder;

        _bpeCache = new Dictionary<string, string>()
        {
            { StartOfText, StartOfText },
            { EndOfText, EndOfText }
        };
    }

    public int SotToken => _vocabEncoder[StartOfText];

    public int EotToken => _vocabEncoder[EndOfText];

    public IReadOnlyCollection<int> Encode(string input)
    {
        // TODO: do I need to re-implement the ftfy.fix_text? 
        var text = Whitespace()
            .Replace(input, " ")
            .Trim()
            .ToLowerInvariant();

        var bpeTokens = new List<int>();
        var matches = BpePattern().Matches(text);

        foreach (Match match in matches)
        {
            var bytes = Encoding.UTF8.GetBytes(match.Value);
            var reEncodedString = new string(bytes.Select(b => _byteEncoder[b]).ToArray());
            var bpeString = BytePairEncode(reEncodedString);

            var tokens = bpeString
                .Split(' ')
                .Select(bpeToken => _vocabEncoder[bpeToken])
                .ToArray();

            bpeTokens.AddRange(tokens);
        }

        return bpeTokens.ToArray();
    }

    public string Decode(IEnumerable<int> tokens)
    {
        byte[] bytes = tokens
            .SelectMany(token => _vocabDecoder[token].Select(@char => _byteDecoder[@char]))
            .ToArray();

        // TODO: error mode!
        return Encoding.UTF8.GetString(bytes).Replace("</w>", " ");
    }

    private string BytePairEncode(string input)
    {
        if (_bpeCache.TryGetValue(input, out var encode)) return encode;

        var word = input.ToCharArray().Select(c => $"{c}").ToArray();
        word[^1] = $"{word[^1]}</w>";
        var pairs = GetPairs(word);

        if (pairs.Count == 0) return $"{input}</w>";

        while (true)
        {
            // get item with the smallest rank
            var bigram = GetBySmallestRank(pairs, _bpeRanks);
            var bigramStr = $"{bigram.Item1} {bigram.Item2}";
            if (_bpeRanks.ContainsKey(bigramStr) == false) break;

            var (first, second) = bigram;
            var newWord = new List<string>();
            int i = 0;
            while (i < word.Length)
            {
                int j = Array.IndexOf(word, first, i);
                if (j < 0)
                {
                    newWord.AddRange(word.Skip(i));
                    break;
                }

                newWord.AddRange(word.Skip(i).Take(j - i));
                i = j;

                if (word[i] == first && i < word.Length - 1 && word[i + 1] == second)
                {
                    newWord.Add(first + second);
                    i += 2;
                }
                else
                {
                    newWord.Add(word[i]);
                    i += 1;
                }
            }

            word = newWord.ToArray();
            if (word.Length == 1) break;

            pairs = GetPairs(word);
        }

        var result = string.Join(" ", word);
        _bpeCache.Add(input, result);
        return result;
    }
}
