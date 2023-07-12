using System.IO.Compression;
using System.Text;
using System.Text.RegularExpressions;

namespace ClipSharp;

public interface ITextTokenizer
{
    IReadOnlyCollection<int> Encode(string input);

    string Decode(IEnumerable<int> tokens);
}

public partial class SimpleTokenizer : ITextTokenizer
{
    [GeneratedRegex("""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", RegexOptions.IgnoreCase | RegexOptions.Compiled)]
    private static partial Regex BpePattern();

    [GeneratedRegex("""\s+""", RegexOptions.Compiled)]
    private static partial Regex Whitespace();


    private readonly Dictionary<byte, char> _byteEncoder;
    private readonly Dictionary<char, byte> _byteDecoder;
    private readonly Dictionary<(string, string), float> _bpeRanks;
    private readonly Dictionary<string, int> _vocabEncoder;
    private readonly Dictionary<int, string> _vocabDecoder;

    private Dictionary<string, string> _bpeCache;

    public static SimpleTokenizer Load()
    {
        // generate byte encoder
        var byteEncoder = BytesToUnicode();

        // generate vocab encoder
        var merges = new List<(string, string)>();
        using (var fileStream = new FileStream("./bpe_simple_vocab_16e6.txt.gz", FileMode.Open))
        using (var textStream = new GZipStream(fileStream, CompressionMode.Decompress))
        {
            var reader = new StreamReader(textStream);

            reader.ReadLine(); // ignore first line
            string? line;
            while ((line = reader.ReadLine()) is not null)
            {
                if (line.Length == 0) continue;

                var bpe = line.Split(' ');
                if (bpe.Length == 0) continue;

                merges.Add((bpe[0], bpe[1]));
            }
        }

        var vocab = byteEncoder.Values.Select(c => $"{c}</w>").ToList();
        foreach (var merge in merges)
        {
            vocab.Add(string.Join(" ", merge));
        }
        vocab.AddRange(new[] { "<|startoftext|>", "<|endoftext|>" });


        SimpleTokenizer tokenizer = new SimpleTokenizer(byteEncoder, merges, vocab);

        return tokenizer;
    }

    private static Dictionary<byte, char> BytesToUnicode()
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
        for (byte b = byte.MinValue; b <= byte.MaxValue; b++)
        {
            if (!bs.Contains(b))
            {
                bs.Add(b);
                cs.Add((char)(256 + n));
                n += 1;
            }
        }

        //cs.AddRange(bs.GetRange(256, 25).Select(b => (char )b));
        var result = new Dictionary<byte, char>();
        for (int i = 0; i < bs.Count; i++) result[bs[i]] = cs[i];
        return result;
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

    private static (string, string) GetBySmallestRank(IEnumerable<(string, string)> pairs,
    IReadOnlyDictionary<(string, string), float> bpeRanks)
    {
        return pairs.MinBy(p => bpeRanks.GetValueOrDefault(p, float.PositiveInfinity));
    }

    public SimpleTokenizer(Dictionary<byte, char> byteEncoder, IReadOnlyCollection<(string, string)> orderedMerges, IReadOnlyCollection<string> orderedVocabulary)
    {
        _byteEncoder = byteEncoder;
        _byteDecoder = byteEncoder.ToDictionary(kv => kv.Value, kv => kv.Key);

        _bpeRanks = new Dictionary<(string, string), float>();
        int index = 0;
        foreach (var merge in orderedMerges)
        {
            _bpeRanks.Add(merge, index);
            index++;
        }

        _vocabEncoder = new Dictionary<string, int>();
        _vocabDecoder = new Dictionary<int, string>();
        index = 0;
        foreach (var entry in orderedVocabulary)
        {
            _vocabEncoder.Add(entry, index);
            _vocabDecoder.Add(index, entry);
            index++;
        }

        _bpeCache = new Dictionary<string, string>()
        {
            { "<|startoftext|>", "<|startoftext|>" },
            { "<|endoftext|>", "<|endoftext|>" }
        };
    }

    public IReadOnlyCollection<int> Encode(string input)
    {
        // TODO: do I need to re-implement the ftfy.fix_text? 
        var text = Whitespace().Replace(input, " ").Trim();

        var bpeTokens = new List<int>();
        var matches = BpePattern().Matches(text);

        foreach (Match match in matches)
        {
            var bytes = Encoding.UTF8.GetBytes(match.Value);
            var encodedMatch = new string(bytes.Select(b => _byteEncoder[b]).ToArray());
            var bpeString = BytePairEncode(encodedMatch);
            bpeTokens.AddRange(bpeString.Split(' ').Select(bpeToken => _vocabEncoder[bpeToken]));
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
            if (_bpeRanks.ContainsKey(bigram) == false) break;

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
