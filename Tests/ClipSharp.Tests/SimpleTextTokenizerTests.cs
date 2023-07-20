using FluentAssertions;

namespace ClipSharp.Tests
{
    public class SimpleTextTokenizerTests
    {
        private SimpleTextTokenizer _tokenizer;

        public SimpleTextTokenizerTests()
        {
            _tokenizer = SimpleTextTokenizer.Load();
        }

        [Fact]
        public void Load_ValidTokenizer()
        {
            // Assert.
            _tokenizer.Should().NotBeNull();
        }

        [Fact]
        public void Encode_EnglishText_ValidTokenList()
        {
            // Act.
            var result = _tokenizer.Encode("This is a test");

            // Assert.
            result.Should().NotBeNull();
            result.Should().ContainInOrder(589, 533, 320, 1628);
        }

        [Fact]
        public void Encode_EnglishText_ValidTokenList2()
        {
            // Act.
            var result = _tokenizer.Encode("make a picture of green tree with flowers around it and a red sky");

            // Assert.
            result.Should().ContainInOrder(
                1078, 320, 1674, 539, 1901, 2677, 593, 4023, 1630,
                585, 537, 320, 736, 2390
                );
        }
    }
}
