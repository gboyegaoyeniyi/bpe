from collections import defaultdict

class BytePairEncoding:
    def __init__(self):
        pass

    def load_tokenizer():
        pass

    def save_tokenizer():
        pass

    def get_stats(self, chunks) -> dict:
        counts = defaultdict(int)

        for chunk in chunks:
            for pair in zip(chunk, chunk[1:]):
                counts[pair] += 1
        return dict(counts)

    def train():
        pass

    def merge():
        pass

    def encode():
        pass

    def decode():
        pass


bpe = BytePairEncoding()

stats = bpe.get_stats([[1,1,1], [2,4,4]])
print(stats)