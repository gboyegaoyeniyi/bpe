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

    def merge_chunk(self, id_chunk, pair, new_id):
        out = []
        i = 0
        while i < len(id_chunk):
            if i < len(id_chunk) - 1 and (id_chunk[i], id_chunk[i + 1]) == pair:
                out.append(new_id)
                i += 2
            else:
                out.append(id_chunk[i])
                i += 1
        return out

    def merge(self, chunked_corpus, pair, new_id):
        merged_corpus_chunks = []
        for chunk in chunked_corpus:
            merged_chunk = self.merge_chunk(chunk, pair, new_id)
            print(f"Merging chunk: {chunk} with pair: {pair} to new_id: {new_id} => {merged_chunk}")
            merged_corpus_chunks.append(merged_chunk)

        return merged_corpus_chunks

    def train():
        pass

    def encode():
        pass

    def decode():
        pass


bpe = BytePairEncoding()

corpus = [[1,1,1], [2,4,4]]
print(f"Corpus: {corpus}")
stats = bpe.get_stats(corpus)
print(f"Stats: {stats}")
merged = bpe.merge(corpus, (1,1), 5)
print(f"Merge: {merged}")

