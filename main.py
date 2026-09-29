from collections import defaultdict

class BytePairEncoding:
    def __init__(self):
        self.merges = {}
        pass

    def load_tokenizer():
        pass

    def save_tokenizer():
        pass

    def get_stats(self, chunked_corpus) -> dict:
        counts = defaultdict(int)

        for chunk in chunked_corpus:
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
            merged_corpus_chunks.append(merged_chunk)

        return merged_corpus_chunks
        
    def prepare_chunks(self, text_corpus):
        return [list(text_corpus.encode("utf-8"))]
        
    def train(self, text_corpus, num_merges):
        chunks = self.prepare_chunks(text_corpus)
        for i in range(num_merges):
            stats = self.get_stats(chunks)
            if not stats:
                break
            pair = max(stats, key=stats.get)
            new_id = 256 + i
            chunks = self.merge(chunks, pair, new_id)
            self.merges[pair] = new_id

        return chunks

    def encode(self, text):
        chunks = self.prepare_chunks(text)
        while True:
            stats = self.get_stats(chunks)
            if not stats:
                break

            learned_pairs = {}

            for pair in stats:
                if pair in self.merges:
                    learned_pairs[pair] = self.merges[pair]

            if not learned_pairs:
                break

            pair = None
            pair_id = float("inf")

            for learned_pair, learned_pair_id in learned_pairs.items():
                if learned_pair_id < pair_id:
                    pair = learned_pair
                    pair_id = learned_pair_id
                    
            chunks = self.merge(chunks, pair, pair_id)
            
        return chunks

    def decode():
        pass


bpe = BytePairEncoding()

corpus = "aabtaaz"


corpus = bpe.train(corpus, 5)
print(f"Merges: {bpe.merges}")
print(f"Final corpus after training: {corpus}")
encoded = bpe.encode("aagaa")
print(f"Encoded: {encoded}")


