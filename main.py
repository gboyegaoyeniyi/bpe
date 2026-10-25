from collections import defaultdict
import regex as re
import json

class BytePairEncoding:
    def __init__(self, tokenizer_path=None):
        self.tokenizer_path = tokenizer_path
        self.merges = {}
        self.vocab = {}
        self.pattern = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def load_tokenizer(self, path=None):
        if path is None:
            if self.tokenizer_path is None:
                raise ValueError("Tokenizer path is not set. Please provide a path to load the tokenizer.")
            path = self.tokenizer_path
        with open(path, "r") as f:
            data = json.load(f)
            self.pattern = data["pattern"]
            self.merges = {tuple(pair): i + 256  for i, pair in enumerate(data["merges"])}

        self._build_vocab()

    def save_tokenizer(self, path=None):
        if path is None:
            if self.tokenizer_path is None:
                raise ValueError("Tokenizer path is not set. Please provide a path to save the tokenizer.")
            path = self.tokenizer_path

        data = {
            "version": 1,
            "pattern": self.pattern,
            "merges": [list(pair) for pair in self.merges.keys()],
        }
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

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
        chunks = re.findall(self.pattern, text_corpus)
        return [list(chunk.encode("utf-8")) for chunk in chunks]

    def _build_vocab(self):
        self.vocab = {i: bytes([i]) for i in range(256)}
        for pair, id in self.merges.items():
            self.vocab[id] = self.vocab[pair[0]] + self.vocab[pair[1]]
        
    def train(self, text_corpus, num_merges, save_tokenizer=False, tokenizer_save_path=None):
        chunks = self.prepare_chunks(text_corpus)
        for i in range(num_merges):
            stats = self.get_stats(chunks)
            if not stats:
                break
            pair = max(stats, key=stats.get)
            new_id = 256 + i
            chunks = self.merge(chunks, pair, new_id)
            self.merges[pair] = new_id

        self._build_vocab()

        if save_tokenizer:
            self.save_tokenizer(path=tokenizer_save_path)

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

        chunk = []
        for chunked in chunks:
            chunk.extend(chunked)
        return chunk

    def decode(self, encoded_bytes):
        decoded_bytes = []
        for token_id in encoded_bytes:
            decoded_bytes.append(self.vocab[token_id])
        return b''.join(decoded_bytes).decode("utf-8")
        


bpe = BytePairEncoding()

corpus = "The grand Ibadan band played at the abundant Abuja banana stand."
bpe.train(text_corpus=corpus, num_merges=100, save_tokenizer=True, tokenizer_save_path="tokenizer.json")
bpe2 = BytePairEncoding()
bpe2.load_tokenizer("tokenizer.json")
encoded = bpe2.encode("A grand band stands in Abuja 🇳🇬")
print(f"Encoded: {encoded}")
decoded = bpe2.decode(encoded)
print(f"Decoded: {decoded}")


