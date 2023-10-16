from spacy.tokenizer import Tokenizer
from spacy.vocab import Vocab

class SpacyTokenizer:
    PAD_TOKEN = "<PAD>"
    PAD_ID = 0
    
    def __init__(self):
        self.vocab = Vocab()
        self.tokenizer = Tokenizer(self.vocab)
        self.lex_to_index = {}
        self.next_index = 1 
        
    def tokenize(self, sentence):
        """Tokenize a sentence into words."""
        return [token.text for token in self.tokenizer(sentence)]

    def tokenize_data(self, data):
        """Tokenize multiple sentences."""
        return [self.tokenize(sentence) for sentence in data]

    def to_ids(self, tokens):
        """Convert tokens to unique numerical IDs."""
        ids = []
        for token in tokens:
            lex_id = self.vocab.strings[token]
            if lex_id not in self.lex_to_index:
                self.lex_to_index[lex_id] = self.next_index
                self.next_index += 1
            ids.append(self.lex_to_index[lex_id])
            
        return ids

    def remove_punctuation(self, sentence):
        """Remove punctuation from a sentence."""
        doc = self.tokenizer(sentence)
        return [token.text for token in doc if not token.is_punct]

    def pad_sequences(self, sequences, max_length=None):
        """Add Pad sequences to the same length."""
        if max_length is None:
            max_length = max(len(sequence) for sequence in sequences)
        return [sequence + [self.PAD_ID] * (max_length - len(sequence)) for sequence in sequences]
    
    @property
    def index_to_lex(self):
        if not hasattr(self, "_index_to_lex"):
            self._index_to_lex = {index: lex for lex, index in self.lex_to_index.items()}
        return self._index_to_lex

    def to_text(self, token_ids):
        return ' '.join([self.index_to_lex[token_id] for token_id in token_ids if token_id in self.index_to_lex])