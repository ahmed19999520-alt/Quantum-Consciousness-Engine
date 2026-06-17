import numpy as np
from typing import List, Dict, Tuple
import re

class TextProcessor:
    def __init__(self):
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab_size = 0
        
    def build_vocab(self, texts: List[str], max_vocab: int = 10000):
        word_freq = {}
        for text in texts:
            words = self.tokenize(text)
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        sorted_words = sorted_words[:max_vocab - 2]
        
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx_to_word = {0: '<PAD>', 1: '<UNK>'}
        
        for idx, (word, _) in enumerate(sorted_words, start=2):
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word
        
        self.vocab_size = len(self.word_to_idx)
        
    def tokenize(self, text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text.split()
    
    def encode(self, text: str, max_length: int = 512) -> np.ndarray:
        words = self.tokenize(text)
        indices = [self.word_to_idx.get(word, 1) for word in words]
        
        if len(indices) < max_length:
            indices = indices + [0] * (max_length - len(indices))
        else:
            indices = indices[:max_length]
        
        return np.array(indices, dtype=np.int64)
    
    def decode(self, indices: np.ndarray) -> str:
        words = [self.idx_to_word.get(idx, '<UNK>') for idx in indices if idx != 0]
        return ' '.join(words)

class ConsciousnessDataset:
    def __init__(self, texts: List[str], responses: List[str]):
        self.texts = texts
        self.responses = responses
        self.processor = TextProcessor()
        
    def prepare(self, max_length: int = 512):
        all_texts = self.texts + self.responses
        self.processor.build_vocab(all_texts)
        
        self.encoded_texts = [
            self.processor.encode(text, max_length)
            for text in self.texts
        ]
        
        self.encoded_responses = [
            self.processor.encode(response, max_length)
            for response in self.responses
        ]
        
        return self.encoded_texts, self.encoded_responses