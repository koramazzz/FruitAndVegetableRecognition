"""
Text feature extraction module
Feature extraction using word embeddings and sentence embeddings
"""

import numpy as np
import pandas as pd
from typing import List
import warnings

try:
    from gensim.models import Word2Vec
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    warnings.warn("Gensim not found. Word2Vec will not be available.")

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    warnings.warn("SpaCy not found. SpaCy embeddings will not be available.")


class TextFeatureExtractor:
    """Text feature extraction class"""
    
    def __init__(self, 
                 method: str = 'word2vec',
                 model_path: str = None,
                 embedding_dim: int = 100):
        """
        Args:
            method: Feature extraction method ('word2vec', 'doc2vec', 'spacy', 'tfidf')
            model_path: Path to pre-trained model (if any)
            embedding_dim: Embedding dimension
        """
        self.method = method
        self.embedding_dim = embedding_dim
        self.model = None
        self.nlp = None
        
        if method == 'spacy' and SPACY_AVAILABLE:
            try:
                # Try loading Turkish model
                self.nlp = spacy.load("tr_core_news_sm")
            except OSError:
                try:
                    # Load English model
                    self.nlp = spacy.load("en_core_web_sm")
                except OSError:
                    warnings.warn("SpaCy model not found. Simple tokenization will be used.")
                    self.nlp = None
        
        if model_path and method in ['word2vec', 'doc2vec']:
            self.load_model(model_path)
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text
        
        Args:
            text: Text string
            
        Returns:
            List of tokens
        """
        if self.nlp:
            doc = self.nlp(text.lower())
            return [token.text for token in doc if not token.is_stop and not token.is_punct]
        else:
            # Basit tokenization
            return text.lower().split()
    
    def train_word2vec(self, texts: List[str], **kwargs):
        """
        Train Word2Vec model
        
        Args:
            texts: List of training texts
            **kwargs: Word2Vec parameters
        """
        if not GENSIM_AVAILABLE:
            raise ImportError("Gensim is not installed. Word2Vec cannot be used.")
        
        tokenized_texts = [self.tokenize(text) for text in texts]
        
        self.model = Word2Vec(
            sentences=tokenized_texts,
            vector_size=self.embedding_dim,
            window=5,
            min_count=1,
            workers=4,
            **kwargs
        )
    
    def train_doc2vec(self, texts: List[str], **kwargs):
        """
        Train Doc2Vec model
        
        Args:
            texts: List of training texts
            **kwargs: Doc2Vec parameters
        """
        if not GENSIM_AVAILABLE:
            raise ImportError("Gensim is not installed. Doc2Vec cannot be used.")
        
        tokenized_texts = [self.tokenize(text) for text in texts]
        tagged_docs = [TaggedDocument(words=text, tags=[i]) for i, text in enumerate(tokenized_texts)]
        
        self.model = Doc2Vec(
            documents=tagged_docs,
            vector_size=self.embedding_dim,
            window=5,
            min_count=1,
            workers=4,
            **kwargs
        )
    
    def extract_word2vec_features(self, text: str) -> np.ndarray:
        """
        Extract features using Word2Vec
        
        Args:
            text: Text string
            
        Returns:
            Feature vector
        """
        if self.model is None:
            raise ValueError("Word2Vec model is not trained or loaded")
        
        tokens = self.tokenize(text)
        if not tokens:
            return np.zeros(self.embedding_dim)
        
        # Average all tokens
        word_vectors = []
        for token in tokens:
            if token in self.model.wv:
                word_vectors.append(self.model.wv[token])
        
        if word_vectors:
            return np.mean(word_vectors, axis=0)
        else:
            return np.zeros(self.embedding_dim)
    
    def extract_doc2vec_features(self, text: str) -> np.ndarray:
        """
        Extract features using Doc2Vec
        
        Args:
            text: Text string
            
        Returns:
            Feature vector
        """
        if self.model is None:
            raise ValueError("Doc2Vec model is not trained or loaded")
        
        tokens = self.tokenize(text)
        if not tokens:
            return np.zeros(self.embedding_dim)
        
        return self.model.infer_vector(tokens)
    
    def extract_spacy_features(self, text: str) -> np.ndarray:
        """
        Extract features using SpaCy
        
        Args:
            text: Text string
            
        Returns:
            Feature vector
        """
        if self.nlp is None:
            raise ValueError("SpaCy model is not loaded")
        
        doc = self.nlp(text)
        if doc.has_vector:
            return doc.vector
        else:
            # Average token vectors
            token_vectors = [token.vector for token in doc if token.has_vector]
            if token_vectors:
                return np.mean(token_vectors, axis=0)
            else:
                return np.zeros(300)  # SpaCy default size
    
    def extract_features(self, text: str) -> np.ndarray:
        """
        Extract features from text
        
        Args:
            text: Text string
            
        Returns:
            Feature vector
        """
        if self.method == 'word2vec':
            return self.extract_word2vec_features(text)
        elif self.method == 'doc2vec':
            return self.extract_doc2vec_features(text)
        elif self.method == 'spacy':
            return self.extract_spacy_features(text)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def load_model(self, model_path: str):
        """
        Load pre-trained model
        
        Args:
            model_path: Model file path
        """
        if self.method == 'word2vec':
            self.model = Word2Vec.load(model_path)
        elif self.method == 'doc2vec':
            self.model = Doc2Vec.load(model_path)
        else:
            raise ValueError(f"Model loading is not supported for {self.method}")
    
    def save_model(self, model_path: str):
        """
        Save model
        
        Args:
            model_path: Save path
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        self.model.save(model_path)


def extract_text_features(texts: List[str],
                         extractor: TextFeatureExtractor = None,
                         train_model: bool = True) -> np.ndarray:
    """
    Extract features from multiple texts
    
    Args:
        texts: List of texts
        extractor: TextFeatureExtractor instance (default created if None)
        train_model: Train model (if True)
        
    Returns:
        Feature matrix (n_samples, n_features)
    """
    if extractor is None:
        extractor = TextFeatureExtractor(method='word2vec', embedding_dim=100)
    
    # Train model (if needed)
    if train_model and extractor.model is None:
        if extractor.method == 'word2vec':
            extractor.train_word2vec(texts)
        elif extractor.method == 'doc2vec':
            extractor.train_doc2vec(texts)
    
    features_list = []
    for text in texts:
        if not text or pd.isna(text):
            # Zero vector for empty text
            features = np.zeros(extractor.embedding_dim)
        else:
            features = extractor.extract_features(str(text))
        features_list.append(features)
    
    return np.array(features_list)

