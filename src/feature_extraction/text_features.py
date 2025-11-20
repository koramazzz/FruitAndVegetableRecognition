"""
Metin özellik çıkarımı modülü
Word embeddings ve sentence embeddings kullanarak özellik çıkarımı
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
    warnings.warn("Gensim bulunamadı. Word2Vec kullanılamayacak.")

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    warnings.warn("SpaCy bulunamadı. SpaCy embeddings kullanılamayacak.")


class TextFeatureExtractor:
    """Metin özellik çıkarımı sınıfı"""
    
    def __init__(self, 
                 method: str = 'word2vec',
                 model_path: str = None,
                 embedding_dim: int = 100):
        """
        Args:
            method: Özellik çıkarım yöntemi ('word2vec', 'doc2vec', 'spacy', 'tfidf')
            model_path: Önceden eğitilmiş model yolu (varsa)
            embedding_dim: Embedding boyutu
        """
        self.method = method
        self.embedding_dim = embedding_dim
        self.model = None
        self.nlp = None
        
        if method == 'spacy' and SPACY_AVAILABLE:
            try:
                # Türkçe model yüklemeyi dene
                self.nlp = spacy.load("tr_core_news_sm")
            except OSError:
                try:
                    # İngilizce model yükle
                    self.nlp = spacy.load("en_core_web_sm")
                except OSError:
                    warnings.warn("SpaCy modeli bulunamadı. Basit tokenization kullanılacak.")
                    self.nlp = None
        
        if model_path and method in ['word2vec', 'doc2vec']:
            self.load_model(model_path)
    
    def tokenize(self, text: str) -> List[str]:
        """
        Metni tokenize et
        
        Args:
            text: Metin string'i
            
        Returns:
            Token listesi
        """
        if self.nlp:
            doc = self.nlp(text.lower())
            return [token.text for token in doc if not token.is_stop and not token.is_punct]
        else:
            # Basit tokenization
            return text.lower().split()
    
    def train_word2vec(self, texts: List[str], **kwargs):
        """
        Word2Vec modelini eğit
        
        Args:
            texts: Eğitim metinleri listesi
            **kwargs: Word2Vec parametreleri
        """
        if not GENSIM_AVAILABLE:
            raise ImportError("Gensim kurulu değil. Word2Vec kullanılamaz.")
        
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
        Doc2Vec modelini eğit
        
        Args:
            texts: Eğitim metinleri listesi
            **kwargs: Doc2Vec parametreleri
        """
        if not GENSIM_AVAILABLE:
            raise ImportError("Gensim kurulu değil. Doc2Vec kullanılamaz.")
        
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
        Word2Vec kullanarak özellik çıkar
        
        Args:
            text: Metin string'i
            
        Returns:
            Özellik vektörü
        """
        if self.model is None:
            raise ValueError("Word2Vec modeli eğitilmemiş veya yüklenmemiş")
        
        tokens = self.tokenize(text)
        if not tokens:
            return np.zeros(self.embedding_dim)
        
        # Tüm token'ların ortalamasını al
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
        Doc2Vec kullanarak özellik çıkar
        
        Args:
            text: Metin string'i
            
        Returns:
            Özellik vektörü
        """
        if self.model is None:
            raise ValueError("Doc2Vec modeli eğitilmemiş veya yüklenmemiş")
        
        tokens = self.tokenize(text)
        if not tokens:
            return np.zeros(self.embedding_dim)
        
        return self.model.infer_vector(tokens)
    
    def extract_spacy_features(self, text: str) -> np.ndarray:
        """
        SpaCy kullanarak özellik çıkar
        
        Args:
            text: Metin string'i
            
        Returns:
            Özellik vektörü
        """
        if self.nlp is None:
            raise ValueError("SpaCy modeli yüklenmemiş")
        
        doc = self.nlp(text)
        if doc.has_vector:
            return doc.vector
        else:
            # Token vektörlerinin ortalamasını al
            token_vectors = [token.vector for token in doc if token.has_vector]
            if token_vectors:
                return np.mean(token_vectors, axis=0)
            else:
                return np.zeros(300)  # SpaCy varsayılan boyutu
    
    def extract_features(self, text: str) -> np.ndarray:
        """
        Metinden özellik çıkar
        
        Args:
            text: Metin string'i
            
        Returns:
            Özellik vektörü
        """
        if self.method == 'word2vec':
            return self.extract_word2vec_features(text)
        elif self.method == 'doc2vec':
            return self.extract_doc2vec_features(text)
        elif self.method == 'spacy':
            return self.extract_spacy_features(text)
        else:
            raise ValueError(f"Bilinmeyen yöntem: {self.method}")
    
    def load_model(self, model_path: str):
        """
        Önceden eğitilmiş modeli yükle
        
        Args:
            model_path: Model dosya yolu
        """
        if self.method == 'word2vec':
            self.model = Word2Vec.load(model_path)
        elif self.method == 'doc2vec':
            self.model = Doc2Vec.load(model_path)
        else:
            raise ValueError(f"Model yükleme {self.method} için desteklenmiyor")
    
    def save_model(self, model_path: str):
        """
        Modeli kaydet
        
        Args:
            model_path: Kayıt yolu
        """
        if self.model is None:
            raise ValueError("Kaydedilecek model yok")
        
        self.model.save(model_path)


def extract_text_features(texts: List[str],
                         extractor: TextFeatureExtractor = None,
                         train_model: bool = True) -> np.ndarray:
    """
    Birden fazla metinden özellik çıkar
    
    Args:
        texts: Metin listesi
        extractor: TextFeatureExtractor instance (None ise varsayılan oluşturulur)
        train_model: Modeli eğit (True ise)
        
    Returns:
        Özellik matrisi (n_samples, n_features)
    """
    if extractor is None:
        extractor = TextFeatureExtractor(method='word2vec', embedding_dim=100)
    
    # Modeli eğit (gerekirse)
    if train_model and extractor.model is None:
        if extractor.method == 'word2vec':
            extractor.train_word2vec(texts)
        elif extractor.method == 'doc2vec':
            extractor.train_doc2vec(texts)
    
    features_list = []
    for text in texts:
        if not text or pd.isna(text):
            # Boş metin için sıfır vektör
            features = np.zeros(extractor.embedding_dim)
        else:
            features = extractor.extract_features(str(text))
        features_list.append(features)
    
    return np.array(features_list)

