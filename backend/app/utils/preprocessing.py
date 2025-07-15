import re
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import unicodedata
import logging

logger = logging.getLogger(__name__)

class TextPreprocessor:
    """Comprehensive text preprocessing for climate change social media data"""
    
    def __init__(self):
        # Climate-specific replacements and normalizations
        self.climate_replacements = {
            'co2': 'carbon dioxide',
            'ghg': 'greenhouse gas',
            'agw': 'anthropogenic global warming',
            'ipcc': 'intergovernmental panel on climate change',
            'unfccc': 'united nations framework convention on climate change',
            'cop': 'conference of the parties',
            'greta': 'climate activist',
            'fff': 'fridays for future'
        }
        
        # Common contractions
        self.contractions = {
            "aren't": "are not", "can't": "cannot", "couldn't": "could not",
            "didn't": "did not", "doesn't": "does not", "don't": "do not",
            "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
            "he'd": "he would", "he'll": "he will", "he's": "he is",
            "i'd": "i would", "i'll": "i will", "i'm": "i am", "i've": "i have",
            "isn't": "is not", "it'd": "it would", "it'll": "it will",
            "it's": "it is", "let's": "let us", "mustn't": "must not",
            "shan't": "shall not", "she'd": "she would", "she'll": "she will",
            "she's": "she is", "shouldn't": "should not", "that's": "that is",
            "there's": "there is", "they'd": "they would", "they'll": "they will",
            "they're": "they are", "they've": "they have", "we'd": "we would",
            "we're": "we are", "we've": "we have", "weren't": "were not",
            "what's": "what is", "where's": "where is", "who's": "who is",
            "won't": "will not", "wouldn't": "would not", "you'd": "you would",
            "you'll": "you will", "you're": "you are", "you've": "you have"
        }
        
        # Emoji patterns
        self.emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE
        )
        
        # URL patterns
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        
        # Social media patterns
        self.mention_pattern = re.compile(r'@\w+')
        self.hashtag_pattern = re.compile(r'#\w+')
        
        # Non-ASCII and special characters
        self.special_chars = re.compile(r'[^\w\s]')
        self.multiple_spaces = re.compile(r'\s+')
        
    def preprocess(self, text: str, level: str = 'standard') -> str:
        """
        Comprehensive text preprocessing
        
        Args:
            text: Input text to preprocess
            level: Preprocessing level ('minimal', 'standard', 'aggressive')
        
        Returns:
            Preprocessed text
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        try:
            # Start with original text
            processed_text = text
            
            # Level 1: Minimal preprocessing
            processed_text = self._basic_cleaning(processed_text)
            
            if level in ['standard', 'aggressive']:
                # Level 2: Standard preprocessing
                processed_text = self._standard_cleaning(processed_text)
                processed_text = self._normalize_climate_terms(processed_text)
                processed_text = self._expand_contractions(processed_text)
            
            if level == 'aggressive':
                # Level 3: Aggressive preprocessing
                processed_text = self._aggressive_cleaning(processed_text)
            
            # Final cleanup
            processed_text = self._final_cleanup(processed_text)
            
            return processed_text
            
        except Exception as e:
            logger.error(f"Error preprocessing text: {str(e)}")
            return text  # Return original text if preprocessing fails
    
    def _basic_cleaning(self, text: str) -> str:
        """Basic text cleaning operations"""
        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text)
        
        # Remove zero-width characters
        text = re.sub(r'[\u200b-\u200d\ufeff]', '', text)
        
        # Fix common encoding issues
        text = text.replace('â€™', "'").replace('â€œ', '"').replace('â€', '"')
        text = text.replace('Ã¡', 'á').replace('Ã©', 'é').replace('Ã­', 'í')
        text = text.replace('Ã³', 'ó').replace('Ãº', 'ú').replace('Ã±', 'ñ')
        
        # Remove excessive whitespace
        text = self.multiple_spaces.sub(' ', text).strip()
        
        return text
    
    def _standard_cleaning(self, text: str) -> str:
        """Standard cleaning operations"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs but keep placeholder
        text = self.url_pattern.sub(' [URL] ', text)
        
        # Handle mentions and hashtags
        text = self.mention_pattern.sub(' [MENTION] ', text)
        text = self.hashtag_pattern.sub(' [HASHTAG] ', text)
        
        # Remove emojis but keep placeholder
        emoji_count = len(self.emoji_pattern.findall(text))
        if emoji_count > 0:
            text = self.emoji_pattern.sub(' [EMOJI] ', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[!]{2,}', ' ! ', text)
        text = re.sub(r'[?]{2,}', ' ? ', text)
        text = re.sub(r'[.]{3,}', ' ... ', text)
        
        # Handle repeated characters
        text = re.sub(r'(.)\1{3,}', r'\1\1', text)
        
        return text
    
    def _aggressive_cleaning(self, text: str) -> str:
        """Aggressive cleaning for analytical purposes"""
        # Remove all placeholders
        text = re.sub(r'\[URL\]|\[MENTION\]|\[HASHTAG\]|\[EMOJI\]', '', text)
        
        # Remove all punctuation except sentence endings
        text = re.sub(r'[^\w\s.!?]', ' ', text)
        
        # Remove numbers unless part of climate terms
        text = re.sub(r'\b(?<!co)\d+(?!°|ppm|ppb)\b', '', text)
        
        # Remove single characters except meaningful ones
        text = re.sub(r'\b[a-z]\b', '', text)
        
        return text
    
    def _normalize_climate_terms(self, text: str) -> str:
        """Normalize climate-specific terminology"""
        for abbrev, full_form in self.climate_replacements.items():
            # Case-insensitive replacement
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            text = re.sub(pattern, full_form, text, flags=re.IGNORECASE)
        
        # Normalize temperature units
        text = re.sub(r'(\d+)\s*°?\s*c\b', r'\1 degrees celsius', text, flags=re.IGNORECASE)
        text = re.sub(r'(\d+)\s*°?\s*f\b', r'\1 degrees fahrenheit', text, flags=re.IGNORECASE)
        
        # Normalize concentration units
        text = re.sub(r'(\d+)\s*ppm\b', r'\1 parts per million', text, flags=re.IGNORECASE)
        text = re.sub(r'(\d+)\s*ppb\b', r'\1 parts per billion', text, flags=re.IGNORECASE)
        
        return text
    
    def _expand_contractions(self, text: str) -> str:
        """Expand contractions to full forms"""
        for contraction, expansion in self.contractions.items():
            pattern = r'\b' + re.escape(contraction) + r'\b'
            text = re.sub(pattern, expansion, text, flags=re.IGNORECASE)
        
        return text
    
    def _final_cleanup(self, text: str) -> str:
        """Final cleanup operations"""
        # Remove excessive whitespace
        text = self.multiple_spaces.sub(' ', text).strip()
        
        # Remove empty parentheses and brackets
        text = re.sub(r'\(\s*\)|\[\s*\]|\{\s*\}', '', text)
        
        # Remove leading/trailing punctuation from words
        text = re.sub(r'\b[^\w\s]+\b', '', text)
        
        # Ensure single space between sentences
        text = re.sub(r'([.!?])\s+', r'\1 ', text)
        
        return text.strip()
    
    def extract_features(self, text: str) -> Dict[str, Any]:
        """Extract features from text for analysis"""
        if pd.isna(text) or not isinstance(text, str):
            return self._empty_features()
        
        try:
            features = {}
            
            # Basic text statistics
            features['original_length'] = len(text)
            features['word_count'] = len(text.split())
            features['sentence_count'] = len(re.split(r'[.!?]+', text))
            features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
            
            # Preprocessing level features
            clean_text = self.preprocess(text, level='standard')
            features['clean_length'] = len(clean_text)
            features['clean_word_count'] = len(clean_text.split())
            
            # Character-level features
            features['uppercase_count'] = sum(1 for c in text if c.isupper())
            features['lowercase_count'] = sum(1 for c in text if c.islower())
            features['digit_count'] = sum(1 for c in text if c.isdigit())
            features['punctuation_count'] = sum(1 for c in text if c in '.,!?;:')
            
            # Social media features
            features['url_count'] = len(self.url_pattern.findall(text))
            features['mention_count'] = len(self.mention_pattern.findall(text))
            features['hashtag_count'] = len(self.hashtag_pattern.findall(text))
            features['emoji_count'] = len(self.emoji_pattern.findall(text))
            
            # Language features
            features['question_count'] = text.count('?')
            features['exclamation_count'] = text.count('!')
            features['caps_ratio'] = features['uppercase_count'] / len(text) if text else 0
            
            # Climate-specific features
            climate_terms = ['climate', 'global warming', 'carbon', 'emission', 'renewable', 'fossil']
            features['climate_terms_count'] = sum(1 for term in climate_terms if term.lower() in text.lower())
            
            # Complexity indicators
            features['unique_words'] = len(set(text.lower().split()))
            features['lexical_diversity'] = features['unique_words'] / features['word_count'] if features['word_count'] > 0 else 0
            
            # Readability approximation
            if features['sentence_count'] > 0 and features['word_count'] > 0:
                avg_sentence_length = features['word_count'] / features['sentence_count']
                features['readability_score'] = max(0, 100 - avg_sentence_length * 2)
            else:
                features['readability_score'] = 50
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            return self._empty_features()
    
    def _empty_features(self) -> Dict[str, Any]:
        """Return empty feature dictionary"""
        return {
            'original_length': 0, 'word_count': 0, 'sentence_count': 0,
            'avg_word_length': 0, 'clean_length': 0, 'clean_word_count': 0,
            'uppercase_count': 0, 'lowercase_count': 0, 'digit_count': 0,
            'punctuation_count': 0, 'url_count': 0, 'mention_count': 0,
            'hashtag_count': 0, 'emoji_count': 0, 'question_count': 0,
            'exclamation_count': 0, 'caps_ratio': 0, 'climate_terms_count': 0,
            'unique_words': 0, 'lexical_diversity': 0, 'readability_score': 50
        }
    
    def batch_preprocess(self, texts: List[str], level: str = 'standard') -> List[str]:
        """Preprocess multiple texts in batch"""
        try:
            return [self.preprocess(text, level) for text in texts]
        except Exception as e:
            logger.error(f"Error in batch preprocessing: {str(e)}")
            return texts
    
    def batch_extract_features(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Extract features from multiple texts in batch"""
        try:
            return [self.extract_features(text) for text in texts]
        except Exception as e:
            logger.error(f"Error in batch feature extraction: {str(e)}")
            return [self._empty_features() for _ in texts]
    
    def detect_language_simple(self, text: str) -> str:
        """Simple heuristic-based language detection"""
        if pd.isna(text) or not isinstance(text, str):
            return 'unknown'
        
        # Simple English detection based on common words
        english_indicators = ['the', 'and', 'is', 'in', 'to', 'of', 'a', 'that', 'it', 'with']
        words = text.lower().split()
        english_word_count = sum(1 for word in words if word in english_indicators)
        
        if len(words) > 0 and english_word_count / len(words) > 0.1:
            return 'english'
        else:
            return 'other'
    
    def clean_for_analysis(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """Clean entire DataFrame for analysis"""
        try:
            df_clean = df.copy()
            
            if text_column not in df_clean.columns:
                logger.warning(f"Column '{text_column}' not found in DataFrame")
                return df_clean
            
            # Preprocess all texts
            logger.info("Preprocessing text data...")
            df_clean[f'{text_column}_clean'] = df_clean[text_column].apply(
                lambda x: self.preprocess(str(x), level='standard')
            )
            
            # Extract features
            logger.info("Extracting text features...")
            features_list = self.batch_extract_features(df_clean[text_column].fillna('').astype(str).tolist())
            features_df = pd.DataFrame(features_list)
            
            # Add features to main DataFrame
            for col in features_df.columns:
                df_clean[f'text_feature_{col}'] = features_df[col]
            
            # Add language detection
            df_clean['detected_language'] = df_clean[text_column].apply(self.detect_language_simple)
            
            # Filter out very short or empty texts
            initial_rows = len(df_clean)
            df_clean = df_clean[df_clean['text_feature_word_count'] >= 2]
            removed_rows = initial_rows - len(df_clean)
            
            if removed_rows > 0:
                logger.info(f"Removed {removed_rows} rows with insufficient text content")
            
            logger.info(f"Text preprocessing completed. Final dataset: {len(df_clean)} rows")
            return df_clean
            
        except Exception as e:
            logger.error(f"Error cleaning DataFrame: {str(e)}")
            return df
    
    def get_preprocessing_stats(self, original_texts: List[str], processed_texts: List[str]) -> Dict[str, Any]:
        """Get statistics about preprocessing operations"""
        try:
            stats = {
                'total_texts': len(original_texts),
                'avg_length_reduction': 0,
                'avg_word_reduction': 0,
                'empty_after_processing': 0,
                'processing_success_rate': 0
            }
            
            if len(original_texts) != len(processed_texts):
                return stats
            
            length_reductions = []
            word_reductions = []
            successful_processing = 0
            
            for orig, proc in zip(original_texts, processed_texts):
                if isinstance(orig, str) and isinstance(proc, str):
                    orig_len = len(orig)
                    proc_len = len(proc)
                    orig_words = len(orig.split())
                    proc_words = len(proc.split())
                    
                    if orig_len > 0:
                        length_reductions.append((orig_len - proc_len) / orig_len * 100)
                    if orig_words > 0:
                        word_reductions.append((orig_words - proc_words) / orig_words * 100)
                    
                    if proc_len > 0:
                        successful_processing += 1
                    else:
                        stats['empty_after_processing'] += 1
            
            if length_reductions:
                stats['avg_length_reduction'] = round(np.mean(length_reductions), 2)
            if word_reductions:
                stats['avg_word_reduction'] = round(np.mean(word_reductions), 2)
            
            stats['processing_success_rate'] = round((successful_processing / len(original_texts)) * 100, 2)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating preprocessing stats: {str(e)}")
            return {'error': str(e)}
