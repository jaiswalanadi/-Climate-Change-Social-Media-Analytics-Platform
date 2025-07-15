import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from textblob import TextBlob
import re
import logging
from typing import Tuple, List, Dict, Any
import pickle
import os

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """Advanced sentiment analyzer for climate change comments"""
    
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.pipeline = None
        self.accuracy = 0.0
        self.label_mapping = {
            'negative': 0,
            'neutral': 1, 
            'positive': 2
        }
        self.reverse_label_mapping = {v: k for k, v in self.label_mapping.items()}
        
        # Climate-specific keywords for enhanced analysis
        self.climate_keywords = {
            'positive': [
                'sustainable', 'renewable', 'clean energy', 'green', 'conservation',
                'solution', 'progress', 'innovative', 'hope', 'future', 'protect',
                'save', 'improve', 'support', 'action', 'reduce emissions'
            ],
            'negative': [
                'disaster', 'crisis', 'dangerous', 'threat', 'catastrophe',
                'destroy', 'damage', 'pollution', 'extinction', 'emergency',
                'deny', 'fake', 'hoax', 'worse', 'terrible', 'apocalypse'
            ],
            'neutral': [
                'data', 'research', 'study', 'analysis', 'temperature',
                'measurement', 'observation', 'fact', 'science', 'report'
            ]
        }
    
    def _extract_climate_features(self, text: str) -> Dict[str, float]:
        """Extract climate-specific features from text"""
        text_lower = text.lower()
        features = {}
        
        # Count climate keywords
        for sentiment, keywords in self.climate_keywords.items():
            count = sum(1 for keyword in keywords if keyword in text_lower)
            features[f'climate_{sentiment}_count'] = count
            features[f'climate_{sentiment}_ratio'] = count / len(text.split()) if text.split() else 0
        
        # Text statistics
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['caps_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        
        return features
    
    def _get_textblob_sentiment(self, text: str) -> Tuple[str, float]:
        """Get sentiment using TextBlob as baseline"""
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            if polarity > 0.1:
                return 'positive', polarity
            elif polarity < -0.1:
                return 'negative', polarity
            else:
                return 'neutral', polarity
        except:
            return 'neutral', 0.0
    
    def _create_sentiment_labels(self, df: pd.DataFrame) -> pd.Series:
        """Create sentiment labels using multiple heuristics"""
        labels = []
        
        for _, row in df.iterrows():
            text = str(row.get('text', ''))
            likes = row.get('likesCount', 0)
            comments = row.get('commentsCount', 0)
            
            # Get TextBlob sentiment
            tb_sentiment, tb_score = self._get_textblob_sentiment(text)
            
            # Extract climate features
            climate_features = self._extract_climate_features(text)
            
            # Enhanced sentiment determination
            pos_indicators = climate_features['climate_positive_count']
            neg_indicators = climate_features['climate_negative_count']
            neu_indicators = climate_features['climate_neutral_count']
            
            # Combine multiple signals
            if pos_indicators > neg_indicators and (tb_sentiment == 'positive' or likes > 2):
                labels.append('positive')
            elif neg_indicators > pos_indicators and (tb_sentiment == 'negative' or '!' in text):
                labels.append('negative')
            elif neu_indicators > 0 or (tb_sentiment == 'neutral' and abs(tb_score) < 0.05):
                labels.append('neutral')
            else:
                # Fall back to TextBlob
                labels.append(tb_sentiment)
        
        return pd.Series(labels)
    
    def train(self, df: pd.DataFrame):
        """Train the sentiment analysis model"""
        try:
            logger.info("Training sentiment analyzer...")
            
            # Clean data
            df_clean = df.dropna(subset=['text']).copy()
            df_clean = df_clean[df_clean['text'].str.len() > 5]  # Filter very short texts
            
            if len(df_clean) < 10:
                raise ValueError("Insufficient data for training")
            
            # Create sentiment labels
            df_clean['sentiment'] = self._create_sentiment_labels(df_clean)
            
            # Prepare features
            X = df_clean['text'].astype(str)
            y = df_clean['sentiment'].map(self.label_mapping)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Create pipeline with TF-IDF and Random Forest
            self.pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(
                    max_features=5000,
                    ngram_range=(1, 2),
                    stop_words='english',
                    lowercase=True,
                    min_df=2,
                    max_df=0.8
                )),
                ('classifier', RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    max_depth=10,
                    class_weight='balanced'
                ))
            ])
            
            # Train model
            self.pipeline.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = self.pipeline.predict(X_test)
            self.accuracy = accuracy_score(y_test, y_pred)
            
            logger.info(f"Model trained with accuracy: {self.accuracy:.3f}")
            logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
            
            # Save model
            self._save_model()
            
        except Exception as e:
            logger.error(f"Error training sentiment analyzer: {str(e)}")
            # Fallback to simple model
            self._create_fallback_model()
    
    def _create_fallback_model(self):
        """Create a simple fallback model"""
        logger.info("Creating fallback sentiment model...")
        self.accuracy = 0.6  # Conservative estimate
    
    def predict(self, text: str) -> Tuple[str, float]:
        """Predict sentiment for a single text"""
        try:
            if self.pipeline is not None:
                # Use trained model
                prediction = self.pipeline.predict([text])[0]
                prediction_proba = self.pipeline.predict_proba([text])[0]
                
                sentiment = self.reverse_label_mapping[prediction]
                confidence = float(np.max(prediction_proba))
                
                return sentiment, confidence
            else:
                # Use TextBlob as fallback
                return self._get_textblob_sentiment(text)
                
        except Exception as e:
            logger.error(f"Error predicting sentiment: {str(e)}")
            return 'neutral', 0.5
    
    def predict_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        """Predict sentiment for multiple texts"""
        results = []
        for text in texts:
            sentiment, confidence = self.predict(text)
            results.append((sentiment, confidence))
        return results
    
    def get_accuracy(self) -> float:
        """Get model accuracy"""
        return self.accuracy
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the model"""
        try:
            if self.pipeline is not None:
                # Get feature names
                feature_names = self.pipeline['tfidf'].get_feature_names_out()
                # Get importance scores
                importance = self.pipeline['classifier'].feature_importances_
                
                # Create feature importance dictionary
                feature_importance = dict(zip(feature_names, importance))
                # Sort and return top 20
                sorted_features = sorted(feature_importance.items(), 
                                       key=lambda x: x[1], reverse=True)[:20]
                return dict(sorted_features)
            return {}
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            return {}
    
    def analyze_sentiment_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze sentiment trends over time"""
        try:
            if 'date' not in df.columns or 'text' not in df.columns:
                return {}
            
            df_clean = df.dropna(subset=['text', 'date']).copy()
            
            # Predict sentiments
            sentiments = [self.predict(text)[0] for text in df_clean['text']]
            df_clean['predicted_sentiment'] = sentiments
            
            # Convert date column
            df_clean['date'] = pd.to_datetime(df_clean['date'])
            df_clean['month'] = df_clean['date'].dt.to_period('M')
            
            # Calculate trends
            monthly_sentiment = df_clean.groupby(['month', 'predicted_sentiment']).size().unstack(fill_value=0)
            
            # Calculate percentages
            monthly_sentiment_pct = monthly_sentiment.div(monthly_sentiment.sum(axis=1), axis=0) * 100
            
            return {
                'monthly_counts': monthly_sentiment.to_dict(),
                'monthly_percentages': monthly_sentiment_pct.to_dict(),
                'overall_sentiment_distribution': df_clean['predicted_sentiment'].value_counts().to_dict()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment trends: {str(e)}")
            return {}
    
    def _save_model(self):
        """Save the trained model"""
        try:
            os.makedirs('data/models', exist_ok=True)
            with open('data/models/sentiment_model.pkl', 'wb') as f:
                pickle.dump(self.pipeline, f)
            logger.info("Sentiment model saved successfully")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
    
    def load_model(self):
        """Load a pre-trained model"""
        try:
            with open('data/models/sentiment_model.pkl', 'rb') as f:
                self.pipeline = pickle.load(f)
            logger.info("Sentiment model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
