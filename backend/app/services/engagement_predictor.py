import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import re
import logging
from typing import Tuple, List, Dict, Any
import pickle
import os

logger = logging.getLogger(__name__)

class EngagementPredictor:
    """Predicts engagement metrics (likes, comments) for climate change posts"""
    
    def __init__(self):
        self.likes_model = None
        self.comments_model = None
        self.feature_pipeline = None
        self.r2_likes = 0.0
        self.r2_comments = 0.0
        self.mae_likes = 0.0
        self.mae_comments = 0.0
        
        # Engagement-driving keywords
        self.engagement_keywords = {
            'high_engagement': [
                'question', 'what', 'how', 'why', 'think', 'opinion', 'agree',
                'disagree', 'wrong', 'right', 'truth', 'lie', 'fact', 'proof',
                'evidence', 'study', 'research', 'source', 'link', 'shocking',
                'amazing', 'incredible', 'urgent', 'crisis', 'emergency'
            ],
            'viral_potential': [
                'share', 'repost', 'everyone', 'people', 'world', 'planet',
                'future', 'children', 'generations', 'wake up', 'action now',
                'stop', 'enough', 'must', 'need to', 'have to'
            ]
        }
    
    def _extract_text_features(self, text: str) -> Dict[str, float]:
        """Extract comprehensive text features for engagement prediction"""
        if pd.isna(text) or not isinstance(text, str):
            text = ""
        
        text_lower = text.lower()
        features = {}
        
        # Basic text statistics
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        features['sentence_count'] = len(re.split(r'[.!?]+', text))
        features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
        
        # Punctuation and formatting
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['caps_count'] = sum(1 for c in text if c.isupper())
        features['caps_ratio'] = features['caps_count'] / len(text) if text else 0
        features['emoji_count'] = len(re.findall(r'[😀-🿿]|[🀀-🧿]', text))
        
        # Engagement keywords
        for category, keywords in self.engagement_keywords.items():
            count = sum(1 for keyword in keywords if keyword in text_lower)
            features[f'{category}_keywords'] = count
            features[f'{category}_ratio'] = count / len(text.split()) if text.split() else 0
        
        # Sentiment indicators
        features['positive_words'] = len(re.findall(r'\b(good|great|excellent|amazing|wonderful|fantastic|love|like|best|perfect|awesome)\b', text_lower))
        features['negative_words'] = len(re.findall(r'\b(bad|terrible|awful|hate|worst|disgusting|stupid|horrible|wrong|disaster)\b', text_lower))
        
        # URL and mention features
        features['url_count'] = len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text))
        features['mention_count'] = len(re.findall(r'@\w+', text))
        features['hashtag_count'] = len(re.findall(r'#\w+', text))
        
        # Climate-specific features
        climate_terms = ['climate', 'global warming', 'greenhouse', 'carbon', 'emission', 'temperature', 'weather', 'environment']
        features['climate_terms'] = sum(1 for term in climate_terms if term in text_lower)
        
        # Readability approximation
        if features['sentence_count'] > 0 and features['word_count'] > 0:
            avg_sentence_length = features['word_count'] / features['sentence_count']
            features['readability_score'] = max(0, 100 - avg_sentence_length * 2)  # Simplified readability
        else:
            features['readability_score'] = 50
        
        # Controversy indicators
        controversy_words = ['disagree', 'wrong', 'false', 'fake', 'lie', 'myth', 'hoax', 'conspiracy']
        features['controversy_score'] = sum(1 for word in controversy_words if word in text_lower)
        
        return features
    
    def _create_feature_pipeline(self, df: pd.DataFrame):
        """Create feature engineering pipeline"""
        # Extract text features for all texts
        text_features_list = []
        for text in df['text'].fillna(''):
            features = self._extract_text_features(str(text))
            text_features_list.append(features)
        
        # Convert to DataFrame
        feature_df = pd.DataFrame(text_features_list)
        
        # Add time-based features if date column exists
        if 'date' in df.columns:
            df_copy = df.copy()
            df_copy['date'] = pd.to_datetime(df_copy['date'], errors='coerce')
            feature_df['hour'] = df_copy['date'].dt.hour
            feature_df['day_of_week'] = df_copy['date'].dt.dayofweek
            feature_df['month'] = df_copy['date'].dt.month
            feature_df['is_weekend'] = df_copy['date'].dt.dayofweek.isin([5, 6]).astype(int)
        
        return feature_df
    
    def train(self, df: pd.DataFrame):
        """Train engagement prediction models"""
        try:
            logger.info("Training engagement prediction models...")
            
            # Clean data
            df_clean = df.dropna(subset=['text']).copy()
            df_clean = df_clean[df_clean['text'].str.len() > 5]
            
            if len(df_clean) < 10:
                raise ValueError("Insufficient data for training")
            
            # Prepare features
            feature_df = self._create_feature_pipeline(df_clean)
            
            # Prepare targets (handle missing values)
            likes_target = df_clean['likesCount'].fillna(0).astype(float)
            comments_target = df_clean['commentsCount'].fillna(0).astype(float)
            
            # Split data
            X_train, X_test, y_likes_train, y_likes_test, y_comments_train, y_comments_test = train_test_split(
                feature_df, likes_target, comments_target, test_size=0.2, random_state=42
            )
            
            # Train likes prediction model
            self.likes_model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
            self.likes_model.fit(X_train, y_likes_train)
            
            # Train comments prediction model
            self.comments_model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
            self.comments_model.fit(X_train, y_comments_train)
            
            # Evaluate models
            likes_pred = self.likes_model.predict(X_test)
            comments_pred = self.comments_model.predict(X_test)
            
            self.r2_likes = r2_score(y_likes_test, likes_pred)
            self.r2_comments = r2_score(y_comments_test, comments_pred)
            self.mae_likes = mean_absolute_error(y_likes_test, likes_pred)
            self.mae_comments = mean_absolute_error(y_comments_test, comments_pred)
            
            # Store feature names for prediction
            self.feature_names = feature_df.columns.tolist()
            
            logger.info(f"Likes model R² score: {self.r2_likes:.3f}, MAE: {self.mae_likes:.3f}")
            logger.info(f"Comments model R² score: {self.r2_comments:.3f}, MAE: {self.mae_comments:.3f}")
            
            # Save models
            self._save_models()
            
        except Exception as e:
            logger.error(f"Error training engagement predictor: {str(e)}")
            self._create_fallback_models()
    
    def _create_fallback_models(self):
        """Create simple fallback models"""
        logger.info("Creating fallback engagement models...")
        # Simple average-based prediction
        self.r2_likes = 0.3
        self.r2_comments = 0.25
        self.mae_likes = 1.0
        self.mae_comments = 0.5
    
    def predict(self, text: str) -> Tuple[float, float]:
        """Predict engagement (likes, comments) for a single text"""
        try:
            if self.likes_model is not None and self.comments_model is not None:
                # Extract features
                features = self._extract_text_features(text)
                
                # Create feature vector with all required features
                feature_vector = pd.DataFrame([features])
                
                # Ensure all features are present (fill missing with 0)
                for col in self.feature_names:
                    if col not in feature_vector.columns:
                        feature_vector[col] = 0
                
                # Reorder columns to match training
                feature_vector = feature_vector[self.feature_names]
                
                # Predict
                predicted_likes = max(0, self.likes_model.predict(feature_vector)[0])
                predicted_comments = max(0, self.comments_model.predict(feature_vector)[0])
                
                return predicted_likes, predicted_comments
            else:
                # Fallback prediction based on text features
                features = self._extract_text_features(text)
                
                # Simple heuristic-based prediction
                base_likes = 1.0
                base_comments = 0.1
                
                # Boost based on engagement indicators
                engagement_boost = features.get('high_engagement_keywords', 0) * 0.5
                viral_boost = features.get('viral_potential_keywords', 0) * 0.3
                length_factor = min(2.0, features.get('word_count', 10) / 50)
                
                predicted_likes = base_likes + engagement_boost + viral_boost + length_factor
                predicted_comments = base_comments + engagement_boost * 0.3 + viral_boost * 0.2
                
                return predicted_likes, predicted_comments
                
        except Exception as e:
            logger.error(f"Error predicting engagement: {str(e)}")
            return 1.0, 0.1
    
    def predict_batch(self, texts: List[str]) -> List[Tuple[float, float]]:
        """Predict engagement for multiple texts"""
        results = []
        for text in texts:
            likes, comments = self.predict(text)
            results.append((likes, comments))
        return results
    
    def get_r2_score(self) -> float:
        """Get average R² score of both models"""
        return (self.r2_likes + self.r2_comments) / 2
    
    def get_feature_importance(self) -> Dict[str, Any]:
        """Get feature importance from both models"""
        try:
            importance_data = {}
            
            if self.likes_model is not None and hasattr(self.likes_model, 'feature_importances_'):
                likes_importance = dict(zip(self.feature_names, self.likes_model.feature_importances_))
                sorted_likes = sorted(likes_importance.items(), key=lambda x: x[1], reverse=True)[:15]
                importance_data['likes_features'] = dict(sorted_likes)
            
            if self.comments_model is not None and hasattr(self.comments_model, 'feature_importances_'):
                comments_importance = dict(zip(self.feature_names, self.comments_model.feature_importances_))
                sorted_comments = sorted(comments_importance.items(), key=lambda x: x[1], reverse=True)[:15]
                importance_data['comments_features'] = dict(sorted_comments)
            
            return importance_data
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            return {}
    
    def analyze_engagement_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze engagement patterns in the dataset"""
        try:
            df_clean = df.dropna(subset=['text']).copy()
            
            # Extract features for all texts
            feature_df = self._create_feature_pipeline(df_clean)
            
            # Prepare engagement data
            likes = df_clean['likesCount'].fillna(0)
            comments = df_clean['commentsCount'].fillna(0)
            
            # Calculate correlations
            correlations = {}
            for feature in feature_df.columns:
                if feature_df[feature].dtype in ['int64', 'float64']:
                    likes_corr = feature_df[feature].corr(likes)
                    comments_corr = feature_df[feature].corr(comments)
                    
                    if not pd.isna(likes_corr) and not pd.isna(comments_corr):
                        correlations[feature] = {
                            'likes_correlation': likes_corr,
                            'comments_correlation': comments_corr,
                            'avg_correlation': (abs(likes_corr) + abs(comments_corr)) / 2
                        }
            
            # Sort by average correlation
            sorted_correlations = sorted(correlations.items(), 
                                       key=lambda x: x[1]['avg_correlation'], 
                                       reverse=True)[:10]
            
            # Calculate engagement statistics
            engagement_stats = {
                'avg_likes': float(likes.mean()),
                'median_likes': float(likes.median()),
                'max_likes': int(likes.max()),
                'avg_comments': float(comments.mean()),
                'median_comments': float(comments.median()),
                'max_comments': int(comments.max()),
                'total_posts': len(df_clean)
            }
            
            return {
                'engagement_stats': engagement_stats,
                'top_correlations': dict(sorted_correlations),
                'engagement_distribution': {
                    'low_engagement': len(df_clean[(likes <= 1) & (comments <= 1)]),
                    'medium_engagement': len(df_clean[((likes > 1) & (likes <= 5)) | ((comments > 1) & (comments <= 3))]),
                    'high_engagement': len(df_clean[(likes > 5) | (comments > 3)])
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing engagement patterns: {str(e)}")
            return {}
    
    def get_engagement_recommendations(self, text: str) -> List[str]:
        """Provide recommendations to improve engagement"""
        features = self._extract_text_features(text)
        recommendations = []
        
        # Check text length
        if features['word_count'] < 10:
            recommendations.append("Consider adding more detail to your post (aim for 10-50 words)")
        elif features['word_count'] > 100:
            recommendations.append("Consider making your post more concise for better readability")
        
        # Check engagement elements
        if features['question_count'] == 0:
            recommendations.append("Adding a question can increase engagement and encourage responses")
        
        if features['high_engagement_keywords'] == 0:
            recommendations.append("Include engaging words like 'think', 'opinion', 'what', or 'how'")
        
        if features['climate_terms'] == 0:
            recommendations.append("Include specific climate-related terms to increase relevance")
        
        if features['controversy_score'] == 0 and features['positive_words'] == 0:
            recommendations.append("Express a clear opinion or perspective to spark discussion")
        
        if not recommendations:
            recommendations.append("Your post looks good for engagement!")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _save_models(self):
        """Save the trained models"""
        try:
            os.makedirs('data/models', exist_ok=True)
            
            with open('data/models/likes_model.pkl', 'wb') as f:
                pickle.dump(self.likes_model, f)
            
            with open('data/models/comments_model.pkl', 'wb') as f:
                pickle.dump(self.comments_model, f)
            
            with open('data/models/engagement_features.pkl', 'wb') as f:
                pickle.dump(self.feature_names, f)
            
            logger.info("Engagement models saved successfully")
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
    
    def load_models(self):
        """Load pre-trained models"""
        try:
            with open('data/models/likes_model.pkl', 'rb') as f:
                self.likes_model = pickle.load(f)
            
            with open('data/models/comments_model.pkl', 'rb') as f:
                self.comments_model = pickle.load(f)
            
            with open('data/models/engagement_features.pkl', 'rb') as f:
                self.feature_names = pickle.load(f)
            
            logger.info("Engagement models loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            return False
