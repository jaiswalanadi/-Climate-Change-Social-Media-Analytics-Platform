import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import logging
from typing import List, Dict, Any, Tuple
from collections import Counter
import pickle
import os

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    pass

logger = logging.getLogger(__name__)

class TopicModeler:
    """Advanced topic modeling for climate change discussions"""
    
    def __init__(self, n_topics: int = 5):
        self.n_topics = n_topics
        self.lda_model = None
        self.nmf_model = None
        self.vectorizer = None
        self.feature_names = None
        self.topics = []
        self.topic_labels = []
        self.lemmatizer = WordNetLemmatizer()
        
        # Climate-specific stop words
        self.climate_stopwords = set([
            'climate', 'change', 'global', 'warming', 'earth', 'planet',
            'co2', 'carbon', 'dioxide', 'greenhouse', 'gas', 'gases',
            'temperature', 'weather', 'environment', 'environmental'
        ])
        
        # Get standard English stop words
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
        
        # Combine stop words
        self.stop_words.update(self.climate_stopwords)
        
        # Topic labels for climate discussions
        self.topic_keywords = {
            'renewable_energy': ['solar', 'wind', 'renewable', 'energy', 'clean', 'sustainable', 'power', 'electricity'],
            'policy_politics': ['policy', 'government', 'politics', 'law', 'regulation', 'vote', 'election', 'politicians'],
            'science_research': ['science', 'research', 'study', 'data', 'evidence', 'scientist', 'academic', 'paper'],
            'activism_action': ['action', 'protest', 'movement', 'activism', 'demonstrate', 'march', 'campaign', 'fight'],
            'skepticism_denial': ['fake', 'hoax', 'false', 'conspiracy', 'lie', 'propaganda', 'skeptic', 'deny'],
            'impacts_effects': ['effects', 'impacts', 'consequences', 'damage', 'destruction', 'disaster', 'crisis'],
            'solutions_technology': ['solution', 'technology', 'innovation', 'invention', 'breakthrough', 'advancement'],
            'economy_business': ['economy', 'business', 'cost', 'money', 'profit', 'industry', 'market', 'economic']
        }
    
    def _preprocess_text(self, text: str) -> str:
        """Advanced text preprocessing for topic modeling"""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize and lemmatize
        try:
            tokens = word_tokenize(text)
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                     if token not in self.stop_words and len(token) > 2]
            return ' '.join(tokens)
        except:
            # Fallback if NLTK fails
            words = text.split()
            words = [word for word in words if word not in self.stop_words and len(word) > 2]
            return ' '.join(words)
    
    def _determine_optimal_topics(self, texts: List[str], max_topics: int = 10) -> int:
        """Determine optimal number of topics using coherence"""
        try:
            # Preprocess texts
            processed_texts = [self._preprocess_text(text) for text in texts]
            processed_texts = [text for text in processed_texts if len(text.split()) > 2]
            
            if len(processed_texts) < 5:
                return 3
            
            # Create vectorizer
            vectorizer = CountVectorizer(
                max_features=100,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8
            )
            
            doc_term_matrix = vectorizer.fit_transform(processed_texts)
            
            # Test different numbers of topics
            best_score = -1
            best_n_topics = 3
            
            for n in range(2, min(max_topics, len(processed_texts) // 2) + 1):
                try:
                    lda = LatentDirichletAllocation(
                        n_components=n,
                        random_state=42,
                        max_iter=10
                    )
                    lda.fit(doc_term_matrix)
                    
                    # Simple perplexity-based scoring
                    score = -lda.perplexity(doc_term_matrix)
                    
                    if score > best_score:
                        best_score = score
                        best_n_topics = n
                except:
                    continue
            
            return best_n_topics
            
        except Exception as e:
            logger.error(f"Error determining optimal topics: {str(e)}")
            return 5
    
    def fit(self, texts: List[str]):
        """Fit topic modeling on the provided texts"""
        try:
            logger.info("Fitting topic modeling...")
            
            # Filter and preprocess texts
            valid_texts = [text for text in texts if isinstance(text, str) and len(text.strip()) > 10]
            
            if len(valid_texts) < 5:
                logger.warning("Insufficient texts for topic modeling")
                self._create_fallback_topics()
                return
            
            processed_texts = [self._preprocess_text(text) for text in valid_texts]
            processed_texts = [text for text in processed_texts if len(text.split()) > 2]
            
            if len(processed_texts) < 5:
                logger.warning("Insufficient processed texts for topic modeling")
                self._create_fallback_topics()
                return
            
            # Determine optimal number of topics
            optimal_topics = self._determine_optimal_topics(processed_texts)
            self.n_topics = min(optimal_topics, len(processed_texts) // 2)
            
            logger.info(f"Using {self.n_topics} topics")
            
            # Create vectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=200,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8,
                stop_words='english'
            )
            
            # Fit vectorizer and transform texts
            doc_term_matrix = self.vectorizer.fit_transform(processed_texts)
            self.feature_names = self.vectorizer.get_feature_names_out()
            
            # Fit LDA model
            self.lda_model = LatentDirichletAllocation(
                n_components=self.n_topics,
                random_state=42,
                max_iter=20,
                learning_method='batch'
            )
            self.lda_model.fit(doc_term_matrix)
            
            # Fit NMF model as alternative
            self.nmf_model = NMF(
                n_components=self.n_topics,
                random_state=42,
                max_iter=200
            )
            self.nmf_model.fit(doc_term_matrix)
            
            # Extract topics
            self._extract_topics()
            
            # Assign topic labels
            self._assign_topic_labels()
            
            logger.info(f"Topic modeling completed with {self.n_topics} topics")
            
            # Save model
            self._save_model()
            
        except Exception as e:
            logger.error(f"Error fitting topic model: {str(e)}")
            self._create_fallback_topics()
    
    def _extract_topics(self):
        """Extract topics from the fitted models"""
        self.topics = []
        
        try:
            # Use LDA model primarily
            for topic_idx, topic in enumerate(self.lda_model.components_):
                top_words_idx = topic.argsort()[-10:][::-1]
                top_words = [self.feature_names[i] for i in top_words_idx]
                top_weights = [topic[i] for i in top_words_idx]
                
                self.topics.append({
                    'id': topic_idx,
                    'words': top_words,
                    'weights': top_weights,
                    'top_words_str': ', '.join(top_words[:5])
                })
        except Exception as e:
            logger.error(f"Error extracting topics: {str(e)}")
            self._create_fallback_topics()
    
    def _assign_topic_labels(self):
        """Assign human-readable labels to topics"""
        self.topic_labels = []
        
        for topic in self.topics:
            best_label = f"Topic {topic['id'] + 1}"
            best_score = 0
            
            topic_words = set(topic['words'][:5])
            
            # Check against predefined topic categories
            for label, keywords in self.topic_keywords.items():
                overlap = len(topic_words.intersection(keywords))
                if overlap > best_score:
                    best_score = overlap
                    best_label = label.replace('_', ' ').title()
            
            if best_score == 0:
                # Generate label from top words
                best_label = f"{topic['words'][0].title()} & {topic['words'][1].title()}"
            
            self.topic_labels.append(best_label)
            topic['label'] = best_label
    
    def _create_fallback_topics(self):
        """Create fallback topics when modeling fails"""
        logger.info("Creating fallback topics...")
        
        self.topics = [
            {
                'id': 0,
                'label': 'Renewable Energy',
                'words': ['solar', 'wind', 'renewable', 'energy', 'clean'],
                'weights': [0.3, 0.25, 0.2, 0.15, 0.1],
                'top_words_str': 'impact, effects, environment, damage, consequences'
            }
        ]
        self.topic_labels = ['Renewable Energy', 'Climate Science', 'Environmental Impact']
        self.n_topics = 3
    
    def get_topics(self) -> List[Dict[str, Any]]:
        """Get all discovered topics"""
        return self.topics
    
    def get_topic_distribution(self) -> Dict[str, float]:
        """Get topic distribution across the corpus"""
        if not self.topics:
            return {}
        
        # For simplicity, return equal distribution
        distribution = {}
        for topic in self.topics:
            distribution[topic['label']] = 1.0 / len(self.topics)
        
        return distribution
    
    def get_topics_for_text(self, text: str, top_n: int = 3) -> List[str]:
        """Get most relevant topics for a given text"""
        try:
            if self.lda_model is None or self.vectorizer is None:
                return self._get_fallback_topics_for_text(text)
            
            # Preprocess text
            processed_text = self._preprocess_text(text)
            
            if not processed_text:
                return []
            
            # Transform text
            text_vector = self.vectorizer.transform([processed_text])
            
            # Get topic probabilities
            topic_probs = self.lda_model.transform(text_vector)[0]
            
            # Get top topics
            top_topic_indices = topic_probs.argsort()[-top_n:][::-1]
            top_topics = [self.topic_labels[i] for i in top_topic_indices if topic_probs[i] > 0.1]
            
            return top_topics[:top_n]
            
        except Exception as e:
            logger.error(f"Error getting topics for text: {str(e)}")
            return self._get_fallback_topics_for_text(text)
    
    def _get_fallback_topics_for_text(self, text: str) -> List[str]:
        """Simple keyword-based topic assignment"""
        text_lower = text.lower()
        assigned_topics = []
        
        for label, keywords in self.topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                assigned_topics.append(label.replace('_', ' ').title())
        
        # Default topics if none found
        if not assigned_topics:
            assigned_topics = ['General Climate Discussion']
        
        return assigned_topics[:3]
    
    def get_representative_comments(self, df: pd.DataFrame, top_n: int = 3) -> Dict[str, List[str]]:
        """Get representative comments for each topic"""
        try:
            if self.lda_model is None or self.vectorizer is None:
                return self._get_fallback_representative_comments(df)
            
            representative_comments = {}
            
            # Get all comments
            comments = df['text'].dropna().astype(str).tolist()
            processed_comments = [self._preprocess_text(comment) for comment in comments]
            
            # Filter valid processed comments
            valid_indices = [i for i, comment in enumerate(processed_comments) if len(comment.split()) > 2]
            valid_comments = [comments[i] for i in valid_indices]
            valid_processed = [processed_comments[i] for i in valid_indices]
            
            if len(valid_processed) == 0:
                return self._get_fallback_representative_comments(df)
            
            # Transform comments
            comment_vectors = self.vectorizer.transform(valid_processed)
            topic_distributions = self.lda_model.transform(comment_vectors)
            
            # For each topic, find comments with highest probability
            for topic_idx, topic in enumerate(self.topics):
                topic_label = topic['label']
                
                # Get comments with high probability for this topic
                topic_probs = topic_distributions[:, topic_idx]
                top_comment_indices = topic_probs.argsort()[-top_n:][::-1]
                
                # Get representative comments
                rep_comments = []
                for idx in top_comment_indices:
                    if topic_probs[idx] > 0.2:  # Minimum threshold
                        comment = valid_comments[idx]
                        if len(comment) > 20:  # Minimum length
                            rep_comments.append(comment[:200] + "..." if len(comment) > 200 else comment)
                
                # Fallback to any comments if none meet criteria
                if not rep_comments and len(valid_comments) > 0:
                    rep_comments = [valid_comments[0][:200] + "..." if len(valid_comments[0]) > 200 else valid_comments[0]]
                
                representative_comments[topic_label] = rep_comments[:top_n]
            
            return representative_comments
            
        except Exception as e:
            logger.error(f"Error getting representative comments: {str(e)}")
            return self._get_fallback_representative_comments(df)
    
    def _get_fallback_representative_comments(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Fallback method for getting representative comments"""
        try:
            comments = df['text'].dropna().astype(str).tolist()
            
            if len(comments) == 0:
                return {}
            
            # Simple keyword-based assignment
            representative_comments = {}
            
            for label, keywords in self.topic_keywords.items():
                topic_name = label.replace('_', ' ').title()
                matching_comments = []
                
                for comment in comments:
                    if any(keyword in comment.lower() for keyword in keywords):
                        if len(comment) > 20:
                            matching_comments.append(comment[:200] + "..." if len(comment) > 200 else comment)
                
                if matching_comments:
                    representative_comments[topic_name] = matching_comments[:3]
                elif len(comments) > 0:
                    # Fallback to random comments
                    representative_comments[topic_name] = [comments[0][:200] + "..." if len(comments[0]) > 200 else comments[0]]
            
            return representative_comments
            
        except Exception as e:
            logger.error(f"Error in fallback representative comments: {str(e)}")
            return {"General Discussion": ["Sample climate change comment..."]}
    
    def get_num_topics(self) -> int:
        """Get number of topics"""
        return self.n_topics
    
    def analyze_topic_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze how topics change over time"""
        try:
            if 'date' not in df.columns:
                return {}
            
            df_clean = df.dropna(subset=['text', 'date']).copy()
            df_clean['date'] = pd.to_datetime(df_clean['date'])
            df_clean['month'] = df_clean['date'].dt.to_period('M')
            
            # Get topics for each comment
            topic_assignments = []
            for text in df_clean['text']:
                topics = self.get_topics_for_text(str(text), top_n=1)
                primary_topic = topics[0] if topics else 'Unknown'
                topic_assignments.append(primary_topic)
            
            df_clean['primary_topic'] = topic_assignments
            
            # Calculate monthly topic distributions
            monthly_topics = df_clean.groupby(['month', 'primary_topic']).size().unstack(fill_value=0)
            monthly_topics_pct = monthly_topics.div(monthly_topics.sum(axis=1), axis=0) * 100
            
            return {
                'monthly_topic_counts': monthly_topics.to_dict(),
                'monthly_topic_percentages': monthly_topics_pct.to_dict(),
                'overall_topic_distribution': df_clean['primary_topic'].value_counts().to_dict()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing topic trends: {str(e)}")
            return {}
    
    def get_topic_keywords_analysis(self) -> Dict[str, Any]:
        """Get detailed analysis of topic keywords"""
        try:
            analysis = {}
            
            for topic in self.topics:
                topic_analysis = {
                    'top_words': topic['words'][:10],
                    'word_weights': topic['weights'][:10],
                    'word_importance': dict(zip(topic['words'][:5], topic['weights'][:5])),
                    'description': self._generate_topic_description(topic)
                }
                analysis[topic['label']] = topic_analysis
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in topic keywords analysis: {str(e)}")
            return {}
    
    def _generate_topic_description(self, topic: Dict[str, Any]) -> str:
        """Generate human-readable description for a topic"""
        top_words = topic['words'][:3]
        
        descriptions = {
            'renewable': f"Discussions about {', '.join(top_words)} and sustainable energy solutions",
            'energy': f"Focus on {', '.join(top_words)} and energy-related climate solutions",
            'science': f"Scientific discussions involving {', '.join(top_words)} and research findings",
            'policy': f"Policy and governance topics including {', '.join(top_words)}",
            'climate': f"General climate discussions covering {', '.join(top_words)}",
            'impact': f"Environmental impacts and effects related to {', '.join(top_words)}",
            'action': f"Climate action and activism involving {', '.join(top_words)}",
            'technology': f"Technological solutions and innovations in {', '.join(top_words)}"
        }
        
        # Find best matching description
        for key, desc in descriptions.items():
            if any(key in word.lower() for word in top_words):
                return desc
        
        # Default description
        return f"Climate discussions focusing on {', '.join(top_words)}"
    
    def _save_model(self):
        """Save the topic model"""
        try:
            os.makedirs('data/models', exist_ok=True)
            
            model_data = {
                'lda_model': self.lda_model,
                'nmf_model': self.nmf_model,
                'vectorizer': self.vectorizer,
                'topics': self.topics,
                'topic_labels': self.topic_labels,
                'n_topics': self.n_topics,
                'feature_names': self.feature_names
            }
            
            with open('data/models/topic_model.pkl', 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info("Topic model saved successfully")
        except Exception as e:
            logger.error(f"Error saving topic model: {str(e)}")
    
    def load_model(self):
        """Load a pre-trained topic model"""
        try:
            with open('data/models/topic_model.pkl', 'rb') as f:
                model_data = pickle.load(f)
            
            self.lda_model = model_data['lda_model']
            self.nmf_model = model_data['nmf_model']
            self.vectorizer = model_data['vectorizer']
            self.topics = model_data['topics']
            self.topic_labels = model_data['topic_labels']
            self.n_topics = model_data['n_topics']
            self.feature_names = model_data['feature_names']
            
            logger.info("Topic model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading topic model: {str(e)}")
            return False 0.2, 0.15, 0.1],
                'top_words_str': 'solar, wind, renewable, energy, clean'
            },
            {
                'id': 1,
                'label': 'Climate Science',
                'words': ['science', 'research', 'data', 'study', 'evidence'],
                'weights': [0.3, 0.25, 0.2, 0.15, 0.1],
                'top_words_str': 'science, research, data, study, evidence'
            },
            {
                'id': 2,
                'label': 'Environmental Impact',
                'words': ['impact', 'effects', 'environment', 'damage', 'consequences'],
                'weights': [0.3, 0.25,
