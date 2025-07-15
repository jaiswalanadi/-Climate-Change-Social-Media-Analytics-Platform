import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import logging
from typing import Dict, Any, List, Optional
import hashlib

logger = logging.getLogger(__name__)

class DataProcessor:
    """Comprehensive data processing for climate change social media data"""
    
    def __init__(self):
        self.processed_data = None
        self.data_quality_report = {}
    
    def load_dataset(self, file_path: str) -> pd.DataFrame:
        """Load and initially process the dataset"""
        try:
            logger.info(f"Loading dataset from {file_path}")
            
            # Read CSV file
            df = pd.read_csv(file_path)
            
            logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
            
            # Process the dataset
            self.processed_data = self._clean_and_process(df)
            
            # Generate data quality report
            self.data_quality_report = self._generate_quality_report(df, self.processed_data)
            
            return self.processed_data
            
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise
    
    def _clean_and_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Comprehensive data cleaning and processing"""
        try:
            df_clean = df.copy()
            
            # Standardize column names
            df_clean.columns = df_clean.columns.str.lower().str.strip()
            
            # Process date column
            if 'date' in df_clean.columns:
                df_clean['date'] = pd.to_datetime(df_clean['date'], errors='coerce')
                df_clean['year'] = df_clean['date'].dt.year
                df_clean['month'] = df_clean['date'].dt.month
                df_clean['day_of_week'] = df_clean['date'].dt.dayofweek
                df_clean['hour'] = df_clean['date'].dt.hour
                df_clean['is_weekend'] = df_clean['day_of_week'].isin([5, 6]).astype(int)
            
            # Clean text column
            if 'text' in df_clean.columns:
                df_clean['text'] = df_clean['text'].astype(str)
                df_clean['text_clean'] = df_clean['text'].apply(self._clean_text)
                df_clean['text_length'] = df_clean['text'].str.len()
                df_clean['word_count'] = df_clean['text'].str.split().str.len()
                df_clean['has_url'] = df_clean['text'].str.contains(r'http', case=False, na=False)
                df_clean['has_mention'] = df_clean['text'].str.contains(r'@\w+', case=False, na=False)
                df_clean['has_hashtag'] = df_clean['text'].str.contains(r'#\w+', case=False, na=False)
            
            # Process engagement metrics
            if 'likescount' in df_clean.columns:
                df_clean['likescount'] = pd.to_numeric(df_clean['likescount'], errors='coerce').fillna(0)
                df_clean['likes_log'] = np.log1p(df_clean['likescount'])
            
            if 'commentscount' in df_clean.columns:
                df_clean['commentscount'] = pd.to_numeric(df_clean['commentscount'], errors='coerce').fillna(0)
                df_clean['comments_log'] = np.log1p(df_clean['commentscount'])
            
            # Calculate engagement metrics
            if 'likescount' in df_clean.columns and 'commentscount' in df_clean.columns:
                df_clean['total_engagement'] = df_clean['likescount'] + df_clean['commentscount']
                df_clean['engagement_rate'] = df_clean['total_engagement'] / (df_clean['text_length'] + 1)
                
                # Engagement categories
                df_clean['engagement_category'] = pd.cut(
                    df_clean['total_engagement'],
                    bins=[-np.inf, 0, 2, 5, np.inf],
                    labels=['no_engagement', 'low', 'medium', 'high']
                )
            
            # Hash profile names for privacy (if not already hashed)
            if 'profilename' in df_clean.columns:
                # Check if already hashed (64 character hex strings)
                sample_profile = str(df_clean['profilename'].iloc[0]) if len(df_clean) > 0 else ""
                if not (len(sample_profile) == 64 and all(c in '0123456789abcdef' for c in sample_profile)):
                    df_clean['profilename'] = df_clean['profilename'].apply(
                        lambda x: hashlib.sha256(str(x).encode()).hexdigest() if pd.notna(x) else x
                    )
            
            # Remove duplicates
            initial_rows = len(df_clean)
            if 'text' in df_clean.columns:
                df_clean = df_clean.drop_duplicates(subset=['text'], keep='first')
            else:
                df_clean = df_clean.drop_duplicates()
            
            duplicates_removed = initial_rows - len(df_clean)
            if duplicates_removed > 0:
                logger.info(f"Removed {duplicates_removed} duplicate rows")
            
            # Filter out very short or empty texts
            if 'text' in df_clean.columns:
                initial_rows = len(df_clean)
                df_clean = df_clean[df_clean['text_length'] >= 5]
                short_texts_removed = initial_rows - len(df_clean)
                if short_texts_removed > 0:
                    logger.info(f"Removed {short_texts_removed} very short texts")
            
            logger.info(f"Data processing completed. Final dataset: {len(df_clean)} rows")
            return df_clean
            
        except Exception as e:
            logger.error(f"Error in data cleaning: {str(e)}")
            return df
    
    def _clean_text(self, text: str) -> str:
        """Clean individual text entries"""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove zero-width characters
        text = re.sub(r'[\u200b-\u200d\ufeff]', '', text)
        
        # Fix common encoding issues
        text = text.replace('â€™', "'").replace('â€œ', '"').replace('â€', '"')
        
        return text
    
    def _generate_quality_report(self, original_df: pd.DataFrame, processed_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive data quality report"""
        try:
            report = {
                'original_rows': len(original_df),
                'processed_rows': len(processed_df),
                'rows_removed': len(original_df) - len(processed_df),
                'columns_original': len(original_df.columns),
                'columns_processed': len(processed_df.columns),
                'missing_values': {},
                'data_types': {},
                'date_range': {},
                'text_statistics': {},
                'engagement_statistics': {}
            }
            
            # Missing values analysis
            for col in processed_df.columns:
                missing_count = processed_df[col].isna().sum()
                missing_pct = (missing_count / len(processed_df)) * 100
                report['missing_values'][col] = {
                    'count': int(missing_count),
                    'percentage': round(missing_pct, 2)
                }
            
            # Data types
            report['data_types'] = {col: str(dtype) for col, dtype in processed_df.dtypes.items()}
            
            # Date range analysis
            if 'date' in processed_df.columns:
                valid_dates = processed_df['date'].dropna()
                if len(valid_dates) > 0:
                    report['date_range'] = {
                        'start_date': valid_dates.min().isoformat(),
                        'end_date': valid_dates.max().isoformat(),
                        'date_span_days': (valid_dates.max() - valid_dates.min()).days,
                        'valid_dates': len(valid_dates),
                        'invalid_dates': len(processed_df) - len(valid_dates)
                    }
            
            # Text statistics
            if 'text' in processed_df.columns:
                text_data = processed_df['text'].dropna()
                if len(text_data) > 0:
                    report['text_statistics'] = {
                        'total_comments': len(text_data),
                        'avg_length': round(text_data.str.len().mean(), 2),
                        'median_length': int(text_data.str.len().median()),
                        'min_length': int(text_data.str.len().min()),
                        'max_length': int(text_data.str.len().max()),
                        'avg_words': round(text_data.str.split().str.len().mean(), 2),
                        'empty_texts': int((text_data.str.len() == 0).sum()),
                        'has_urls': int(text_data.str.contains(r'http', case=False, na=False).sum()),
                        'has_mentions': int(text_data.str.contains(r'@\w+', case=False, na=False).sum()),
                        'has_hashtags': int(text_data.str.contains(r'#\w+', case=False, na=False).sum())
                    }
            
            # Engagement statistics
            if 'likescount' in processed_df.columns:
                likes_data = processed_df['likescount'].dropna()
                if len(likes_data) > 0:
                    report['engagement_statistics']['likes'] = {
                        'total_likes': int(likes_data.sum()),
                        'avg_likes': round(likes_data.mean(), 2),
                        'median_likes': int(likes_data.median()),
                        'max_likes': int(likes_data.max()),
                        'zero_likes': int((likes_data == 0).sum())
                    }
            
            if 'commentscount' in processed_df.columns:
                comments_data = processed_df['commentscount'].dropna()
                if len(comments_data) > 0:
                    report['engagement_statistics']['comments'] = {
                        'total_comments': int(comments_data.sum()),
                        'avg_comments': round(comments_data.mean(), 2),
                        'median_comments': int(comments_data.median()),
                        'max_comments': int(comments_data.max()),
                        'zero_comments': int((comments_data == 0).sum())
                    }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating quality report: {str(e)}")
            return {}
    
    def get_dataset_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get comprehensive dataset summary"""
        try:
            summary = {
                'basic_info': {
                    'total_rows': len(df),
                    'total_columns': len(df.columns),
                    'columns': list(df.columns),
                    'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)
                },
                'data_quality': self.data_quality_report,
                'temporal_analysis': {},
                'content_analysis': {},
                'engagement_analysis': {}
            }
            
            # Temporal analysis
            if 'date' in df.columns:
                date_data = df['date'].dropna()
                if len(date_data) > 0:
                    # Posts per month
                    monthly_posts = date_data.dt.to_period('M').value_counts().sort_index()
                    
                    # Posts per day of week
                    dow_posts = date_data.dt.day_name().value_counts()
                    
                    # Posts per hour
                    hourly_posts = date_data.dt.hour.value_counts().sort_index()
                    
                    summary['temporal_analysis'] = {
                        'posts_per_month': monthly_posts.to_dict(),
                        'posts_per_day_of_week': dow_posts.to_dict(),
                        'posts_per_hour': hourly_posts.to_dict(),
                        'most_active_month': str(monthly_posts.idxmax()) if len(monthly_posts) > 0 else None,
                        'most_active_day': dow_posts.idxmax() if len(dow_posts) > 0 else None,
                        'most_active_hour': int(hourly_posts.idxmax()) if len(hourly_posts) > 0 else None
                    }
            
            # Content analysis
            if 'text' in df.columns:
                text_data = df['text'].dropna()
                if len(text_data) > 0:
                    # Most common words (simple analysis)
                    all_words = ' '.join(text_data.astype(str)).lower().split()
                    word_freq = pd.Series(all_words).value_counts().head(20)
                    
                    # Climate keywords frequency
                    climate_keywords = ['climate', 'global', 'warming', 'carbon', 'emission', 'temperature', 
                                      'greenhouse', 'fossil', 'renewable', 'energy', 'pollution', 'environment']
                    keyword_freq = {}
                    for keyword in climate_keywords:
                        count = text_data.str.lower().str.contains(keyword, na=False).sum()
                        if count > 0:
                            keyword_freq[keyword] = int(count)
                    
                    summary['content_analysis'] = {
                        'top_words': word_freq.to_dict(),
                        'climate_keywords_frequency': keyword_freq,
                        'avg_words_per_comment': round(text_data.str.split().str.len().mean(), 2),
                        'languages_detected': self._detect_languages(text_data.head(100))
                    }
            
            # Engagement analysis
            if 'likescount' in df.columns or 'commentscount' in df.columns:
                engagement_data = {}
                
                if 'likescount' in df.columns:
                    likes = df['likescount'].fillna(0)
                    engagement_data['likes_distribution'] = {
                        'zero_likes': int((likes == 0).sum()),
                        'low_likes_1_2': int(((likes >= 1) & (likes <= 2)).sum()),
                        'medium_likes_3_5': int(((likes >= 3) & (likes <= 5)).sum()),
                        'high_likes_6_plus': int((likes >= 6).sum())
                    }
                
                if 'commentscount' in df.columns:
                    comments = df['commentscount'].fillna(0)
                    engagement_data['comments_distribution'] = {
                        'zero_comments': int((comments == 0).sum()),
                        'low_comments_1': int(comments == 1),
                        'medium_comments_2_3': int(((comments >= 2) & (comments <= 3)).sum()),
                        'high_comments_4_plus': int((comments >= 4).sum())
                    }
                
                # Top performing posts
                if 'text' in df.columns and 'likescount' in df.columns:
                    top_posts = df.nlargest(5, 'likescount')[['text', 'likescount', 'commentscount']].to_dict('records')
                    engagement_data['top_performing_posts'] = [
                        {
                            'text_preview': post['text'][:100] + "..." if len(post['text']) > 100 else post['text'],
                            'likes': int(post['likescount']) if pd.notna(post['likescount']) else 0,
                            'comments': int(post['commentscount']) if pd.notna(post['commentscount']) else 0
                        }
                        for post in top_posts
                    ]
                
                summary['engagement_analysis'] = engagement_data
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating dataset summary: {str(e)}")
            return {'error': str(e)}
    
    def _detect_languages(self, text_sample: pd.Series) -> Dict[str, int]:
        """Simple language detection based on common words"""
        try:
            # Simple heuristic-based language detection
            languages = {'english': 0, 'other': 0}
            
            english_indicators = ['the', 'and', 'is', 'in', 'to', 'of', 'a', 'that', 'it', 'with', 'for', 'as', 'was', 'on', 'are']
            
            for text in text_sample:
                if pd.notna(text) and isinstance(text, str):
                    words = text.lower().split()
                    english_word_count = sum(1 for word in words if word in english_indicators)
                    
                    if english_word_count > 0:
                        languages['english'] += 1
                    else:
                        languages['other'] += 1
            
            return languages
            
        except Exception as e:
            logger.error(f"Error in language detection: {str(e)}")
            return {'english': 1, 'other': 0}
    
    def export_processed_data(self, output_path: str, format: str = 'csv'):
        """Export processed data to file"""
        try:
            if self.processed_data is None:
                raise ValueError("No processed data available")
            
            if format.lower() == 'csv':
                self.processed_data.to_csv(output_path, index=False)
            elif format.lower() == 'json':
                self.processed_data.to_json(output_path, orient='records', date_format='iso')
            elif format.lower() == 'excel':
                self.processed_data.to_excel(output_path, index=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Data exported to {output_path} in {format} format")
            
        except Exception as e:
            logger.error(f"Error exporting data: {str(e)}")
            raise
    
    def get_data_quality_score(self) -> float:
        """Calculate overall data quality score (0-100)"""
        try:
            if not self.data_quality_report:
                return 0.0
            
            score = 100.0
            
            # Deduct points for missing data
            for col, missing_info in self.data_quality_report.get('missing_values', {}).items():
                missing_pct = missing_info.get('percentage', 0)
                if missing_pct > 50:
                    score -= 15
                elif missing_pct > 25:
                    score -= 10
                elif missing_pct > 10:
                    score -= 5
            
            # Deduct points for removed rows
            rows_removed_pct = (self.data_quality_report.get('rows_removed', 0) / 
                              max(self.data_quality_report.get('original_rows', 1), 1)) * 100
            if rows_removed_pct > 20:
                score -= 10
            elif rows_removed_pct > 10:
                score -= 5
            
            # Bonus for good text quality
            text_stats = self.data_quality_report.get('text_statistics', {})
            if text_stats:
                avg_length = text_stats.get('avg_length', 0)
                if avg_length > 50:  # Good average text length
                    score += 5
                
                if text_stats.get('empty_texts', 0) == 0:  # No empty texts
                    score += 5
            
            return max(0.0, min(100.0, score))
            
        except Exception as e:
            logger.error(f"Error calculating data quality score: {str(e)}")
            return 50.0
