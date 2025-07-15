import axios from 'axios';

// API Configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000, // 30 seconds
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
api.interceptors.request.use(
  (config) => {
    console.log(`Making ${config.method?.toUpperCase()} request to: ${config.url}`);
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor
api.interceptors.response.use(
  (response) => {
    return response.data;
  },
  (error) => {
    console.error('API Error:', error.response?.data || error.message);
    
    if (error.response?.status === 500) {
      throw new Error('Server error. Please try again later.');
    }
    
    if (error.response?.status === 404) {
      throw new Error('Requested resource not found.');
    }
    
    if (error.code === 'ECONNABORTED') {
      throw new Error('Request timeout. Please try again.');
    }
    
    throw error.response?.data || error;
  }
);

// Type definitions
export interface CommentAnalysisRequest {
  text: string;
}

export interface CommentAnalysisResponse {
  text: string;
  sentiment: string;
  sentiment_score: number;
  predicted_likes: number;
  predicted_comments: number;
  topics: string[];
  processed_at: string;
}

export interface BatchAnalysisRequest {
  comments: string[];
}

export interface TopicModelingResponse {
  topics: Array<{
    id: number;
    label: string;
    words: string[];
    weights: number[];
    top_words_str: string;
  }>;
  topic_distribution: Record<string, number>;
  representative_comments: Record<string, string[]>;
}

export interface ReportResponse {
  report_id: string;
  generated_at: string;
  summary: {
    total_comments: number;
    date_range: string;
    avg_engagement: number;
    dominant_sentiment: string;
    top_topic: string;
    engagement_trend: string;
    data_quality_score: number;
  };
  detailed_analysis: {
    sentiment_analysis: any;
    engagement_analysis: any;
    topic_analysis: any;
    temporal_analysis: any;
    content_analysis: any;
  };
  insights: string[];
  recommendations: string[];
  methodology: string[];
  data_sources: string[];
}

export interface ModelMetrics {
  model_metrics: {
    sentiment_accuracy: number;
    engagement_r2: number;
    num_topics: number;
    dataset_size: number;
  };
  dataset_info: {
    rows: number;
    last_updated: string;
  };
  api_status: string;
}

export interface DataSummary {
  basic_info: {
    total_rows: number;
    total_columns: number;
    columns: string[];
    memory_usage_mb: number;
    date_range?: string;
    avg_engagement?: number;
  };
  data_quality: any;
  temporal_analysis: any;
  content_analysis: any;
  engagement_analysis: any;
}

// API Service Class
class ApiService {
  // Health and Status
  async healthCheck(): Promise<{ status: string; timestamp: string; models_loaded: boolean }> {
    return api.get('/health');
  }

  async getModelMetrics(): Promise<ModelMetrics> {
    return api.get('/metrics');
  }

  async getDataSummary(): Promise<DataSummary> {
    return api.get('/data/summary');
  }

  // Comment Analysis
  async analyzeComment(request: CommentAnalysisRequest): Promise<CommentAnalysisResponse> {
    return api.post('/analyze/comment', request);
  }

  async analyzeBatch(request: BatchAnalysisRequest): Promise<{
    results: CommentAnalysisResponse[];
    processed_count: number;
    processed_at: string;
  }> {
    return api.post('/analyze/batch', request);
  }

  // Topic Modeling
  async getTopicAnalysis(): Promise<TopicModelingResponse> {
    return api.get('/analysis/topics');
  }

  // Reports
  async generateReport(): Promise<ReportResponse> {
    return api.post('/reports/generate');
  }

  // Data Upload
  async uploadDataset(file: File): Promise<{
    message: string;
    rows_processed: number;
    analysis_summary: any;
  }> {
    const formData = new FormData();
    formData.append('file', file);

    return api.post('/data/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
  }

  // Utility methods
  async testConnection(): Promise<boolean> {
    try {
      await this.healthCheck();
      return true;
    } catch (error) {
      console.error('Connection test failed:', error);
      return false;
    }
  }

  // Get API status
  async getApiInfo(): Promise<{
    message: string;
    version: string;
    status: string;
    endpoints: Record<string, string>;
  }> {
    return api.get('/');
  }
}

// Create and export singleton instance
export const apiService = new ApiService();

// Export utility functions
export const formatApiError = (error: any): string => {
  if (error?.detail) {
    return Array.isArray(error.detail) ? error.detail[0]?.msg || 'Unknown error' : error.detail;
  }
  if (error?.message) {
    return error.message;
  }
  return 'An unexpected error occurred';
};

export const isApiError = (error: any): boolean => {
  return error && (error.detail || error.message || error.status);
};

// Export default
export default apiService;
