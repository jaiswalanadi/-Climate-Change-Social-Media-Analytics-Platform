# Climate Change Social Media Analytics Platform

A comprehensive AI-powered platform for analyzing climate change discussions on social media. This project provides sentiment analysis, engagement prediction, topic modeling, and comprehensive reporting capabilities for climate-related social media content.

## 🌍 Project Overview

This platform analyzes over 500 user comments from NASA's Climate Change Facebook page (2020-2023) to provide insights into public opinion, engagement patterns, and discussion themes around climate change topics.

### 🔥 Key Features

- **Sentiment Analysis Engine** - Multi-class sentiment classification with climate-specific optimization
- **Engagement Prediction** - Predict likes and comments based on content analysis
- **Topic Modeling** - Discover key themes using LDA with climate domain expertise
- **Interactive Dashboard** - Real-time analytics and visualization
- **Comprehensive Reports** - Automated insights and recommendations
- **Data Upload & Processing** - Support for CSV, JSON, and Excel files

## 🏗️ Architecture

### Backend (Python FastAPI)
- **FastAPI** - High-performance REST API
- **Machine Learning** - scikit-learn, transformers, NLTK
- **Data Processing** - pandas, numpy
- **Database** - SQLAlchemy with SQLite/PostgreSQL support

### Frontend (React TypeScript)
- **React 18** - Modern React with TypeScript
- **Material-UI** - Professional UI components
- **Redux Toolkit** - State management
- **Recharts** - Data visualization

## 📊 Core Functionality

### 1. Sentiment Analysis
- **Climate-optimized NLP** with domain-specific keywords
- **Multi-signal approach** combining TextBlob, engagement metrics, and ML models
- **Confidence scoring** for prediction reliability
- **Temporal sentiment tracking**

### 2. Engagement Prediction
- **Feature extraction** from text characteristics
- **Gradient Boosting models** for likes and comments prediction
- **Real-time recommendations** for content optimization
- **Performance analytics** with R² scoring

### 3. Topic Modeling
- **Latent Dirichlet Allocation (LDA)** with climate domain optimization
- **Automatic topic number selection** using coherence scoring
- **Human-readable topic labels** with representative comments
- **Trend analysis** over time

### 4. Data Processing
- **Comprehensive text preprocessing** with multiple levels
- **Data quality scoring** and validation
- **Privacy protection** with SHA-256 hashing
- **Temporal and engagement feature extraction**

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 16+
- Git

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd climate-analytics-platform
```

2. **Backend Setup**
```bash
# Navigate to backend
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create data directories
mkdir -p data/raw data/processed data/models reports logs

# Copy your dataset
copy path\to\climate_nasa.csv data\raw\

# Start backend server
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

3. **Frontend Setup**
```bash
# Open new terminal and navigate to frontend
cd frontend

# Install dependencies
npm install

# Start development server
npm start
```

4. **Access the application**
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## 📁 Project Structure

```
climate-analytics-platform/
├── frontend/                 # React TypeScript frontend
│   ├── src/
│   │   ├── components/      # React components
│   │   │   ├── Dashboard/
│   │   │   ├── SentimentAnalyzer/
│   │   │   ├── TopicModeling/
│   │   │   ├── EngagementPredictor/
│   │   │   └── Reports/
│   │   ├── services/        # API services
│   │   ├── store/          # Redux store and slices
│   │   └── utils/          # Utility functions
│   └── package.json
├── backend/                 # Python FastAPI backend
│   ├── app/
│   │   ├── models/         # ML models
│   │   ├── services/       # Business logic
│   │   │   ├── sentiment_analyzer.py
│   │   │   ├── engagement_predictor.py
│   │   │   ├── topic_modeler.py
│   │   │   ├── data_processor.py
│   │   │   └── report_generator.py
│   │   ├── utils/          # Utilities
│   │   ├── routes/         # API routes
│   │   └── config/         # Configuration
│   ├── data/              # Data storage
│   │   ├── raw/           # Raw datasets
│   │   ├── processed/     # Processed data
│   │   └── models/        # Trained models
│   └── requirements.txt
├── docs/                   # Documentation
├── tests/                  # Test files
└── deployment/             # Deployment configs
```

## 🔧 Configuration

### Environment Variables
Create `.env` files in both frontend and backend directories:

**Backend (.env)**
```env
DATABASE_URL=sqlite:///./climate_analytics.db
SECRET_KEY=your-secret-key-here
DEBUG=True
LOG_LEVEL=INFO
```

**Frontend (.env)**
```env
REACT_APP_API_URL=http://localhost:8000
```

## 📈 Model Performance

### Sentiment Analysis
- **Accuracy**: ~85% on climate-specific content
- **Features**: TF-IDF vectorization with climate keywords
- **Model**: Random Forest with balanced classes

### Engagement Prediction
- **R² Score**: ~0.65 for likes prediction
- **Features**: Text characteristics, climate terms, social signals
- **Model**: Gradient Boosting Regressor

### Topic Modeling
- **Algorithm**: Latent Dirichlet Allocation (LDA)
- **Topics**: 3-10 topics (auto-selected)
- **Optimization**: Climate-specific stop words and preprocessing

## 🔍 API Endpoints

### Core Analysis
- `POST /analyze/comment` - Single comment analysis
- `POST /analyze/batch` - Batch comment analysis
- `GET /analysis/topics` - Topic modeling results

### Data Management
- `POST /data/upload` - Upload new dataset
- `GET /data/summary` - Dataset statistics
- `GET /metrics` - Model performance metrics

### Reports
- `POST /reports/generate` - Generate comprehensive report
- `GET /health` - System health check

## 🎯 Usage Examples

### Analyze a Comment
```python
import requests

response = requests.post("http://localhost:8000/analyze/comment", json={
    "text": "Renewable energy is the future! Solar and wind power will save our planet."
})

result = response.json()
print(f"Sentiment: {result['sentiment']}")
print(f"Predicted Likes: {result['predicted_likes']}")
print(f"Topics: {result['topics']}")
```

### Upload Dataset
```python
files = {'file': open('climate_data.csv', 'rb')}
response = requests.post("http://localhost:8000/data/upload", files=files)
print(response.json())
```

## 🚀 Deployment

### Local Development
```bash
# Backend
cd backend && python -m uvicorn app.main:app --reload

# Frontend
cd frontend && npm start
```

### Production Deployment

**Docker Deployment** (recommended)
```bash
# Build and run with Docker Compose
docker-compose up --build
```

**Railway/Vercel Deployment**
- Backend: Deploy to Railway using provided railway.json
- Frontend: Deploy to Vercel with automatic builds

**Traditional VPS**
- Backend: Use gunicorn with systemd service
- Frontend: Build and serve with nginx

## 🧪 Testing

```bash
# Backend tests
cd backend
python -m pytest tests/

# Frontend tests
cd frontend
npm test
```

## 📊 Data Format

### Input CSV Format
```csv
date,likesCount,profileName,commentsCount,text
2022-09-07T17:12:32.000Z,2,hashed_profile_name,0,"Climate change comment text..."
```

### Analysis Output
```json
{
  "sentiment": "positive",
  "sentiment_score": 0.85,
  "predicted_likes": 3,
  "predicted_comments": 1,
  "topics": ["Renewable Energy", "Climate Action"]
}
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📋 TODO & Future Enhancements

- [ ] Real-time social media API integration
- [ ] Multi-language support
- [ ] Advanced NLP models (BERT, GPT)
- [ ] Interactive data visualization
- [ ] Automated report scheduling
- [ ] User authentication and roles
- [ ] API rate limiting and caching
- [ ] Mobile responsive design improvements

## 🐛 Troubleshooting

### Common Issues

**Backend won't start**
- Check Python version (3.10+)
- Ensure all dependencies installed
- Verify data directories exist

**Frontend connection issues**
- Check backend is running on port 8000
- Verify CORS configuration
- Check network/firewall settings

**Model training failures**
- Ensure sufficient data (10+ rows)
- Check data format and columns
- Review preprocessing logs

### Logs and Debugging
- Backend logs: `logs/climate_analytics.log`
- Frontend console: Browser Developer Tools
- API documentation: http://localhost:8000/docs

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- NASA Climate Change Facebook page for the dataset
- Climate science community for domain expertise
- Open source ML/NLP libraries and frameworks
- Material-UI and React community

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Check the documentation in `/docs`
- Review API documentation at `/docs` endpoint

---

**Built with ❤️ for climate change awareness and data-driven insights**
