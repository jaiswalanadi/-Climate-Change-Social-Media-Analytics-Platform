from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import logging

from app.services.sentiment_analyzer import SentimentAnalyzer
from app.services.engagement_predictor import EngagementPredictor
from app.services.topic_modeler import TopicModeler
from app.services.data_processor import DataProcessor
from app.services.report_generator import ReportGenerator
from app.utils.preprocessing import TextPreprocessor
from app.config.settings import Settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Climate Change Analytics Platform",
    description="AI-powered social media analytics for climate change discussions",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
settings = Settings()
data_processor = DataProcessor()
sentiment_analyzer = SentimentAnalyzer()
engagement_predictor = EngagementPredictor()
topic_modeler = TopicModeler()
report_generator = ReportGenerator()
text_preprocessor = TextPreprocessor()

# Pydantic models
class CommentAnalysisRequest(BaseModel):
    text: str

class CommentAnalysisResponse(BaseModel):
    text: str
    sentiment: str
    sentiment_score: float
    predicted_likes: int
    predicted_comments: int
    topics: List[str]
    processed_at: datetime

class BatchAnalysisRequest(BaseModel):
    comments: List[str]

class DatasetUploadResponse(BaseModel):
    message: str
    rows_processed: int
    analysis_summary: Dict[str, Any]

class TopicModelingResponse(BaseModel):
    topics: List[Dict[str, Any]]
    topic_distribution: Dict[str, float]
    representative_comments: Dict[str, List[str]]

class ReportResponse(BaseModel):
    report_id: str
    generated_at: datetime
    summary: Dict[str, Any]
    insights: List[str]
    recommendations: List[str]

# Global variables to store processed data
current_dataset = None
model_metrics = {}

@app.on_event("startup")
async def startup_event():
    """Initialize models and load data on startup"""
    logger.info("Starting Climate Change Analytics Platform...")
    
    # Load and process the default dataset
    try:
        default_data_path = "data/raw/climate_nasa.csv"
        if os.path.exists(default_data_path):
            global current_dataset
            current_dataset = data_processor.load_dataset(default_data_path)
            logger.info(f"Loaded default dataset with {len(current_dataset)} rows")
            
            # Initialize models with the dataset
            await initialize_models()
        else:
            logger.warning("Default dataset not found. Models will be initialized on first data upload.")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")

async def initialize_models():
    """Initialize ML models with current dataset"""
    global model_metrics
    
    if current_dataset is not None and len(current_dataset) > 0:
        try:
            # Train sentiment analyzer
            sentiment_analyzer.train(current_dataset)
            
            # Train engagement predictor
            engagement_predictor.train(current_dataset)
            
            # Fit topic modeler
            topic_modeler.fit(current_dataset['text'].dropna().tolist())
            
            # Calculate model metrics
            model_metrics = {
                "sentiment_accuracy": sentiment_analyzer.get_accuracy(),
                "engagement_r2": engagement_predictor.get_r2_score(),
                "num_topics": topic_modeler.get_num_topics(),
                "dataset_size": len(current_dataset)
            }
            
            logger.info("Models initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Climate Change Analytics Platform API",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "analyze_comment": "/analyze/comment",
            "analyze_batch": "/analyze/batch",
            "upload_dataset": "/data/upload",
            "topic_modeling": "/analysis/topics",
            "generate_report": "/reports/generate",
            "model_metrics": "/metrics"
        }
    }

@app.post("/analyze/comment", response_model=CommentAnalysisResponse)
async def analyze_comment(request: CommentAnalysisRequest):
    """Analyze a single comment for sentiment, engagement, and topics"""
    try:
        # Preprocess text
        processed_text = text_preprocessor.preprocess(request.text)
        
        # Sentiment analysis
        sentiment, sentiment_score = sentiment_analyzer.predict(request.text)
        
        # Engagement prediction
        predicted_likes, predicted_comments = engagement_predictor.predict(request.text)
        
        # Topic extraction
        topics = topic_modeler.get_topics_for_text(request.text)
        
        return CommentAnalysisResponse(
            text=request.text,
            sentiment=sentiment,
            sentiment_score=sentiment_score,
            predicted_likes=int(predicted_likes),
            predicted_comments=int(predicted_comments),
            topics=topics,
            processed_at=datetime.now()
        )
    except Exception as e:
        logger.error(f"Error analyzing comment: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/analyze/batch")
async def analyze_batch(request: BatchAnalysisRequest):
    """Analyze multiple comments in batch"""
    try:
        results = []
        for comment in request.comments:
            if comment.strip():  # Skip empty comments
                sentiment, sentiment_score = sentiment_analyzer.predict(comment)
                predicted_likes, predicted_comments = engagement_predictor.predict(comment)
                topics = topic_modeler.get_topics_for_text(comment)
                
                results.append({
                    "text": comment,
                    "sentiment": sentiment,
                    "sentiment_score": sentiment_score,
                    "predicted_likes": int(predicted_likes),
                    "predicted_comments": int(predicted_comments),
                    "topics": topics
                })
        
        return {
            "results": results,
            "processed_count": len(results),
            "processed_at": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error in batch analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")

@app.post("/data/upload", response_model=DatasetUploadResponse)
async def upload_dataset(file: UploadFile = File(...)):
    """Upload and process a new dataset"""
    try:
        global current_dataset
        
        # Save uploaded file
        upload_path = f"data/raw/uploaded_{file.filename}"
        with open(upload_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Load and process dataset
        current_dataset = data_processor.load_dataset(upload_path)
        
        # Reinitialize models with new data
        await initialize_models()
        
        # Generate analysis summary
        analysis_summary = data_processor.get_dataset_summary(current_dataset)
        
        return DatasetUploadResponse(
            message="Dataset uploaded and processed successfully",
            rows_processed=len(current_dataset),
            analysis_summary=analysis_summary
        )
    except Exception as e:
        logger.error(f"Error uploading dataset: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/analysis/topics", response_model=TopicModelingResponse)
async def get_topic_analysis():
    """Get topic modeling results for the current dataset"""
    try:
        if current_dataset is None:
            raise HTTPException(status_code=400, detail="No dataset loaded")
        
        topics = topic_modeler.get_topics()
        topic_distribution = topic_modeler.get_topic_distribution()
        representative_comments = topic_modeler.get_representative_comments(current_dataset)
        
        return TopicModelingResponse(
            topics=topics,
            topic_distribution=topic_distribution,
            representative_comments=representative_comments
        )
    except Exception as e:
        logger.error(f"Error in topic analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Topic analysis failed: {str(e)}")

@app.post("/reports/generate", response_model=ReportResponse)
async def generate_report():
    """Generate comprehensive analysis report"""
    try:
        if current_dataset is None:
            raise HTTPException(status_code=400, detail="No dataset loaded")
        
        report = report_generator.generate_comprehensive_report(
            current_dataset,
            sentiment_analyzer,
            engagement_predictor,
            topic_modeler
        )
        
        return ReportResponse(
            report_id=report["report_id"],
            generated_at=datetime.now(),
            summary=report["summary"],
            insights=report["insights"],
            recommendations=report["recommendations"]
        )
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

@app.get("/metrics")
async def get_model_metrics():
    """Get current model performance metrics"""
    return {
        "model_metrics": model_metrics,
        "dataset_info": {
            "rows": len(current_dataset) if current_dataset is not None else 0,
            "last_updated": datetime.now().isoformat()
        },
        "api_status": "healthy"
    }

@app.get("/data/summary")
async def get_data_summary():
    """Get summary statistics of current dataset"""
    try:
        if current_dataset is None:
            raise HTTPException(status_code=400, detail="No dataset loaded")
        
        return data_processor.get_dataset_summary(current_dataset)
    except Exception as e:
        logger.error(f"Error getting data summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Data summary failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": len(model_metrics) > 0
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
