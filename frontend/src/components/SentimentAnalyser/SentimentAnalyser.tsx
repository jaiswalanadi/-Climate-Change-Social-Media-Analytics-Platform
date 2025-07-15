import React, { useState } from 'react';
import {
  Container,
  Grid,
  Card,
  CardContent,
  Typography,
  TextField,
  Button,
  Box,
  Chip,
  Alert,
  CircularProgress,
  Divider,
  Paper,
  LinearProgress,
} from '@mui/material';
import {
  Sentiment as SentimentIcon,
  Send as SendIcon,
  TrendingUp,
  Comment,
  ThumbUp,
} from '@mui/icons-material';
import { useAppDispatch, useAppSelector } from '../../store/store';
import { analyzeComment, clearError } from '../../store/slices/sentimentSlice';

const SentimentAnalyzer: React.FC = () => {
  const [inputText, setInputText] = useState('');
  const dispatch = useAppDispatch();
  const { currentAnalysis, loading, error, analysisHistory } = useAppSelector(state => state.sentiment);

  const handleAnalyze = async () => {
    if (!inputText.trim()) return;
    
    dispatch(clearError());
    await dispatch(analyzeComment({ text: inputText }));
  };

  const getSentimentColor = (sentiment: string) => {
    switch (sentiment.toLowerCase()) {
      case 'positive': return '#4CAF50';
      case 'negative': return '#F44336';
      default: return '#FF9800';
    }
  };

  const getSentimentIcon = (sentiment: string) => {
    switch (sentiment.toLowerCase()) {
      case 'positive': return '😊';
      case 'negative': return '😟';
      default: return '😐';
    }
  };

  const formatConfidence = (score: number) => {
    return `${(score * 100).toFixed(1)}%`;
  };

  return (
    <Container maxWidth="lg">
      <Box sx={{ mb: 4 }}>
        <Typography variant="h3" component="h1" sx={{ fontWeight: 600, mb: 2 }}>
          Sentiment Analysis
        </Typography>
        <Typography variant="h6" color="text.secondary" sx={{ mb: 4 }}>
          Analyze the sentiment of climate change discussions using AI
        </Typography>

        <Grid container spacing={3}>
          {/* Input Section */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Analyze Text
                </Typography>
                
                {error && (
                  <Alert severity="error" sx={{ mb: 2 }}>
                    {error}
                  </Alert>
                )}

                <TextField
                  fullWidth
                  multiline
                  rows={6}
                  variant="outlined"
                  placeholder="Enter a climate change comment to analyze..."
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  sx={{ mb: 2 }}
                />

                <Box display="flex" justifyContent="space-between" alignItems="center">
                  <Typography variant="body2" color="text.secondary">
                    {inputText.length} characters
                  </Typography>
                  <Button
                    variant="contained"
                    startIcon={loading ? <CircularProgress size={20} /> : <SendIcon />}
                    onClick={handleAnalyze}
                    disabled={loading || !inputText.trim()}
                  >
                    {loading ? 'Analyzing...' : 'Analyze'}
                  </Button>
                </Box>
              </CardContent>
            </Card>
          </Grid>

          {/* Results Section */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Analysis Results
                </Typography>

                {currentAnalysis ? (
                  <Box>
                    {/* Sentiment Result */}
                    <Box sx={{ mb: 3 }}>
                      <Box display="flex" alignItems="center" mb={2}>
                        <Typography variant="h4" sx={{ mr: 1 }}>
                          {getSentimentIcon(currentAnalysis.sentiment)}
                        </Typography>
                        <Box>
                          <Typography variant="h6" sx={{ fontWeight: 600 }}>
                            {currentAnalysis.sentiment.charAt(0).toUpperCase() + currentAnalysis.sentiment.slice(1)}
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            Confidence: {formatConfidence(currentAnalysis.sentiment_score)}
                          </Typography>
                        </Box>
                      </Box>
                      
                      <LinearProgress
                        variant="determinate"
                        value={currentAnalysis.sentiment_score * 100}
                        sx={{
                          height: 8,
                          borderRadius: 4,
                          backgroundColor: 'grey.300',
                          '& .MuiLinearProgress-bar': {
                            backgroundColor: getSentimentColor(currentAnalysis.sentiment),
                          },
                        }}
                      />
                    </Box>

                    <Divider sx={{ my: 2 }} />

                    {/* Engagement Predictions */}
                    <Box sx={{ mb: 3 }}>
                      <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 2 }}>
                        Engagement Predictions
                      </Typography>
                      
                      <Grid container spacing={2}>
                        <Grid item xs={6}>
                          <Paper sx={{ p: 2, textAlign: 'center' }}>
                            <ThumbUp sx={{ color: 'primary.main', mb: 1 }} />
                            <Typography variant="h6">
                              {Math.round(currentAnalysis.predicted_likes)}
                            </Typography>
                            <Typography variant="body2" color="text.secondary">
                              Predicted Likes
                            </Typography>
                          </Paper>
                        </Grid>
                        <Grid item xs={6}>
                          <Paper sx={{ p: 2, textAlign: 'center' }}>
                            <Comment sx={{ color: 'secondary.main', mb: 1 }} />
                            <Typography variant="h6">
                              {Math.round(currentAnalysis.predicted_comments)}
                            </Typography>
                            <Typography variant="body2" color="text.secondary">
                              Predicted Comments
                            </Typography>
                          </Paper>
                        </Grid>
                      </Grid>
                    </Box>

                    <Divider sx={{ my: 2 }} />

                    {/* Topics */}
                    <Box>
                      <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 2 }}>
                        Detected Topics
                      </Typography>
                      <Box display="flex" flexWrap="wrap" gap={1}>
                        {currentAnalysis.topics.map((topic, index) => (
                          <Chip
                            key={index}
                            label={topic}
                            variant="outlined"
                            size="small"
                            color="primary"
                          />
                        ))}
                        {currentAnalysis.topics.length === 0 && (
                          <Typography variant="body2" color="text.secondary">
                            No specific topics detected
                          </Typography>
                        )}
                      </Box>
                    </Box>
                  </Box>
                ) : (
                  <Box sx={{ textAlign: 'center', py: 4 }}>
                    <SentimentIcon sx={{ fontSize: 60, color: 'text.secondary', mb: 2 }} />
                    <Typography variant="body1" color="text.secondary">
                      Enter text to see sentiment analysis results
                    </Typography>
                  </Box>
                )}
              </CardContent>
            </Card>
          </Grid>

          {/* Analysis History */}
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Recent Analysis History
                </Typography>

                {analysisHistory.length > 0 ? (
                  <Box>
                    {analysisHistory.slice(0, 5).map((analysis, index) => (
                      <Box key={index} sx={{ mb: 2, p: 2, bgcolor: 'grey.50', borderRadius: 2 }}>
                        <Grid container spacing={2} alignItems="center">
                          <Grid item xs={12} md={6}>
                            <Typography variant="body2" sx={{ 
                              overflow: 'hidden',
                              textOverflow: 'ellipsis',
                              display: '-webkit-box',
                              WebkitLineClamp: 2,
                              WebkitBoxOrient: 'vertical',
                            }}>
                              "{analysis.text}"
                            </Typography>
                          </Grid>
                          <Grid item xs={12} md={2}>
                            <Chip
                              label={analysis.sentiment}
                              size="small"
                              sx={{
                                backgroundColor: getSentimentColor(analysis.sentiment),
                                color: 'white',
                              }}
                            />
                          </Grid>
                          <Grid item xs={6} md={2}>
                            <Typography variant="body2" color="text.secondary">
                              👍 {Math.round(analysis.predicted_likes)}
                            </Typography>
                          </Grid>
                          <Grid item xs={6} md={2}>
                            <Typography variant="body2" color="text.secondary">
                              💬 {Math.round(analysis.predicted_comments)}
                            </Typography>
                          </Grid>
                        </Grid>
                      </Box>
                    ))}
                  </Box>
                ) : (
                  <Box sx={{ textAlign: 'center', py: 2 }}>
                    <Typography variant="body2" color="text.secondary">
                      No analysis history yet. Start by analyzing some text!
                    </Typography>
                  </Box>
                )}
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </Box>
    </Container>
  );
};

export default SentimentAnalyzer;
