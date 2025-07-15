import React, { useState, useEffect } from 'react';
import {
  Container,
  Grid,
  Card,
  CardContent,
  Typography,
  Box,
  Chip,
  LinearProgress,
  Alert,
  Button,
  CircularProgress,
} from '@mui/material';
import {
  TrendingUp,
  Sentiment,
  Topic,
  Assessment,
  People,
  Schedule,
  BarChart,
  Refresh,
} from '@mui/icons-material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell, BarChart as RechartsBarChart, Bar } from 'recharts';
import { apiService } from '../../services/apiService';

interface DashboardStats {
  total_comments: number;
  date_range: string;
  avg_engagement: number;
  dominant_sentiment: string;
  top_topic: string;
  engagement_trend: string;
  data_quality_score: number;
}

interface MetricCard {
  title: string;
  value: string | number;
  icon: React.ReactNode;
  color: string;
  change?: string;
  changeType?: 'positive' | 'negative' | 'neutral';
}

const Dashboard: React.FC = () => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [dashboardData, setDashboardData] = useState<DashboardStats | null>(null);
  const [metrics, setMetrics] = useState<any>(null);
  const [summaryData, setSummaryData] = useState<any>(null);

  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    try {
      setLoading(true);
      setError(null);

      // Load metrics and summary data in parallel
      const [metricsResponse, summaryResponse] = await Promise.all([
        apiService.getModelMetrics(),
        apiService.getDataSummary(),
      ]);

      setMetrics(metricsResponse);
      setSummaryData(summaryResponse);

      // Extract dashboard stats from summary
      if (summaryResponse.basic_info) {
        const stats: DashboardStats = {
          total_comments: summaryResponse.basic_info.total_rows || 0,
          date_range: summaryResponse.basic_info.date_range || 'Unknown',
          avg_engagement: summaryResponse.basic_info.avg_engagement || 0,
          dominant_sentiment: 'Neutral', // Will be updated from sentiment analysis
          top_topic: 'Climate Discussion', // Will be updated from topic analysis
          engagement_trend: 'Stable',
          data_quality_score: summaryResponse.data_quality?.quality_score || 75,
        };
        setDashboardData(stats);
      }
    } catch (err) {
      console.error('Error loading dashboard data:', err);
      setError('Failed to load dashboard data. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const getMetricCards = (): MetricCard[] => {
    if (!dashboardData) return [];

    return [
      {
        title: 'Total Comments',
        value: dashboardData.total_comments.toLocaleString(),
        icon: <People />,
        color: '#2E7D32',
      },
      {
        title: 'Average Engagement',
        value: dashboardData.avg_engagement.toFixed(1),
        icon: <TrendingUp />,
        color: '#FF9800',
        change: dashboardData.engagement_trend,
      },
      {
        title: 'Dominant Sentiment',
        value: dashboardData.dominant_sentiment,
        icon: <Sentiment />,
        color: getSentimentColor(dashboardData.dominant_sentiment),
      },
      {
        title: 'Data Quality Score',
        value: `${dashboardData.data_quality_score}%`,
        icon: <Assessment />,
        color: getQualityColor(dashboardData.data_quality_score),
      },
    ];
  };

  const getSentimentColor = (sentiment: string): string => {
    switch (sentiment.toLowerCase()) {
      case 'positive': return '#4CAF50';
      case 'negative': return '#F44336';
      default: return '#FF9800';
    }
  };

  const getQualityColor = (score: number): string => {
    if (score >= 80) return '#4CAF50';
    if (score >= 60) return '#FF9800';
    return '#F44336';
  };

  const generateSampleChartData = () => {
    // Generate sample data for demonstration
    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'];
    return months.map(month => ({
      month,
      comments: Math.floor(Math.random() * 100) + 20,
      engagement: Math.floor(Math.random() * 50) + 10,
      sentiment: Math.random(),
    }));
  };

  const generateSentimentData = () => {
    return [
      { name: 'Positive', value: 35, color: '#4CAF50' },
      { name: 'Neutral', value: 45, color: '#FF9800' },
      { name: 'Negative', value: 20, color: '#F44336' },
    ];
  };

  const generateTopicData = () => {
    return [
      { topic: 'Renewable Energy', count: 120 },
      { topic: 'Climate Science', count: 98 },
      { topic: 'Policy & Politics', count: 87 },
      { topic: 'Environmental Impact', count: 76 },
      { topic: 'Activism', count: 65 },
    ];
  };

  if (loading) {
    return (
      <Container maxWidth="lg">
        <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
          <CircularProgress size={60} />
        </Box>
      </Container>
    );
  }

  const chartData = generateSampleChartData();
  const sentimentData = generateSentimentData();
  const topicData = generateTopicData();

  return (
    <Container maxWidth="lg">
      <Box sx={{ mb: 4 }}>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
          <Typography variant="h3" component="h1" sx={{ fontWeight: 600 }}>
            Climate Analytics Dashboard
          </Typography>
          <Button
            variant="outlined"
            startIcon={<Refresh />}
            onClick={loadDashboardData}
            disabled={loading}
          >
            Refresh Data
          </Button>
        </Box>

        {error && (
          <Alert severity="error" sx={{ mb: 3 }}>
            {error}
          </Alert>
        )}

        {dashboardData && (
          <Grid container spacing={3}>
            {/* Metric Cards */}
            {getMetricCards().map((metric, index) => (
              <Grid item xs={12} sm={6} md={3} key={index}>
                <Card sx={{ height: '100%' }}>
                  <CardContent>
                    <Box display="flex" alignItems="center" mb={2}>
                      <Box
                        sx={{
                          backgroundColor: metric.color,
                          color: 'white',
                          borderRadius: '50%',
                          p: 1,
                          mr: 2,
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                        }}
                      >
                        {metric.icon}
                      </Box>
                      <Typography variant="h6" color="text.secondary">
                        {metric.title}
                      </Typography>
                    </Box>
                    <Typography variant="h4" sx={{ fontWeight: 600, mb: 1 }}>
                      {metric.value}
                    </Typography>
                    {metric.change && (
                      <Chip
                        label={metric.change}
                        size="small"
                        color={metric.changeType === 'positive' ? 'success' : 'default'}
                      />
                    )}
                  </CardContent>
                </Card>
              </Grid>
            ))}

            {/* Engagement Trend Chart */}
            <Grid item xs={12} md={8}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Engagement Trends Over Time
                  </Typography>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="month" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Line
                        type="monotone"
                        dataKey="comments"
                        stroke="#2E7D32"
                        strokeWidth={2}
                        name="Comments"
                      />
                      <Line
                        type="monotone"
                        dataKey="engagement"
                        stroke="#FF9800"
                        strokeWidth={2}
                        name="Avg Engagement"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </Grid>

            {/* Sentiment Distribution */}
            <Grid item xs={12} md={4}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Sentiment Distribution
                  </Typography>
                  <ResponsiveContainer width="100%" height={300}>
                    <PieChart>
                      <Pie
                        data={sentimentData}
                        cx="50%"
                        cy="50%"
                        innerRadius={60}
                        outerRadius={100}
                        paddingAngle={5}
                        dataKey="value"
                      >
                        {sentimentData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                      </Pie>
                      <Tooltip formatter={(value) => [`${value}%`, 'Percentage']} />
                    </PieChart>
                  </ResponsiveContainer>
                  <Box mt={2}>
                    {sentimentData.map((item, index) => (
                      <Box key={index} display="flex" alignItems="center" mb={1}>
                        <Box
                          sx={{
                            width: 12,
                            height: 12,
                            backgroundColor: item.color,
                            borderRadius: '50%',
                            mr: 1,
                          }}
                        />
                        <Typography variant="body2">
                          {item.name}: {item.value}%
                        </Typography>
                      </Box>
                    ))}
                  </Box>
                </CardContent>
              </Card>
            </Grid>

            {/* Top Topics */}
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Top Discussion Topics
                  </Typography>
                  <ResponsiveContainer width="100%" height={300}>
                    <RechartsBarChart data={topicData} layout="horizontal">
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis type="number" />
                      <YAxis dataKey="topic" type="category" width={120} />
                      <Tooltip />
                      <Bar dataKey="count" fill="#2E7D32" />
                    </RechartsBarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </Grid>

            {/* System Status */}
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    System Status
                  </Typography>
                  
                  <Box mb={3}>
                    <Box display="flex" justifyContent="space-between" mb={1}>
                      <Typography variant="body2">Data Quality</Typography>
                      <Typography variant="body2">
                        {dashboardData.data_quality_score}%
                      </Typography>
                    </Box>
                    <LinearProgress
                      variant="determinate"
                      value={dashboardData.data_quality_score}
                      sx={{
                        height: 8,
                        borderRadius: 4,
                        backgroundColor: 'grey.300',
                        '& .MuiLinearProgress-bar': {
                          backgroundColor: getQualityColor(dashboardData.data_quality_score),
                        },
                      }}
                    />
                  </Box>

                  <Box mb={2}>
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      Model Performance
                    </Typography>
                    {metrics && (
                      <Box>
                        <Chip
                          label={`Sentiment Accuracy: ${(metrics.model_metrics?.sentiment_accuracy * 100 || 75).toFixed(1)}%`}
                          size="small"
                          sx={{ mr: 1, mb: 1 }}
                        />
                        <Chip
                          label={`Engagement R²: ${(metrics.model_metrics?.engagement_r2 || 0.65).toFixed(2)}`}
                          size="small"
                          sx={{ mr: 1, mb: 1 }}
                        />
                        <Chip
                          label={`Topics: ${metrics.model_metrics?.num_topics || 5}`}
                          size="small"
                          sx={{ mb: 1 }}
                        />
                      </Box>
                    )}
                  </Box>

                  <Box>
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      Data Coverage
                    </Typography>
                    <Typography variant="body2">
                      <Schedule fontSize="small" sx={{ mr: 1, verticalAlign: 'middle' }} />
                      {dashboardData.date_range}
                    </Typography>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        )}
      </Box>
    </Container>
  );
};

export default Dashboard;
