import React, { useState, useEffect } from 'react';
import {
  Container,
  Grid,
  Card,
  CardContent,
  Typography,
  Box,
  Chip,
  Alert,
  CircularProgress,
  Button,
  List,
  ListItem,
  ListItemText,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  LinearProgress,
} from '@mui/material';
import {
  Topic as TopicIcon,
  ExpandMore as ExpandMoreIcon,
  Refresh as RefreshIcon,
  Psychology as PsychologyIcon,
} from '@mui/icons-material';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, BarChart, Bar, XAxis, YAxis, CartesianGrid, Legend } from 'recharts';
import { apiService, TopicModelingResponse } from '../../services/apiService';

const COLORS = ['#2E7D32', '#4CAF50', '#66BB6A', '#81C784', '#A5D6A7', '#C8E6C9', '#FF9800', '#FFB74D', '#FFCC02', '#FFF176'];

const TopicModeling: React.FC = () => {
  const [topicData, setTopicData] = useState<TopicModelingResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadTopicData();
  }, []);

  const loadTopicData = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await apiService.getTopicAnalysis();
      setTopicData(data);
    } catch (err: any) {
      setError(err.message || 'Failed to load topic analysis');
    } finally {
      setLoading(false);
    }
  };

  const formatDistributionData = () => {
    if (!topicData?.topic_distribution) return [];
    
    return Object.entries(topicData.topic_distribution).map(([name, value], index) => ({
      name,
      value: Math.round(value * 100),
      color: COLORS[index % COLORS.length],
    }));
  };

  const formatTopicsForChart = () => {
    if (!topicData?.topics) return [];
    
    return topicData.topics.map(topic => ({
      name: topic.label,
      relevance: Math.round((topicData.topic_distribution[topic.label] || 0) * 100),
      keywords: topic.words.slice(0, 3).join(', '),
    }));
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

  const distributionData = formatDistributionData();
  const chartData = formatTopicsForChart();

  return (
    <Container maxWidth="lg">
      <Box sx={{ mb: 4 }}>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
          <Typography variant="h3" component="h1" sx={{ fontWeight: 600 }}>
            Topic Modeling
          </Typography>
          <Button
            variant="outlined"
            startIcon={<RefreshIcon />}
            onClick={loadTopicData}
            disabled={loading}
          >
            Refresh Analysis
          </Button>
        </Box>

        <Typography variant="h6" color="text.secondary" sx={{ mb: 4 }}>
          Discover key themes and topics in climate change discussions
        </Typography>

        {error && (
          <Alert severity="error" sx={{ mb: 3 }}>
            {error}
          </Alert>
        )}

        {topicData && (
          <Grid container spacing={3}>
            {/* Topic Overview */}
            <Grid item xs={12} md={8}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Topic Distribution
                  </Typography>
                  
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" angle={-45} textAnchor="end" height={80} />
                      <YAxis label={{ value: 'Relevance %', angle: -90, position: 'insideLeft' }} />
                      <Tooltip 
                        formatter={(value, name, props) => [
                          `${value}%`,
                          'Relevance',
                          `Keywords: ${props.payload.keywords}`
                        ]}
                      />
                      <Bar dataKey="relevance" fill="#2E7D32" />
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </Grid>

            {/* Topic Distribution Pie Chart */}
            <Grid item xs={12} md={4}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Topic Proportions
                  </Typography>
                  
                  <ResponsiveContainer width="100%" height={300}>
                    <PieChart>
                      <Pie
                        data={distributionData}
                        cx="50%"
                        cy="50%"
                        outerRadius={80}
                        dataKey="value"
                        label={({name, value}) => `${value}%`}
                      >
                        {distributionData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                      </Pie>
                      <Tooltip formatter={(value) => [`${value}%`, 'Share']} />
                    </PieChart>
                  </ResponsiveContainer>

                  <Box mt={2}>
                    {distributionData.map((item, index) => (
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
                        <Typography variant="body2" noWrap>
                          {item.name}
                        </Typography>
                      </Box>
                    ))}
                  </Box>
                </CardContent>
              </Card>
            </Grid>

            {/* Detailed Topic Analysis */}
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Detailed Topic Analysis
                  </Typography>
                  
                  {topicData.topics.map((topic, index) => (
                    <Accordion key={topic.id} sx={{ mb: 1 }}>
                      <AccordionSummary
                        expandIcon={<ExpandMoreIcon />}
                        aria-controls={`topic-${topic.id}-content`}
                        id={`topic-${topic.id}-header`}
                      >
                        <Box sx={{ width: '100%', mr: 2 }}>
                          <Box display="flex" justifyContent="space-between" alignItems="center" width="100%">
                            <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                              {topic.label}
                            </Typography>
                            <Box display="flex" alignItems="center" gap={1}>
                              <LinearProgress
                                variant="determinate"
                                value={(topicData.topic_distribution[topic.label] || 0) * 100}
                                sx={{ width: 100, mr: 1 }}
                              />
                              <Typography variant="body2" color="text.secondary">
                                {Math.round((topicData.topic_distribution[topic.label] || 0) * 100)}%
                              </Typography>
                            </Box>
                          </Box>
                          <Box mt={1}>
                            {topic.words.slice(0, 5).map((word, wordIndex) => (
                              <Chip
                                key={wordIndex}
                                label={word}
                                size="small"
                                variant="outlined"
                                sx={{ mr: 0.5, mb: 0.5 }}
                              />
                            ))}
                          </Box>
                        </Box>
                      </AccordionSummary>
                      
                      <AccordionDetails>
                        <Grid container spacing={2}>
                          <Grid item xs={12} md={6}>
                            <Typography variant="subtitle2" gutterBottom>
                              Key Words & Weights
                            </Typography>
                            <List dense>
                              {topic.words.slice(0, 10).map((word, wordIndex) => (
                                <ListItem key={wordIndex} sx={{ py: 0.5 }}>
                                  <ListItemText
                                    primary={word}
                                    secondary={`Weight: ${(topic.weights[wordIndex] || 0).toFixed(3)}`}
                                  />
                                </ListItem>
                              ))}
                            </List>
                          </Grid>
                          
                          <Grid item xs={12} md={6}>
                            <Typography variant="subtitle2" gutterBottom>
                              Representative Comments
                            </Typography>
                            {topicData.representative_comments[topic.label]?.slice(0, 3).map((comment, commentIndex) => (
                              <Box key={commentIndex} sx={{ mb: 2, p: 2, bgcolor: 'grey.50', borderRadius: 1 }}>
                                <Typography variant="body2">
                                  "{comment}"
                                </Typography>
                              </Box>
                            )) || (
                              <Typography variant="body2" color="text.secondary">
                                No representative comments available
                              </Typography>
                            )}
                          </Grid>
                        </Grid>
                      </AccordionDetails>
                    </Accordion>
                  ))}
                </CardContent>
              </Card>
            </Grid>

            {/* Topic Insights */}
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Box display="flex" alignItems="center" mb={2}>
                    <PsychologyIcon sx={{ mr: 1, color: 'primary.main' }} />
                    <Typography variant="h6">
                      Topic Insights
                    </Typography>
                  </Box>
                  
                  <Grid container spacing={2}>
                    <Grid item xs={12} md={4}>
                      <Box sx={{ textAlign: 'center', p: 2 }}>
                        <Typography variant="h4" color="primary.main">
                          {topicData.topics.length}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Topics Discovered
                        </Typography>
                      </Box>
                    </Grid>
                    
                    <Grid item xs={12} md={4}>
                      <Box sx={{ textAlign: 'center', p: 2 }}>
                        <Typography variant="h4" color="secondary.main">
                          {distributionData.length > 0 ? distributionData[0].name : 'N/A'}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Most Discussed Topic
                        </Typography>
                      </Box>
                    </Grid>
                    
                    <Grid item xs={12} md={4}>
                      <Box sx={{ textAlign: 'center', p: 2 }}>
                        <Typography variant="h4" color="success.main">
                          {distributionData.length > 0 ? `${distributionData[0].value}%` : '0%'}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Topic Coverage
                        </Typography>
                      </Box>
                    </Grid>
                  </Grid>

                  <Box mt={3}>
                    <Typography variant="body1" paragraph>
                      <strong>Analysis Summary:</strong> The topic modeling reveals {topicData.topics.length} distinct 
                      themes in the climate change discussions. The most prominent topic appears to be related to{' '}
                      {distributionData.length > 0 ? distributionData[0].name.toLowerCase() : 'general climate discussion'}, 
                      accounting for {distributionData.length > 0 ? distributionData[0].value : 0}% of the conversations.
                    </Typography>
                    
                    <Typography variant="body2" color="text.secondary">
                      Topics are discovered using Latent Dirichlet Allocation (LDA) with climate-specific preprocessing 
                      and optimization. The model automatically determines topic coherence and assigns human-readable labels 
                      based on keyword analysis and domain expertise.
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

export default TopicModeling;
