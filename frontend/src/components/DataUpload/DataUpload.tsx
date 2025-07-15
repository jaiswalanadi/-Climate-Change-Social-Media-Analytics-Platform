import React, { useState, useCallback } from 'react';
import {
  Container,
  Grid,
  Card,
  CardContent,
  Typography,
  Box,
  Button,
  Alert,
  CircularProgress,
  LinearProgress,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider,
  Chip,
  Paper,
} from '@mui/material';
import {
  CloudUpload as CloudUploadIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Info as InfoIcon,
  Description as DescriptionIcon,
  Assessment as AssessmentIcon,
  Schedule as ScheduleIcon,
} from '@mui/icons-material';
import { apiService } from '../../services/apiService';

interface UploadResult {
  message: string;
  rows_processed: number;
  analysis_summary: any;
}

const DataUpload: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState<UploadResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setError(null);
      setUploadResult(null);
    }
  };

  const handleDrop = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragOver(false);
    
    const files = event.dataTransfer.files;
    if (files.length > 0) {
      const file = files[0];
      if (isValidFileType(file)) {
        setSelectedFile(file);
        setError(null);
        setUploadResult(null);
      } else {
        setError('Please upload a CSV, JSON, or Excel file');
      }
    }
  }, []);

  const handleDragOver = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragOver(true);
  }, []);

  const handleDragLeave = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragOver(false);
  }, []);

  const isValidFileType = (file: File) => {
    const validTypes = ['.csv', '.json', '.xlsx', '.xls'];
    return validTypes.some(type => file.name.toLowerCase().endsWith(type));
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    try {
      setUploading(true);
      setError(null);
      
      const result = await apiService.uploadDataset(selectedFile);
      setUploadResult(result);
    } catch (err: any) {
      setError(err.message || 'Upload failed');
    } finally {
      setUploading(false);
    }
  };

  const getFileIcon = (fileName: string) => {
    const extension = fileName.toLowerCase().split('.').pop();
    switch (extension) {
      case 'csv':
        return '📊';
      case 'json':
        return '📄';
      case 'xlsx':
      case 'xls':
        return '📈';
      default:
        return '📁';
    }
  };

  const renderUploadArea = () => (
    <Paper
      sx={{
        border: dragOver ? '2px dashed #4CAF50' : '2px dashed #ccc',
        borderRadius: 2,
        p: 4,
        textAlign: 'center',
        bgcolor: dragOver ? 'action.hover' : 'background.paper',
        cursor: 'pointer',
        transition: 'all 0.3s ease',
        '&:hover': {
          borderColor: 'primary.main',
          bgcolor: 'action.hover',
        },
      }}
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onClick={() => document.getElementById('file-input')?.click()}
    >
      <input
        id="file-input"
        type="file"
        accept=".csv,.json,.xlsx,.xls"
        onChange={handleFileSelect}
        style={{ display: 'none' }}
      />
      
      <CloudUploadIcon sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
      <Typography variant="h6" gutterBottom>
        {dragOver ? 'Drop your file here' : 'Upload Climate Data'}
      </Typography>
      <Typography variant="body2" color="text.secondary" paragraph>
        Drag and drop your file here, or click to browse
      </Typography>
      <Typography variant="caption" color="text.secondary">
        Supported formats: CSV, JSON, Excel (.xlsx, .xls)
      </Typography>
    </Paper>
  );

  return (
    <Container maxWidth="lg">
      <Box sx={{ mb: 4 }}>
        <Typography variant="h3" component="h1" sx={{ fontWeight: 600, mb: 2 }}>
          Data Upload
        </Typography>
        <Typography variant="h6" color="text.secondary" sx={{ mb: 4 }}>
          Upload your climate change social media dataset for analysis
        </Typography>

        <Grid container spacing={3}>
          {/* Upload Section */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Upload Dataset
                </Typography>

                {error && (
                  <Alert severity="error" sx={{ mb: 2 }}>
                    {error}
                  </Alert>
                )}

                {!selectedFile ? (
                  renderUploadArea()
                ) : (
                  <Box sx={{ bgcolor: 'grey.100', p: 2, borderRadius: 1, fontFamily: 'monospace', fontSize: '0.875rem' }}>
                  date,likesCount,profileName,commentsCount,text<br/>
                  2022-09-07T17:12:32.000Z,2,user_hash,0,"Climate comment..."
                </Box>
              </CardContent>
            </Card>
          </Grid>

          {/* Upload Results */}
          {uploadResult && (
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Box display="flex" alignItems="center" mb={2}>
                    <CheckCircleIcon color="success" sx={{ mr: 1 }} />
                    <Typography variant="h6">
                      Upload Successful!
                    </Typography>
                  </Box>

                  <Alert severity="success" sx={{ mb: 3 }}>
                    {uploadResult.message}
                  </Alert>

                  <Grid container spacing={3}>
                    <Grid item xs={12} md={4}>
                      <Paper sx={{ p: 2, textAlign: 'center' }}>
                        <Typography variant="h4" color="primary.main">
                          {uploadResult.rows_processed.toLocaleString()}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Rows Processed
                        </Typography>
                      </Paper>
                    </Grid>

                    {uploadResult.analysis_summary && (
                      <>
                        <Grid item xs={12} md={4}>
                          <Paper sx={{ p: 2, textAlign: 'center' }}>
                            <Typography variant="h4" color="secondary.main">
                              {uploadResult.analysis_summary.engagement_statistics?.likes?.total_likes || 0}
                            </Typography>
                            <Typography variant="body2" color="text.secondary">
                              Total Likes
                            </Typography>
                          </Paper>
                        </Grid>

                        <Grid item xs={12} md={4}>
                          <Paper sx={{ p: 2, textAlign: 'center' }}>
                            <Typography variant="h4" color="success.main">
                              {Math.round(uploadResult.analysis_summary.text_statistics?.avg_length || 0)}
                            </Typography>
                            <Typography variant="body2" color="text.secondary">
                              Avg Comment Length
                            </Typography>
                          </Paper>
                        </Grid>
                      </>
                    )}
                  </Grid>

                  {uploadResult.analysis_summary && (
                    <Box mt={3}>
                      <Typography variant="h6" gutterBottom>
                        Dataset Summary
                      </Typography>
                      
                      <Grid container spacing={2}>
                        {uploadResult.analysis_summary.temporal_analysis && (
                          <Grid item xs={12} md={6}>
                            <Typography variant="subtitle2" gutterBottom>
                              Temporal Coverage
                            </Typography>
                            <Box sx={{ bgcolor: 'grey.50', p: 2, borderRadius: 1 }}>
                              <Typography variant="body2">
                                <Schedule fontSize="small" sx={{ mr: 1, verticalAlign: 'middle' }} />
                                {uploadResult.analysis_summary.temporal_analysis.most_active_month || 'N/A'} (Most Active)
                              </Typography>
                              <Typography variant="body2" color="text.secondary">
                                Peak posting time: {uploadResult.analysis_summary.temporal_analysis.most_active_hour || 'N/A'}:00
                              </Typography>
                            </Box>
                          </Grid>
                        )}

                        {uploadResult.analysis_summary.content_analysis && (
                          <Grid item xs={12} md={6}>
                            <Typography variant="subtitle2" gutterBottom>
                              Content Quality
                            </Typography>
                            <Box sx={{ bgcolor: 'grey.50', p: 2, borderRadius: 1 }}>
                              <Typography variant="body2">
                                Climate-related: {uploadResult.analysis_summary.content_analysis.climate_related_percentage || 0}%
                              </Typography>
                              <Typography variant="body2" color="text.secondary">
                                Avg words per comment: {Math.round(uploadResult.analysis_summary.content_analysis.avg_words_per_comment || 0)}
                              </Typography>
                            </Box>
                          </Grid>
                        )}
                      </Grid>
                    </Box>
                  )}

                  <Box mt={3}>
                    <Typography variant="body1" paragraph>
                      Your dataset has been successfully processed and is ready for analysis. 
                      You can now use the sentiment analyzer, topic modeling, and engagement prediction features.
                    </Typography>
                    
                    <Box display="flex" gap={2}>
                      <Button variant="contained" href="/sentiment">
                        Start Sentiment Analysis
                      </Button>
                      <Button variant="outlined" href="/topics">
                        Explore Topics
                      </Button>
                      <Button variant="outlined" href="/reports">
                        Generate Report
                      </Button>
                    </Box>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          )}

          {/* Sample Data Information */}
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Need Sample Data?
                </Typography>
                
                <Typography variant="body1" paragraph>
                  If you don't have your own dataset, the platform comes pre-loaded with a sample dataset 
                  of over 500 climate change comments from NASA's Facebook page (2020-2023).
                </Typography>

                <Box display="flex" flexWrap="wrap" gap={1} mb={2}>
                  <Chip label="522 Comments" variant="outlined" />
                  <Chip label="2020-2023 Period" variant="outlined" />
                  <Chip label="NASA Climate Data" variant="outlined" />
                  <Chip label="Privacy Protected" variant="outlined" />
                </Box>

                <Typography variant="body2" color="text.secondary">
                  The sample dataset includes anonymized user profiles, engagement metrics, and 
                  diverse climate change discussions suitable for testing all platform features.
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </Box>
    </Container>
  );
};

export default DataUpload;>
                    <Paper sx={{ p: 2, mb: 2, bgcolor: 'grey.50' }}>
                      <Box display="flex" alignItems="center">
                        <Typography variant="h6" sx={{ mr: 1 }}>
                          {getFileIcon(selectedFile.name)}
                        </Typography>
                        <Box flexGrow={1}>
                          <Typography variant="subtitle1">
                            {selectedFile.name}
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            {formatFileSize(selectedFile.size)}
                          </Typography>
                        </Box>
                        <Button
                          variant="text"
                          color="error"
                          onClick={() => setSelectedFile(null)}
                        >
                          Remove
                        </Button>
                      </Box>
                    </Paper>

                    <Button
                      variant="contained"
                      startIcon={uploading ? <CircularProgress size={20} /> : <CloudUploadIcon />}
                      onClick={handleUpload}
                      disabled={uploading}
                      fullWidth
                      size="large"
                    >
                      {uploading ? 'Uploading...' : 'Upload & Process'}
                    </Button>

                    {uploading && (
                      <Box sx={{ mt: 2 }}>
                        <LinearProgress />
                        <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                          Processing your data... This may take a few moments.
                        </Typography>
                      </Box>
                    )}
                  </Box>
                )}
              </CardContent>
            </Card>
          </Grid>

          {/* Requirements & Guidelines */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Data Requirements
                </Typography>

                <List>
                  <ListItem>
                    <ListItemIcon>
                      <CheckCircleIcon color="success" />
                    </ListItemIcon>
                    <ListItemText
                      primary="Required Columns"
                      secondary="text (comment content), date (timestamp)"
                    />
                  </ListItem>
                  
                  <ListItem>
                    <ListItemIcon>
                      <InfoIcon color="info" />
                    </ListItemIcon>
                    <ListItemText
                      primary="Optional Columns"
                      secondary="likesCount, commentsCount, profileName"
                    />
                  </ListItem>
                  
                  <ListItem>
                    <ListItemIcon>
                      <AssessmentIcon color="primary" />
                    </ListItemIcon>
                    <ListItemText
                      primary="Minimum Data"
                      secondary="At least 10 comments for meaningful analysis"
                    />
                  </ListItem>
                  
                  <ListItem>
                    <ListItemIcon>
                      <DescriptionIcon color="secondary" />
                    </ListItemIcon>
                    <ListItemText
                      primary="File Size"
                      secondary="Maximum 10MB per file"
                    />
                  </ListItem>
                </List>

                <Divider sx={{ my: 2 }} />

                <Typography variant="subtitle2" gutterBottom>
                  Expected CSV Format:
                </Typography>
                <Box
