import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Provider } from 'react-redux';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { CssBaseline, Box } from '@mui/material';
import { store } from './store/store';
import Navbar from './components/Layout/Navbar';
import Dashboard from './components/Dashboard/Dashboard';
import SentimentAnalyzer from './components/SentimentAnalyzer/SentimentAnalyzer';
import TopicModeling from './components/TopicModeling/TopicModeling';
import EngagementPredictor from './components/EngagementPredictor/EngagementPredictor';
import Reports from './components/Reports/Reports';
import DataUpload from './components/DataUpload/DataUpload';

// Create custom theme
const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#2E7D32', // Green for environmental theme
      light: '#4CAF50',
      dark: '#1B5E20',
    },
    secondary: {
      main: '#FF9800', // Orange for accent
      light: '#FFB74D',
      dark: '#F57C00',
    },
    background: {
      default: '#F1F8E9',
      paper: '#FFFFFF',
    },
    text: {
      primary: '#1B5E20',
      secondary: '#388E3C',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontSize: '2.5rem',
      fontWeight: 600,
      color: '#1B5E20',
    },
    h2: {
      fontSize: '2rem',
      fontWeight: 500,
      color: '#2E7D32',
    },
    h3: {
      fontSize: '1.5rem',
      fontWeight: 500,
    },
  },
  components: {
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: '12px',
          boxShadow: '0 4px 6px rgba(0, 0, 0, 0.07)',
          border: '1px solid rgba(76, 175, 80, 0.1)',
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: '8px',
          textTransform: 'none',
          fontWeight: 500,
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          borderRadius: '16px',
        },
      },
    },
  },
});

const App: React.FC = () => {
  return (
    <Provider store={store}>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <Router>
          <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
            <Navbar />
            <Box component="main" sx={{ flexGrow: 1, p: 3, pt: { xs: 2, sm: 3 } }}>
              <Routes>
                <Route path="/" element={<Dashboard />} />
                <Route path="/dashboard" element={<Dashboard />} />
                <Route path="/sentiment" element={<SentimentAnalyzer />} />
                <Route path="/topics" element={<TopicModeling />} />
                <Route path="/engagement" element={<EngagementPredictor />} />
                <Route path="/reports" element={<Reports />} />
                <Route path="/upload" element={<DataUpload />} />
              </Routes>
            </Box>
          </Box>
        </Router>
      </ThemeProvider>
    </Provider>
  );
};

export default App;
