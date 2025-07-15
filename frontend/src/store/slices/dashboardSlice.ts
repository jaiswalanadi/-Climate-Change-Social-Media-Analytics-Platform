import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import { apiService, ModelMetrics, DataSummary } from '../../services/apiService';

// Types
interface DashboardState {
  metrics: ModelMetrics | null;
  dataSummary: DataSummary | null;
  loading: boolean;
  error: string | null;
  lastUpdated: string | null;
  connectionStatus: 'connected' | 'disconnected' | 'checking';
}

// Initial state
const initialState: DashboardState = {
  metrics: null,
  dataSummary: null,
  loading: false,
  error: null,
  lastUpdated: null,
  connectionStatus: 'checking',
};

// Async thunks
export const fetchDashboardData = createAsyncThunk(
  'dashboard/fetchData',
  async (_, { rejectWithValue }) => {
    try {
      const [metrics, dataSummary] = await Promise.all([
        apiService.getModelMetrics(),
        apiService.getDataSummary(),
      ]);
      return { metrics, dataSummary };
    } catch (error: any) {
      return rejectWithValue(error.message || 'Failed to fetch dashboard data');
    }
  }
);

export const checkConnection = createAsyncThunk(
  'dashboard/checkConnection',
  async (_, { rejectWithValue }) => {
    try {
      const isConnected = await apiService.testConnection();
      return isConnected;
    } catch (error: any) {
      return rejectWithValue(false);
    }
  }
);

// Slice
const dashboardSlice = createSlice({
  name: 'dashboard',
  initialState,
  reducers: {
    clearError: (state) => {
      state.error = null;
    },
    setConnectionStatus: (state, action: PayloadAction<'connected' | 'disconnected' | 'checking'>) => {
      state.connectionStatus = action.payload;
    },
  },
  extraReducers: (builder) => {
    builder
      // Fetch dashboard data
      .addCase(fetchDashboardData.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchDashboardData.fulfilled, (state, action) => {
        state.loading = false;
        state.metrics = action.payload.metrics;
        state.dataSummary = action.payload.dataSummary;
        state.lastUpdated = new Date().toISOString();
        state.connectionStatus = 'connected';
      })
      .addCase(fetchDashboardData.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload as string;
        state.connectionStatus = 'disconnected';
      })
      // Check connection
      .addCase(checkConnection.pending, (state) => {
        state.connectionStatus = 'checking';
      })
      .addCase(checkConnection.fulfilled, (state, action) => {
        state.connectionStatus = action.payload ? 'connected' : 'disconnected';
      })
      .addCase(checkConnection.rejected, (state) => {
        state.connectionStatus = 'disconnected';
      });
  },
});

export const { clearError, setConnectionStatus } = dashboardSlice.actions;
export default dashboardSlice.reducer;
