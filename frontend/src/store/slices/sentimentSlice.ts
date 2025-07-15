import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import { apiService, CommentAnalysisRequest, CommentAnalysisResponse, BatchAnalysisRequest } from '../../services/apiService';

interface SentimentState {
  currentAnalysis: CommentAnalysisResponse | null;
  batchResults: CommentAnalysisResponse[];
  loading: boolean;
  error: string | null;
  analysisHistory: CommentAnalysisResponse[];
}

const initialState: SentimentState = {
  currentAnalysis: null,
  batchResults: [],
  loading: false,
  error: null,
  analysisHistory: [],
};

export const analyzeComment = createAsyncThunk(
  'sentiment/analyzeComment',
  async (request: CommentAnalysisRequest, { rejectWithValue }) => {
    try {
      const result = await apiService.analyzeComment(request);
      return result;
    } catch (error: any) {
      return rejectWithValue(error.message || 'Failed to analyze comment');
    }
  }
);

export const analyzeBatch = createAsyncThunk(
  'sentiment/analyzeBatch',
  async (request: BatchAnalysisRequest, { rejectWithValue }) => {
    try {
      const result = await apiService.analyzeBatch(request);
      return result.results;
    } catch (error: any) {
      return rejectWithValue(error.message || 'Failed to analyze batch');
    }
  }
);

const sentimentSlice = createSlice({
  name: 'sentiment',
  initialState,
  reducers: {
    clearError: (state) => {
      state.error = null;
    },
    clearCurrentAnalysis: (state) => {
      state.currentAnalysis = null;
    },
    clearBatchResults: (state) => {
      state.batchResults = [];
    },
    addToHistory: (state, action: PayloadAction<CommentAnalysisResponse>) => {
      state.analysisHistory.unshift(action.payload);
      if (state.analysisHistory.length > 50) {
        state.analysisHistory = state.analysisHistory.slice(0, 50);
      }
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(analyzeComment.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(analyzeComment.fulfilled, (state, action) => {
        state.loading = false;
        state.currentAnalysis = action.payload;
        state.analysisHistory.unshift(action.payload);
        if (state.analysisHistory.length > 50) {
          state.analysisHistory = state.analysisHistory.slice(0, 50);
        }
      })
      .addCase(analyzeComment.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload as string;
      })
      .addCase(analyzeBatch.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(analyzeBatch.fulfilled, (state, action) => {
        state.loading = false;
        state.batchResults = action.payload;
      })
      .addCase(analyzeBatch.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload as string;
      });
  },
});

export const { clearError, clearCurrentAnalysis, clearBatchResults, addToHistory } = sentimentSlice.actions;
export default sentimentSlice.reducer;
