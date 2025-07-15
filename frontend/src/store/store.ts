import { configureStore } from '@reduxjs/toolkit';
import { useDispatch, useSelector, TypedUseSelectorHook } from 'react-redux';

// Slice imports (we'll create these)
import dashboardSlice from './slices/dashboardSlice';
import sentimentSlice from './slices/sentimentSlice';
import topicSlice from './slices/topicSlice';
import engagementSlice from './slices/engagementSlice';
import reportSlice from './slices/reportSlice';
import uiSlice from './slices/uiSlice';

export const store = configureStore({
  reducer: {
    dashboard: dashboardSlice,
    sentiment: sentimentSlice,
    topic: topicSlice,
    engagement: engagementSlice,
    report: reportSlice,
    ui: uiSlice,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: {
        ignoredActions: ['persist/PERSIST', 'persist/REHYDRATE'],
      },
    }),
  devTools: process.env.NODE_ENV !== 'production',
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;

// Typed hooks
export const useAppDispatch = () => useDispatch<AppDispatch>();
export const useAppSelector: TypedUseSelectorHook<RootState> = useSelector;
