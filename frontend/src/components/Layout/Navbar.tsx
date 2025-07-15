import React, { useState } from 'react';
import {
  AppBar,
  Toolbar,
  Typography,
  Button,
  IconButton,
  Drawer,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Box,
  useMediaQuery,
  useTheme,
  Divider,
} from '@mui/material';
import {
  Menu as MenuIcon,
  Dashboard as DashboardIcon,
  Sentiment as SentimentIcon,
  Topic as TopicIcon,
  TrendingUp as TrendingUpIcon,
  Assessment as ReportIcon,
  CloudUpload as UploadIcon,
  Eco as EcoIcon,
} from '@mui/icons-material';
import { useNavigate, useLocation } from 'react-router-dom';

const Navbar: React.FC = () => {
  const [mobileOpen, setMobileOpen] = useState(false);
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const navigate = useNavigate();
  const location = useLocation();

  const menuItems = [
    { text: 'Dashboard', icon: <DashboardIcon />, path: '/dashboard' },
    { text: 'Sentiment Analysis', icon: <SentimentIcon />, path: '/sentiment' },
    { text: 'Topic Modeling', icon: <TopicIcon />, path: '/topics' },
    { text: 'Engagement Predictor', icon: <TrendingUpIcon />, path: '/engagement' },
    { text: 'Reports', icon: <ReportIcon />, path: '/reports' },
    { text: 'Upload Data', icon: <UploadIcon />, path: '/upload' },
  ];

  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };

  const handleNavigation = (path: string) => {
    navigate(path);
    if (isMobile) {
      setMobileOpen(false);
    }
  };

  const isActive = (path: string) => {
    return location.pathname === path || (path === '/dashboard' && location.pathname === '/');
  };

  const drawer = (
    <Box sx={{ width: 280 }}>
      <Box sx={{ p: 2, display: 'flex', alignItems: 'center', gap: 1 }}>
        <EcoIcon sx={{ color: 'primary.main', fontSize: 32 }} />
        <Typography variant="h6" sx={{ color: 'primary.main', fontWeight: 600 }}>
          Climate Analytics
        </Typography>
      </Box>
      <Divider />
      <List sx={{ pt: 1 }}>
        {menuItems.map((item) => (
          <ListItem
            key={item.text}
            onClick={() => handleNavigation(item.path)}
            sx={{
              cursor: 'pointer',
              mx: 1,
              borderRadius: 2,
              mb: 0.5,
              bgcolor: isActive(item.path) ? 'primary.light' : 'transparent',
              color: isActive(item.path) ? 'white' : 'text.primary',
              '&:hover': {
                bgcolor: isActive(item.path) ? 'primary.main' : 'action.hover',
              },
            }}
          >
            <ListItemIcon
              sx={{
                color: isActive(item.path) ? 'white' : 'primary.main',
                minWidth: 40,
              }}
            >
              {item.icon}
            </ListItemIcon>
            <ListItemText 
              primary={item.text}
              primaryTypographyProps={{
                fontWeight: isActive(item.path) ? 600 : 400,
              }}
            />
          </ListItem>
        ))}
      </List>
    </Box>
  );

  return (
    <>
      <AppBar
        position="static"
        elevation={2}
        sx={{
          bgcolor: 'white',
          borderBottom: '1px solid',
          borderColor: 'divider',
        }}
      >
        <Toolbar>
          {isMobile && (
            <IconButton
              color="primary"
              aria-label="open drawer"
              edge="start"
              onClick={handleDrawerToggle}
              sx={{ mr: 2 }}
            >
              <MenuIcon />
            </IconButton>
          )}
          
          <Box sx={{ display: 'flex', alignItems: 'center', flexGrow: 1, gap: 1 }}>
            <EcoIcon sx={{ color: 'primary.main', fontSize: 32 }} />
            <Typography
              variant="h5"
              component="div"
              sx={{
                color: 'primary.main',
                fontWeight: 700,
                cursor: 'pointer',
              }}
              onClick={() => navigate('/dashboard')}
            >
              Climate Analytics Platform
            </Typography>
          </Box>

          {!isMobile && (
            <Box sx={{ display: 'flex', gap: 1 }}>
              {menuItems.map((item) => (
                <Button
                  key={item.text}
                  onClick={() => handleNavigation(item.path)}
                  startIcon={item.icon}
                  variant={isActive(item.path) ? 'contained' : 'text'}
                  color="primary"
                  sx={{
                    textTransform: 'none',
                    fontWeight: isActive(item.path) ? 600 : 400,
                    px: 2,
                  }}
                >
                  {item.text}
                </Button>
              ))}
            </Box>
          )}
        </Toolbar>
      </AppBar>

      {isMobile && (
        <Drawer
          variant="temporary"
          open={mobileOpen}
          onClose={handleDrawerToggle}
          ModalProps={{
            keepMounted: true, // Better open performance on mobile.
          }}
          sx={{
            '& .MuiDrawer-paper': {
              boxSizing: 'border-box',
              width: 280,
            },
          }}
        >
          {drawer}
        </Drawer>
      )}
    </>
  );
};

export default Navbar;
