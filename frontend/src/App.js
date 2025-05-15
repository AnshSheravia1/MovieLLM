import React, { useState } from 'react';
import {
  Container,
  Box,
  Typography,
  TextField,
  Button,
  Paper,
  CircularProgress,
  ThemeProvider,
  createTheme
} from '@mui/material';
import axios from 'axios';

const muiTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#90caf9',
    },
    secondary: {
      main: '#f48fb1',
    },
  },
});

function App() {
  const [theme, setTheme] = useState('');
  const [generatedScript, setGeneratedScript] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleGenerate = async () => {
    if (!theme.trim()) {
      setError('Please enter a theme');
      return;
    }

    setLoading(true);
    setError('');
    setGeneratedScript(null);
    try {
      const response = await axios.post('http://localhost:8000/generate', { 
        theme: theme,
      });
      setGeneratedScript(response.data);
    } catch (err) {
      setError('Failed to generate script. Please try again.');
      console.error('Error:', err);
      if (err.response) {
        console.error('Error response data:', err.response.data);
        setError(`Failed to generate script: ${err.response.data.detail || 'Server error'}`);
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <ThemeProvider theme={muiTheme}>
      <Container maxWidth="md" sx={{ py: 4 }}>
        <Typography variant="h3" component="h1" gutterBottom align="center">
          MovieLLM
        </Typography>
        <Typography variant="h6" gutterBottom align="center" color="text.secondary">
          AI Movie Script Generator
        </Typography>

        <Paper elevation={3} sx={{ p: 3, mb: 3 }}>
          <Box sx={{ mb: 3 }}>
            <TextField
              fullWidth
              label="Enter your movie theme"
              variant="outlined"
              value={theme}
              onChange={(e) => setTheme(e.target.value)}
              placeholder="e.g., A romantic drama about two people from different backgrounds"
              error={!!error}
              helperText={error}
            />
          </Box>
          <Button
            variant="contained"
            color="primary"
            onClick={handleGenerate}
            disabled={loading}
            fullWidth
          >
            {loading ? <CircularProgress size={24} /> : 'Generate Script'}
          </Button>
        </Paper>

        {generatedScript && (
          <Paper elevation={3} sx={{ p: 3, mt: 3 }}>
            <Typography variant="h5" component="h2" gutterBottom>
              {generatedScript.title}
            </Typography>

            {generatedScript.synopsis && (
              <Box sx={{ mb: 2 }}>
                <Typography variant="h6" gutterBottom>Synopsis:</Typography>
                <Typography variant="body1" sx={{ whiteSpace: 'pre-wrap' }}>
                  {generatedScript.synopsis}
                </Typography>
              </Box>
            )}

            {generatedScript.characters && generatedScript.characters.length > 0 && (
              <Box sx={{ mb: 2 }}>
                <Typography variant="h6" gutterBottom>Characters:</Typography>
                <ul>
                  {generatedScript.characters.map((char, index) => (
                    <li key={index}>
                      <Typography component="span" sx={{ fontWeight: 'bold' }}>{char.name}:</Typography> {char.description || 'No description.'}
                    </li>
                  ))}
                </ul>
              </Box>
            )}

            {generatedScript.acts && generatedScript.acts.length > 0 && (
              <Box sx={{ mb: 2 }}>
                <Typography variant="h6" gutterBottom>Acts:</Typography>
                {generatedScript.acts.map((act, index) => (
                  <Box key={index} sx={{ mb: 2, pl: 2 }}>
                    <Typography variant="subtitle1" sx={{ fontWeight: 'bold' }}>{act.act_title}</Typography>
                    <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap' }}>
                      {act.summary}
                    </Typography>
                  </Box>
                ))}
              </Box>
            )}

            {generatedScript.scenes && generatedScript.scenes.length > 0 && (
              <Box sx={{ mb: 2 }}>
                <Typography variant="h6" gutterBottom>Sample Scenes:</Typography>
                {generatedScript.scenes.map((scene, sceneIndex) => (
                  <Box key={sceneIndex} sx={{ mb: 3, pl: 2 }}>
                    {scene.scene_heading && (
                       <Typography variant="subtitle1" sx={{ fontWeight: 'bold' }}>
                         {scene.scene_heading}
                       </Typography>
                    )}
                    <Typography variant="subtitle1" sx={{ fontWeight: 'bold' }}>
                      Scene Description:
                    </Typography>
                    <Typography component="pre" sx={{ whiteSpace: 'pre-wrap', fontFamily: 'monospace', bgcolor: 'grey.800', p: 2, borderRadius: 1, mb: 2 }}>
                      {scene.description || "No description provided."}
                    </Typography>

                    {scene.dialogue && scene.dialogue.length > 0 && (
                      <>
                        <Typography variant="subtitle1" sx={{ fontWeight: 'bold' }}>
                          Dialogue:
                        </Typography>
                        {scene.dialogue.map((line, lineIndex) => (
                          <Box key={lineIndex} sx={{ mb: 1, ml: 2 }}>
                            <Typography variant="body1" sx={{ fontWeight: 'bold' }}>
                              {line.character}:
                            </Typography>
                            <Typography variant="body2" sx={{ pl: 2, whiteSpace: 'pre-wrap' }}>
                              {line.text}
                            </Typography>
                          </Box>
                        ))}
                      </>
                    )}
                  </Box>
                ))}
              </Box>
            )}
          </Paper>
        )}
      </Container>
    </ThemeProvider>
  );
}

export default App; 