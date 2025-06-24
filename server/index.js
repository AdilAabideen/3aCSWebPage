const express = require('express');
const path = require('path'); // ← THIS LINE IS IMPORTANT
const cors = require('cors');
const bodyParser = require('body-parser');
const compression = require('compression');
const subscribeRoute = require('./routes/subscribe');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware - MUST come before routes
app.use(cors());
app.use(bodyParser.json());
app.use(compression());

// Serve the static frontend
app.use(express.static(path.join(__dirname, '../codebase/dist')));

// API Routes
app.use('/api', subscribeRoute);

// SPA Fallback (for React Router) - MUST be last
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, '../codebase/dist/index.html'));
});

app.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
});