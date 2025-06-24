const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');

const subscribeRoute = require('./routes/subscribe');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(bodyParser.json());

// Routes
app.use('/api', subscribeRoute);

app.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
});