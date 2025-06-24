const MailerLite = require('@mailerlite/mailerlite-nodejs').default;
const express = require('express');
const router = express.Router();
require('dotenv').config();

const mailerlite = new MailerLite({
    api_key: process.env.MAILER_LITE_API_KEY,
});

router.post('/subscribe', async (req, res) => {
    const { email } = req.body;

    if (!email) {
        return res.status(400).json({ message: 'Email is required' });
    }

    const params = {
        email: email,
        groups: ["158001098652648546"],
        subscribed_at: new Date().toISOString().replace('T', ' ').replace(/\.\d{3}Z$/, ''),
    }
    mailerlite.subscribers.createOrUpdate(params)
        .then((response) => {
            console.log('Successfully subscribed!');
            res.status(200).json({ message: 'Successfully subscribed!' });
        })
        .catch((error) => {
            console.log(error);
            console.error('MailerLite error:');
            res.status(500).json({ message: 'Subscription failed.' });
        })
    
})

module.exports = router;
