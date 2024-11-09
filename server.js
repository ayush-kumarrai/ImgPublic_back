const express = require('express');
const cors = require('cors');
const { GoogleGenerativeAI } = require('@google/generative-ai');
const sharp = require('sharp');
require('dotenv').config();

const app = express();
const port = process.env.PORT || 5001;

// Middleware
app.use(cors());
app.use(express.json({ limit: '50mb' }));

// Configure Gemini AI
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

// Helper function to convert base64 to image for Gemini
async function prepareImageForGemini(base64String) {
  try {
    // Remove data URL prefix if present
    const base64Data = base64String.replace(/^data:image\/\w+;base64,/, '');

    // Validate base64 string
    if (!base64Data.match(/^([A-Za-z0-9+/]{4})*([A-Za-z0-9+/]{4}|[A-Za-z0-9+/]{3}=|[A-Za-z0-9+/]{2}==)$/)) {
      throw new Error('Invalid base64 string format');
    }

    // Convert base64 to buffer
    const imageBuffer = Buffer.from(base64Data, 'base64');

    // Optimize image
    const processedImage = await sharp(imageBuffer)
      .resize(800, 800, {
        fit: 'inside',
        withoutEnlargement: true
      })
      .toBuffer();

    return processedImage.toString('base64');
  } catch (error) {
    console.error('Image preparation error:', error);
    throw error;
  }
}

// Route for image analysis
app.post('/analyze', async (req, res) => {
  try {
    const { image } = req.body;

    if (!image) {
      return res.status(400).json({ error: 'No image data provided' });
    }

    // Prepare image for Gemini
    const processedImage = await prepareImageForGemini(image);

    // Use the most appropriate Gemini model for vision
    const model = genAI.getGenerativeModel({
      model: "gemini-1.5-flash-latest" // Updated model
    });

    const prompt = `
      You are an advanced medical imaging AI assistant.
      Carefully analyze this medical image and provide a comprehensive diagnosis.
      Do not include these kind of message " I am an AI chatbot and cannot provide medical advice" or related to it.
      Include:
      - Potential medical conditions or abnormalities
      - Key observations
      - Recommended next steps
      - Provide General Medicine For it.
      - A very small one line Disclaimer that this is an AI assessment and professional medical consultation is essential

    `;

    const result = await model.generateContent({
      contents: [
        {
          role: 'user',
          parts: [
            { text: prompt },
            {
              inlineData: {
                mimeType: 'image/jpeg',
                data: processedImage
              }
            }
          ]
        }
      ]
    });

    // Extract and send diagnosis
    const diagnosis = result.response.text();
    res.json({ diagnosis });

  } catch (error) {
    console.error('Comprehensive image analysis error:', {
      message: error.message,
      stack: error.stack,
      requestBody: req.body
    });

    res.status(500).json({
      error: 'Error analyzing image',
      details: error.toString(),
      fullError: error.stack
    });
  }
});

// Route for chat functionality
app.post('/chat', async (req, res) => {
  try {
    const { messages, diagnosis } = req.body;

    // Use Gemini Pro for chat
    const model = genAI.getGenerativeModel({ 
      model: "gemini-pro" 
    });

    // Prepare chat history and context
    const chatHistory = messages.map(msg => ({
      role: msg.role === 'user' ? 'user' : 'model',
      parts: [{ text: msg.content }]
    }));

    // Add initial context about the diagnosis
    const initialContext = `
      Context: A medical image has been diagnosed with the following initial assessment: 
      ${diagnosis}

      Please provide helpful, detailed, and compassionate responses to the user's questions 
      about this diagnosis. Maintain a supportive and informative tone.
    `;

    // Start chat
    const chat = model.startChat({
      history: [
        { role: 'user', parts: [{ text: initialContext }] }
      ],
      generationConfig: {
        maxOutputTokens: 1000,
      }
    });

    // Get the last user message
    const lastUserMessage = messages.filter(msg => msg.role === 'user').pop();
    
    // Send message and get response
    const result = await chat.sendMessage(lastUserMessage.content);
    const response = result.response.text();

    res.json({ response });

  } catch (error) {
    console.error('Chat error:', {
      message: error.message,
      stack: error.stack,
      requestBody: req.body
    });

    res.status(500).json({
      error: 'Error processing chat message',
      details: error.toString(),
      fullError: error.stack
    });
  }
});

// Start server
app.listen(port, () => {
  console.log(`Server running on port ${port}`);
});