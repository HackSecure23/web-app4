import express from 'express';
import cors from 'cors';
import bodyParser from 'body-parser';
import axios from 'axios';
import natural from 'natural';
import lda from 'lda';

const SentimentAnalyzer = natural.SentimentAnalyzer;
const PorterStemmer = natural.PorterStemmer;
import TfIdf from 'natural/lib/natural/tfidf/tfidf.js';

const app = express();
app.use(cors());
app.use(bodyParser.json());

const API_KEY = 'AIzaSyCh3w2TXdyYnPV_b0u8vXFWrEesZspv_4Q';

// ✅ **Fetch Channel Data**
async function getChannelData(channelName) {
    try {
        console.log(`📡 Searching for channel: ${channelName}`);
        const searchResponse = await axios.get(`https://www.googleapis.com/youtube/v3/search`, {
            params: { key: API_KEY, q: channelName, type: "channel", part: "snippet", maxResults: 1 }
        });

        if (!searchResponse.data.items || searchResponse.data.items.length === 0) {
            console.error("❌ No channel found.");
            return null;
        }

        const channelId = searchResponse.data.items[0].id.channelId;
        const channelTitle = searchResponse.data.items[0].snippet.title;

        const statsResponse = await axios.get(`https://www.googleapis.com/youtube/v3/channels`, {
            params: { key: API_KEY, id: channelId, part: "statistics" }
        });

        const stats = statsResponse.data.items[0]?.statistics || {};

        return {
            channelId,
            title: channelTitle,
            subscriberCount: stats.subscriberCount || "0"
        };

    } catch (error) {
        console.error("❌ Error fetching channel data:", error.response?.data || error);
        return null;
    }
}

// ✅ **Fetch Last 10 Videos & Tags**
async function getLatestVideos(channelId) {
    try {
        console.log(`📡 Fetching latest 10 videos for channel ID: ${channelId}`);

        const videoResponse = await axios.get(`https://www.googleapis.com/youtube/v3/search`, {
            params: { key: API_KEY, channelId: channelId, part: "snippet", maxResults: 10, order: "date", type: "video" }
        });

        const videoIds = videoResponse.data.items.map(video => video.id.videoId);

        const videoDetailsResponse = await axios.get(`https://www.googleapis.com/youtube/v3/videos`, {
            params: { key: API_KEY, id: videoIds.join(','), part: "snippet,statistics" }
        });

        return videoDetailsResponse.data.items.map(video => ({
            videoId: video.id,
            title: video.snippet.title || "No Title",
            description: video.snippet.description || "No Description",
            tags: video.snippet.tags || [],
            views: video.statistics?.viewCount || "N/A",
            likes: video.statistics?.likeCount || "N/A",
            comments: video.statistics?.commentCount || "N/A"
        }));

    } catch (error) {
        console.error("❌ Error fetching latest videos:", error.response?.data || error);
        return [];
    }
}

// ✅ **Perform Advanced Keyword Analysis**
function analyzeKeywords(videos) {
    const tokenizer = new natural.WordTokenizer();
    const tfidf = new TfIdf();
    const sentimentAnalyzer = new SentimentAnalyzer("English", PorterStemmer, "afinn");

    let allTokens = [];
    let sentimentScores = [];
    let videoTexts = [];

    videos.forEach(video => {
        const combinedText = `${video.title} ${video.description} ${video.tags.join(' ')}`.toLowerCase();
        const tokens = tokenizer.tokenize(combinedText);
        allTokens.push(...tokens);
        tfidf.addDocument(tokens.join(' '));
        sentimentScores.push(sentimentAnalyzer.getSentiment(tokens));
        videoTexts.push(combinedText); // Store for topic modeling
    });

    const topKeywords = [...new Set(allTokens)]
        .filter(word => word.length > 3) // Filter out short words
        .slice(0, 15); // Get top 15 keywords

    const bigrams = natural.NGrams.bigrams(allTokens)
        .map(b => b.join(' '))
        .filter(phrase => phrase.split(' ').length === 2)
        .slice(0, 10);

    const trigrams = natural.NGrams.trigrams(allTokens)
        .map(t => t.join(' '))
        .filter(phrase => phrase.split(' ').length === 3)
        .slice(0, 10);

    let tfidfScores = {};
    tfidf.listTerms(0).forEach(item => {
        tfidfScores[item.term] = item.tfidf;
    });

    let topicModeling;
    try {
        const ldaResults = lda(videoTexts, 3, 5); // Extract 3 topics with 5 words each
        topicModeling = ldaResults.map(topic => topic.map(word => word.term).join(", ")); // Fix `[object Object]` issue
    } catch (error) {
        console.error("❌ Error processing LDA:", error);
        topicModeling = [];
    }

    // ✅ **Generate Super Relevant Suggested Keywords**
    const suggestedKeywords = [...new Set([...topKeywords, ...bigrams, ...trigrams])]
        .sort((a, b) => (tfidfScores[b] || 0) - (tfidfScores[a] || 0))
        .slice(0, 10);

    return {
        keywords: topKeywords,
        bigrams,
        trigrams,
        tfidfScores,
        avgSentiment: sentimentScores.length
            ? sentimentScores.reduce((a, b) => a + b, 0) / sentimentScores.length
            : 0,
        topics: topicModeling,
        suggestedKeywords
    };
}

// ✅ **Analyze Competitor & Return Data**
app.post('/analyze', async (req, res) => {
    const { username } = req.body;
    if (!username) return res.status(400).json({ error: "Username is required." });

    try {
        const channelData = await getChannelData(username);
        if (!channelData) return res.status(404).json({ error: "Channel not found." });

        const videos = await getLatestVideos(channelData.channelId);
        const keywordAnalysis = analyzeKeywords(videos);

        res.json({ channelData, videos, keywordAnalysis });

    } catch (error) {
        console.error("❌ Error analyzing competitor:", error);
        res.status(500).json({ error: "Failed to analyze competitor." });
    }
});

// ✅ **Fetch Last Analysis (No MongoDB)**
app.get('/videos', async (req, res) => {
    res.json({ message: "Real-time analysis only. Use /analyze for competitor data." }); //comment
});

const PORT = 5000;
app.listen(PORT, '0.0.0.0', () => console.log(`🚀 Server running on http://localhost:${PORT}`));