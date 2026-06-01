export default async function handler(req, res) {
  // Only allow POST requests
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method Not Allowed' });
  }

  const upstreamUrl = process.env.NEWSRAG_API;
  if (!upstreamUrl) {
    return res.status(500).json({ error: 'NEWSRAG_API environment variable is not configured' });
  }

  try {
    const response = await fetch(upstreamUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(req.body),
    });

    const data = await response.json();
    return res.status(response.status).json(data);
  } catch (error) {
    console.error('Error proxying request to NewsRAG API:', error);
    return res.status(500).json({ error: 'Failed to query upstream API' });
  }
}
