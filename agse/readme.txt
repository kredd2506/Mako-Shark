# Install dependencies
pip install requests beautifulsoup4 redis elasticsearch flask openai

# Set environment variable for API key
export OPENAI_API_KEY="API-key-here"

# Start Redis and Elasticsearch (same as before)
docker run -d -p 6379:6379 redis
docker run -d -p 9200:9200 -e "discovery.type=single-node" elasticsearch:7.17.0
