#!/bin/bash

echo "🚀 Building RAG Backend Docker Image..."
docker build -t rag-backend .

echo "📦 Running RAG Backend Container..."
docker run -d \
  --name rag-backend \
  -p 8000:8000 \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -v $(pwd)/database:/app/database \
  --restart unless-stopped \
  rag-backend

echo "✅ RAG Backend deployed successfully!"
echo "🌐 Access the API at: http://localhost:8000"
echo "📁 Database is persisted in: ./database"
echo ""
echo "To view logs: docker logs rag-backend"
echo "To stop: docker stop rag-backend"
echo "To remove: docker rm rag-backend"
