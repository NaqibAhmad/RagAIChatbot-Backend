# RAG Healthcare API - FastAPI Middleware

A FastAPI-based middleware that provides a REST API interface for your RAG (Retrieval-Augmented Generation) healthcare appointment system. This middleware acts as a bridge between your frontend application and the backend RAG system.

## 🚀 Features

- **RESTful API**: Clean, standardized endpoints for frontend communication
- **CORS Support**: Configured for cross-origin requests from frontend applications
- **Health Monitoring**: Built-in health check endpoints
- **Flexible Search**: Support for hybrid, semantic, and keyword search modes
- **Error Handling**: Comprehensive error handling with proper HTTP status codes
- **API Documentation**: Auto-generated OpenAPI/Swagger documentation
- **Async Support**: Built with async/await for better performance

## 📁 Project Structure

```
RAG/
├── main.py                 # FastAPI application entry point
├── requirements.txt        # Python dependencies
├── src/
│   ├── customTypes.py     # Pydantic models for API requests/responses
│   ├── rag.py            # Main RAG processing logic
│   ├── ragUtils.py       # RAG system utilities
│   └── transcriptMessages.py # Transcript conversion utilities
└── database/              # Vector database storage
```

## 🛠️ Installation & Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Variables

Create a `.env` file in your project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Run the API

```bash
python main.py
```

The API will be available at:
- **API**: http://localhost:8000
- **Documentation**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 📚 API Endpoints

### Health & Status

#### `GET /`
Root endpoint with basic health information.

**Response:**
```json
{
  "status": "healthy",
  "message": "RAG Healthcare API is running",
  "timestamp": "2024-01-01T12:00:00"
}
```

#### `GET /health`
Detailed health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "message": "All systems operational",
  "timestamp": "2024-01-01T12:00:00"
}
```

### RAG Operations

#### `POST /api/rag/query`
Process RAG queries with conversation transcripts.

**Request Body:**
```json
{
  "transcript": [
    {
      "role": "user",
      "content": "I need to book an appointment for heel pain"
    },
    {
      "role": "agent",
      "content": "I can help you with that. What's your preferred date?"
    }
  ],
  "search_type": "hybrid",
  "temperature": 0.0,
  "model": "gpt-4o-mini"
}
```

**Response:**
```json
{
  "response": "I can help you book an appointment for heel pain. Based on our available slots...",
  "context_used": "Context from knowledge base...",
  "documents_retrieved": 3,
  "processing_time": 2.45,
  "search_type": "hybrid"
}
```

#### `POST /api/rag/search-type`
Update the search strategy for RAG queries.

**Query Parameters:**
- `search_type`: One of "hybrid", "semantic", or "keyword"

**Response:**
```json
{
  "message": "Search type updated to semantic",
  "search_type": "semantic",
  "timestamp": "2024-01-01T12:00:00"
}
```

#### `GET /api/rag/status`
Get current RAG system configuration and status.

**Response:**
```json
{
  "status": "operational",
  "persist_directory": "./src/db/DovestoneHealth",
  "embedding_model": "text-embedding-3-large",
  "llm_model": "gpt-4o",
  "timestamp": "2024-01-01T12:00:00"
}
```

## 🔧 Frontend Integration

### JavaScript/TypeScript Example

```typescript
// Example frontend integration
class RAGAPI {
  private baseURL = 'http://localhost:8000';

  async processQuery(transcript: any[]) {
    try {
      const response = await fetch(`${this.baseURL}/api/rag/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          transcript,
          search_type: 'hybrid',
          temperature: 0.0,
          model: 'gpt-4o-mini'
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error processing RAG query:', error);
      throw error;
    }
  }

  async getHealth() {
    const response = await fetch(`${this.baseURL}/health`);
    return await response.json();
  }

  async updateSearchType(searchType: string) {
    const response = await fetch(`${this.baseURL}/api/rag/search-type?search_type=${searchType}`, {
      method: 'POST'
    });
    return await response.json();
  }
}

// Usage
const ragAPI = new RAGAPI();

// Process a query
const result = await ragAPI.processQuery([
  { role: 'user', content: 'I need help with appointment booking' }
]);

console.log(result.response);
```

### React Hook Example

```typescript
import { useState, useCallback } from 'react';

interface RAGResponse {
  response: string;
  context_used?: string;
  documents_retrieved?: number;
  processing_time?: number;
  search_type?: string;
  error?: string;
}

export const useRAG = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const processQuery = useCallback(async (transcript: any[]): Promise<RAGResponse | null> => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('http://localhost:8000/api/rag/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ transcript })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      return result;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error occurred';
      setError(errorMessage);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  return { processQuery, loading, error };
};
```

## 🚨 Error Handling

The API returns appropriate HTTP status codes and error messages:

- **400 Bad Request**: Invalid input data
- **500 Internal Server Error**: Server-side processing errors
- **422 Unprocessable Entity**: Validation errors

All errors include:
- Error message
- Timestamp
- Appropriate HTTP status code

## 🔒 Security Considerations

- **CORS**: Currently configured to allow all origins (`*`). Configure properly for production
- **API Keys**: Ensure `OPENAI_API_KEY` is properly secured
- **Input Validation**: All inputs are validated using Pydantic models
- **Rate Limiting**: Consider adding rate limiting for production use

## 🚀 Production Deployment

### Using Gunicorn

```bash
pip install gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Using Docker

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 📝 Configuration

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `OPENAI_API_KEY` | OpenAI API key for LLM access | Yes | - |
| `PORT` | Server port | No | 8000 |
| `HOST` | Server host | No | 0.0.0.0 |

### CORS Configuration

Modify the CORS middleware in `main.py` for production:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Specific domains
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

## 🧪 Testing

### Manual Testing

1. Start the API: `python main.py`
2. Visit http://localhost:8000/docs for interactive API testing
3. Use the Swagger UI to test endpoints

### Automated Testing

```bash
# Install testing dependencies
pip install pytest httpx

# Run tests (create test files as needed)
pytest
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions:
1. Check the API documentation at `/docs`
2. Review error logs in the console
3. Ensure all environment variables are set correctly
4. Verify the RAG system database is accessible

---

**Happy coding! 🎉**
