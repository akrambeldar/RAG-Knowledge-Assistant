from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_chain import rag_chain, retriever

app = FastAPI(title="RAG Knowledge Assistant", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str

@app.get("/")
def root():
    return {"status": "RAG API running"}

@app.post("/query")
async def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    try:
        docs = retriever.invoke(req.question)
        answer = rag_chain.invoke(req.question)
        return {
            "answer": answer,
            "sources": [
                {
                    "content": d.page_content[:300],
                    "source": d.metadata.get("source", "unknown")
                }
                for d in docs
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/stream")
async def query_stream(req: QueryRequest):
    def generate():
        for chunk in rag_chain.stream(req.question):
            yield chunk
    return StreamingResponse(generate(), media_type="text/plain")
