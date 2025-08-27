from election_pdf_analyzer import (
    extract_pdf_metadata,
    convert_pdf_to_images,
    analyze_image_with_groq,
)
from enhanced_corruption_analysis import (
    load_voter_data,
    detect_voter_id_anomalies,
)
from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import os
import dotenv
import shutil
import asyncio
import json

dotenv.load_dotenv()

app = FastAPI()

# --- Enable CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Route 1: Upload & analyze a voter data JSON ---
@app.post("/api/analyze-json")
async def analyze_json(request: Request):
    try:
        data = await request.json()
        if not data:
            raise HTTPException(
                status_code=400, detail="No JSON data provided")

        df = load_voter_data(data)
        if df.empty:
            raise HTTPException(
                status_code=400, detail="Failed to load voter data")

        results = detect_voter_id_anomalies(df)
        return JSONResponse(content=results)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def event_generator(file_path: str, language: str):
    try:
        yield f"data: {json.dumps({'status': 'starting', 'language': language})}\n\n"

        # Step 1: Metadata
        metadata = extract_pdf_metadata(file_path)
        yield f"data: {json.dumps({'status': 'metadata_extracted', 'metadata': metadata})}\n\n"

        # Step 2: Convert PDF to images
        image_paths, output_folder = convert_pdf_to_images(
            file_path, start_page=3)
        yield f"data: {json.dumps({'status': 'images_converted', 'pages': len(image_paths)})}\n\n"

        if not image_paths:
            yield f"data: {json.dumps({'status': 'error', 'error': 'No images generated'})}\n\n"
            return

        all_pages = []
        total_voters = 0

        for i, img in enumerate(image_paths, 1):
            yield f"data: {json.dumps({'status': 'processing_page', 'page': i})}\n\n"

            # 👇 pass language down into your analysis
            voter_data = analyze_image_with_groq(
                img, os.getenv("GROQ_API_KEY"), language=language)
            count = len(voter_data.get("voters", []))
            total_voters += count
            all_pages.append(voter_data["voters"])
            yield f"data: {json.dumps({'status': 'page_done', 'page': i, 'voter_count': count})}\n\n"
            await asyncio.sleep(0)

        flat_data = [item for sublist in all_pages for item in sublist]
        yield f"data: {json.dumps({'status': 'completed', 'total_voters': total_voters, 'pages': len(all_pages), 'data': flat_data})}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'status': 'error', 'error': str(e)})}\n\n"

# --- Streaming route for PDF analysis ---


@app.post("/api/analyze-pdf")
async def analyze_pdf(file: UploadFile = File(...), language: str = Form("english")):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files allowed")

    os.makedirs("uploads", exist_ok=True)
    upload_path = os.path.join("uploads", file.filename)

    # Save uploaded file
    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Return live updates via SSE
    return StreamingResponse(event_generator(upload_path, language), media_type="text/event-stream")


# --- Health check ---
@app.get("/health")
async def health_check():
    return {"status": "healthy"}
