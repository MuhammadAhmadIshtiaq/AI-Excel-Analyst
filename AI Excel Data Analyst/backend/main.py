from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import shutil
import os
import uuid
from engine import clean_data, profile_data, compute_kpis, detect_anomalies, run_query_engine
from scorecard import financial_scorecard   # ← only new import

app = FastAPI(title="AI Data Analyst API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# 📁  UPLOAD  (unchanged)
# ─────────────────────────────────────────────
@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)):
    ext = file.filename.split(".")[-1].lower()
    if ext not in ("csv", "xlsx", "xls"):
        raise HTTPException(status_code=400, detail="Only CSV or XLSX files are supported.")

    dataset_id = str(uuid.uuid4())
    file_path  = os.path.join(UPLOAD_DIR, f"{dataset_id}.{ext}")

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        df = pd.read_csv(file_path) if ext == "csv" else pd.read_excel(file_path)
        df = clean_data(df)

        if ext == "csv":
            df.to_csv(file_path, index=False)
        else:
            df.to_excel(file_path, index=False)

        return {
            "dataset_id": f"{dataset_id}.{ext}",
            "filename":   file.filename,
            "profile":    profile_data(df),
            "kpis":       compute_kpis(df),
            "anomalies":  detect_anomalies(df),
            "sample":     df.head(5).astype(str).to_dict(orient="records"),
        }
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Upload processing failed: {str(e)}")


# ─────────────────────────────────────────────
# 💬  QUERY  (unchanged)
# ─────────────────────────────────────────────
@app.post("/api/query")
async def query_dataset(dataset_id: str = Form(...), query: str = Form(...)):
    file_path = os.path.join(UPLOAD_DIR, dataset_id)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Dataset not found. Please re-upload your file.")

    try:
        ext = dataset_id.split(".")[-1].lower()
        df  = pd.read_csv(file_path) if ext == "csv" else pd.read_excel(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read dataset: {str(e)}")

    result = run_query_engine(query, df)

    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])

    return result


# ─────────────────────────────────────────────
# 📊  SCORECARD  (new — isolated)
# ─────────────────────────────────────────────
@app.post("/api/scorecard")
async def scorecard_endpoint(dataset_id: str = Form(...)):
    file_path = os.path.join(UPLOAD_DIR, dataset_id)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Dataset not found. Please re-upload your file.")

    try:
        ext = dataset_id.split(".")[-1].lower()
        df  = pd.read_csv(file_path) if ext == "csv" else pd.read_excel(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read dataset: {str(e)}")

    result = financial_scorecard(df)

    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])

    return result


# ─────────────────────────────────────────────
# ❤️  HEALTH CHECK
# ─────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
# from fastapi import FastAPI, UploadFile, File, HTTPException, Form
# from fastapi.middleware.cors import CORSMiddleware
# import pandas as pd
# import shutil
# import os
# import uuid
# from engine import clean_data, profile_data, compute_kpis, detect_anomalies, run_query_engine

# app = FastAPI(title="AI Data Analyst API")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],   # restrict to your frontend URL in production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# UPLOAD_DIR = "uploads"
# os.makedirs(UPLOAD_DIR, exist_ok=True)


# # ─────────────────────────────────────────────
# # 📁  UPLOAD ENDPOINT
# # ─────────────────────────────────────────────
# @app.post("/api/upload")
# async def upload_dataset(file: UploadFile = File(...)):
#     ext = file.filename.split(".")[-1].lower()
#     if ext not in ("csv", "xlsx", "xls"):
#         raise HTTPException(status_code=400, detail="Only CSV or XLSX files are supported.")

#     dataset_id = str(uuid.uuid4())
#     file_path  = os.path.join(UPLOAD_DIR, f"{dataset_id}.{ext}")

#     # Save raw upload
#     with open(file_path, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)

#     try:
#         # Load & clean
#         if ext == "csv":
#             df = pd.read_csv(file_path)
#         else:
#             df = pd.read_excel(file_path)

#         df = clean_data(df)

#         # Persist cleaned version
#         if ext == "csv":
#             df.to_csv(file_path, index=False)
#         else:
#             df.to_excel(file_path, index=False)

#         profile   = profile_data(df)
#         kpis      = compute_kpis(df)
#         anomalies = detect_anomalies(df)
#         sample    = df.head(5).astype(str).to_dict(orient="records")

#         return {
#             "dataset_id": f"{dataset_id}.{ext}",
#             "filename":   file.filename,
#             "profile":    profile,
#             "kpis":       kpis,
#             "anomalies":  anomalies,
#             "sample":     sample,
#         }

#     except Exception as e:
#         # Clean up partial file on error
#         if os.path.exists(file_path):
#             os.remove(file_path)
#         raise HTTPException(status_code=500, detail=f"Upload processing failed: {str(e)}")


# # ─────────────────────────────────────────────
# # 💬  QUERY ENDPOINT
# # ─────────────────────────────────────────────
# @app.post("/api/query")
# async def query_dataset(dataset_id: str = Form(...), query: str = Form(...)):
#     file_path = os.path.join(UPLOAD_DIR, dataset_id)

#     if not os.path.exists(file_path):
#         raise HTTPException(status_code=404, detail="Dataset not found. Please re-upload your file.")

#     try:
#         ext = dataset_id.split(".")[-1].lower()
#         if ext == "csv":
#             df = pd.read_csv(file_path)
#         else:
#             df = pd.read_excel(file_path)

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to read dataset: {str(e)}")

#     # Run the full AI pipeline — this never raises, errors come back in the dict
#     result = run_query_engine(query, df)

#     # If the engine returned an error, send it back as a 422 with the message
#     # so the frontend can display it clearly instead of a generic "something went wrong"
#     if "error" in result:
#         raise HTTPException(status_code=422, detail=result["error"])

#     return result


# # ─────────────────────────────────────────────
# # ❤️  HEALTH CHECK
# # ─────────────────────────────────────────────
# @app.get("/api/health")
# async def health():
#     return {"status": "ok"}


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)