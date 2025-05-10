# backend.py
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directory if needed
app.mount("/static", StaticFiles(directory="static"), name="static")

table_data = [
    ["R1C1", "R1C2", "R1C3", "R1C4", "R1C5"],
    ["R2C1", "R2C2", "R2C3", "R2C4", "R2C5"],
    ["R3C1", "R3C2", "R3C3", "R3C4", "R3C5"],
    ["R4C1", "R4C2", "R4C3", "R4C4", "R4C5"],
    ["R5C1", "R5C2", "R5C3", "R5C4", "R5C5"],
]

# Create 20 click-info values per tab (tab 1 = index 1, tab 2 = index 2)
# info_texts will post the benchmarking result as well as the prompt/question(s) with the response!
info_texts = {
    f"{tab}-{row}-{col}": f"Details for Tab {tab+1}, Row {row+1}, Column {col+1}"
    for tab in [1, 2] for row in range(5) for col in range(1, 5)
}


@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/astrobench_table.html") as f:
        return HTMLResponse(content=f.read())

@app.get("/data")
async def get_table_data():
    return {
        "rows": table_data,
        "texts": info_texts
    }

if __name__ == "__main__":
    print("Visit app at http://localhost:8001/ai4hrp")
    uvicorn.run(app,
        host="0.0.0.0",
        port=8001,
        reload=False,
    )