from fastapi import FastAPI

app = FastAPI(title="Meridian Engine")

@app.websocket("/ws")
async def read_root():
    return {"Hello": "World"}
