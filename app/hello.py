from fastapi import FastAPI

app = FastAPI()

@app.get("/")

def read_root():
    return{"message" : "Hola gente desde FastAPI"}