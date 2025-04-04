import os
import tempfile
import uuid

import requests
from fastapi import FastAPI, HTTPException
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from pydantic import BaseModel

app = FastAPI()

converter = PdfConverter(artifact_dict=create_model_dict())


class URLInput(BaseModel):
    url: str


class TextOutput(BaseModel):
    text: str


def download_pdf(url: str, temp_dir: str) -> str:
    try:
        response = requests.get(url)
        response.raise_for_status()

        file_path = os.path.join(temp_dir, f"{str(uuid.uuid4())}.pdf")
        with open(file_path, "wb") as f:
            f.write(response.content)
        return file_path
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка загрузки PDF: {str(e)}")


@app.post("/extract-text")
async def extract_text_from_pdf(inp: URLInput) -> TextOutput:
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            print(f"processing url={inp.url}")
            pdf_path = download_pdf(inp.url, temp_dir)

            rendered = converter(pdf_path)
            text, _, _ = text_from_rendered(rendered)
            print(f"done url={inp.url} ")
            output = TextOutput(text=text)
            return output
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Ошибка обработки PDF: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
