import os
import tempfile
import uuid

import aiofiles
import aiohttp
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


async def download_pdf_async(url: str, temp_dir: str) -> str:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    raise HTTPException(status_code=400, detail=f"Ошибка загрузки PDF: статус {response.status}")

                content = await response.read()

                file_path = os.path.join(temp_dir, f"{str(uuid.uuid4())}.pdf")
                async with aiofiles.open(file_path, "wb") as f:
                    await f.write(content)

                return file_path
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка загрузки PDF: {str(e)}")


@app.post("/extract-text")
async def extract_text_from_pdf(inp: URLInput) -> TextOutput:
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            print(f"processing url={inp.url}")
            pdf_path = await download_pdf_async(inp.url, temp_dir)

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
