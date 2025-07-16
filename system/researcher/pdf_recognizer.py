import os
import tempfile
import uuid

import aiofiles
import aiohttp
from fastapi import FastAPI, HTTPException
import fitz
from pydantic import BaseModel


app = FastAPI()


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


def converter(pdf_document_path):
    doc = fitz.open(pdf_document_path)

    # Initialize an empty string to store extracted text
    extracted_text = ""

    # Iterate through each page and extract text
    for page_num in range(doc.page_count):
        page = doc[page_num]
        extracted_text += page.get_text()

    # Close the PDF document
    doc.close()

    return extracted_text


async def extract_text_from_pdf(inp: URLInput) -> TextOutput:
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            print(f"processing url={inp.url}")
            pdf_path = await download_pdf_async(inp.url, temp_dir)

            text = converter(pdf_path)
            print(f"done url={inp.url} ")
            output = TextOutput(text=text)
            return output
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Ошибка обработки PDF: {str(e)}")

