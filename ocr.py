import asyncio
import base64
import io

import ollama
from pdf2image import convert_from_path

SYSTEM_PROMPT = """Act as an OCR assistant. Analyze the provided image and:
1. Recognize all visible text in the image as accurately as possible.
2. Maintain the original structure and formatting of the text.
3. If any words or phrases are unclear, indicate this with [unclear] in your transcription.
Provide only the transcription without any additional comments."""


def encode_image_to_base64(image):
    """Convert a PIL image to a base64 encoded string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


async def perform_ocr_async(image, page_num, semaphore):
    """Perform OCR on the given PIL image using Llama 3.2-Vision with semaphore to limit concurrent requests."""
    async with semaphore:
        print(f"Processing page {page_num}...")
        # Save image to a temporary buffer to get the bytes
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        image_bytes = buffered.getvalue()

        # Use Ollama library
        try:
            response = ollama.chat(
                model="llama3.2-vision",
                messages=[
                    {"role": "user", "content": SYSTEM_PROMPT, "images": [image_bytes]}
                ],
            )

            # Extract the model's response
            if response and "message" in response and "content" in response["message"]:
                return page_num, response["message"]["content"].strip()
            else:
                print(
                    f"Error: Unexpected response format from Ollama for page {page_num}"
                )
                return page_num, "[OCR FAILED: Unexpected response format]"
        except Exception as e:
            print(f"Error processing page {page_num}: {str(e)}")
            return page_num, f"[OCR FAILED: {str(e)}]"


async def process_pdf_async(pdf_path, output_file=None, max_concurrent=4):
    """Process a PDF file by converting each page to an image and performing OCR asynchronously."""
    print(f"Processing PDF: {pdf_path}")

    # Convert PDF to images in memory
    images = convert_from_path(pdf_path)[0:4]
    print(f"PDF has {len(images)} pages")

    # Create a semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)

    # Create tasks for each page
    tasks = []
    for i, image in enumerate(images):
        task = perform_ocr_async(image, i + 1, semaphore)
        tasks.append(task)

    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks)

    # Sort results by page number and combine text
    results.sort(key=lambda x: x[0])  # Sort by page number
    all_text = [f"--- PAGE {page_num} ---\n{text}\n" for page_num, text in results]
    combined_text = "\n".join(all_text)

    # Output results
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(combined_text)
        print(f"OCR results saved to {output_file}")
    else:
        print("\nOCR Recognition Results:")
        print(combined_text)

    return combined_text


async def main(pdf_path, output_file=None, max_concurrent=4):
    """
    Main function to process PDF with OCR using specified parameters.

    Args:
        pdf_path (str): Path to the PDF file
        output_file (str, optional): Path to save the OCR results. If None, prints to console.
        max_concurrent (int, optional): Maximum number of concurrent OCR operations. Defaults to 4.
    """
    print(f"Using max {max_concurrent} concurrent requests")
    return await process_pdf_async(pdf_path, output_file, max_concurrent)


if __name__ == "__main__":
    # Example usage
    pdf_input = "data/Building-the-Future-of-UX-in-a-World-of-Generative-AI.pdf"  # Change this to your PDF path
    txt_output = "data/Building-the-Future-of-UX-in-a-World-of-Generative-AI.txt"  # Change this or set to None to print to console
    concurrent_jobs = 4  # Adjust based on your system capabilities

    asyncio.run(main(pdf_input, txt_output, concurrent_jobs))
