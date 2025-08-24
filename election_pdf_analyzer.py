import os
import json
import base64
from pathlib import Path
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
from groq import Groq

# ==================== CONFIGURATION ====================
# Configure these variables according to your needs
PDF_PATH = r"list.pdf"  # Change this to your PDF path

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

LANGUAGE = "Hindi"  # Options: "English" or "Hindi"
OUTPUT_JSON = "data2.json"  # Output JSON file name
# ==================== END CONFIGURATION ================


def encode_image(image_path):
    """Encode image to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def extract_pdf_metadata(pdf_path):
    """Extract constituency and district info from first 2 pages"""
    try:
        reader = PdfReader(pdf_path)
        metadata = {"constituency": "", "district": "",
                    "total_pages": len(reader.pages)}

        # Extract text from first 2 pages
        for page_num in range(min(2, len(reader.pages))):
            page = reader.pages[page_num]
            text = page.extract_text()
            print(f"Extracting metadata from page {page_num + 1}")

            # Look for common patterns in election documents
            lines = text.split('\n')
            for line in lines:
                line = line.strip().upper()
                if 'CONSTITUENCY' in line or 'निर्वाचन क्षेत्र' in line:
                    metadata["constituency"] = line
                elif 'DISTRICT' in line or 'जिला' in line:
                    metadata["district"] = line

        return metadata
    except Exception as e:
        print(f"Error extracting metadata: {e}")
        return {"constituency": "Unknown", "district": "Unknown", "total_pages": 0}


def create_output_folder(pdf_path):
    """Create output folder named after PDF file"""
    pdf_name = Path(pdf_path).stem
    output_folder = f"{pdf_name}_images"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    return output_folder


def convert_pdf_to_images(pdf_path, start_page=3):
    """Convert PDF pages to JPG images starting from specified page"""
    try:
        output_folder = create_output_folder(pdf_path)
        image_paths = []

        # Convert PDF to images with 300 DPI for good quality
        images = convert_from_path(
            pdf_path,
            dpi=300,
            first_page=start_page,
            fmt='jpeg',
            thread_count=4,
        )

        total_pages = len(images) + start_page - 1
        print(f"Total pages in PDF: {total_pages}")
        print(f"Converting pages {start_page} to {total_pages} to images...")

        # Save each image
        for i, image in enumerate(images, start=start_page):
            image_path = os.path.join(output_folder, f"page_{i}.jpg")
            image.save(image_path, 'JPEG', quality=95)
            image_paths.append(image_path)
            print(f"Converted page {i} to {image_path}")

        print(f"Successfully converted {len(image_paths)} pages to images")
        return image_paths, output_folder

    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        return [], ""


def analyze_image_with_groq(image_path, api_key):
    """Analyze image using Groq API to extract voter data"""
    try:
        base64_image = encode_image(image_path)
        client = Groq(api_key=api_key)

        prompt = f"""
        Analyze this Indian election roll/voter list image in {LANGUAGE}. Extract all voter information visible in the image.
        
        Each voter entry contains:
        - Serial Number (top left corner in a box)
        - Name (after "Name :")
        - Father Name (after "Father Name :")
        - House Number (after "House Number :")
        - Age (after "Age :")
        - Gender (after "Gender :")
        - Voter ID (top right corner, format like YNK0869982)
        - Photo status (right side - "Photo Available" or similar)
        
        Extract ALL visible voter entries from this image. Return the data in this exact JSON format:
        {{
            "voters": [
                {{
                    "serial_number": "7",
                    "name": "Rajeswari",
                    "father_name": "kanagaraj",
                    "house_number": "1",
                    "age": "55",
                    "gender": "Female",
                    "voter_id": "YNK0869982",
                    "photo_status": "Photo Available"
                }}
            ]
        }}
        
        Important:
        - Extract data exactly as shown in the image
        - If any field is not visible, use empty string ""
        - Include all voters visible in the image
        - If no voter data is found, return {{"voters": []}}
        - Do not add any extra text, only return valid JSON
        """

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
            model="meta-llama/llama-4-scout-17b-16e-instruct",
        )

        response_text = chat_completion.choices[0].message.content
        print(f"Groq API response for {os.path.basename(image_path)}: Success")

        # Try to parse JSON response
        try:
            # Extract JSON from response if it's wrapped in text
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                json_text = response_text[start_idx:end_idx]
                voter_data = json.loads(json_text)
                return voter_data
            else:
                return {"voters": [], "raw_response": response_text}
        except json.JSONDecodeError:
            return {"voters": [], "raw_response": response_text}

    except Exception as e:
        print(f"Error analyzing image {image_path}: {e}")
        return {"voters": [], "error": str(e)}


def save_to_json(data, json_file_path):
    """Save or append data to JSON file"""
    try:
        # Initialize default structure
        default_structure = {
            "metadata": {
                "constituency": "",
                "district": "",
                "total_pages": 0
            },
            "all_voters": []
        }

        # Load existing data if file exists, otherwise use default structure
        if os.path.exists(json_file_path):
            try:
                with open(json_file_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                # Ensure the structure is correct
                if "all_voters" not in existing_data:
                    existing_data["all_voters"] = []
                if "metadata" not in existing_data:
                    existing_data["metadata"] = default_structure["metadata"]
            except json.JSONDecodeError:
                existing_data = default_structure
        else:
            existing_data = default_structure

        # Update metadata if provided
        if "metadata" in data:
            existing_data["metadata"].update(data["metadata"])

        # Add voters data
        if "voters" in data and isinstance(data["voters"], list):
            # Transform each voter entry to match the desired structure
            for voter in data["voters"]:
                formatted_voter = {
                    "serial_number": str(voter.get("serial_number", "")),
                    "name": str(voter.get("name", "")),
                    "father_husband_name": str(voter.get("father_name", "")),
                    "age": str(voter.get("age", "")),
                    "gender": str(voter.get("gender", "")),
                    "house_number": str(voter.get("house_number", "")),
                    "voter_id": str(voter.get("voter_id", "")),
                    "photo_status": str(voter.get("photo_status", ""))
                }
                if not existing_data.get("all_voters"):
                    existing_data["all_voters"] = []
                existing_data["all_voters"].append(formatted_voter)

        # Save updated data
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)

        print(f"Data saved to {json_file_path}")

    except Exception as e:
        print(f"Error saving to JSON: {e}")
        # In case of error, try to at least save the voter data
        try:
            if "voters" in data and isinstance(data["voters"], list):
                with open(json_file_path, 'w', encoding='utf-8') as f:
                    json.dump(
                        {"metadata": default_structure["metadata"], "all_voters": data["voters"]}, f, ensure_ascii=False, indent=2)
        except:
            pass  # If even the recovery save fails, we've already printed the original error


def process_all_images(image_paths, output_folder, api_key):
    """Process all images and extract voter data from each page"""
    print(f"Processing {len(image_paths)} images...")

    all_voters_data = []
    total_voters = 0

    for i, image_path in enumerate(image_paths, 1):
        print(
            f"\nProcessing image {i}/{len(image_paths)}: {os.path.basename(image_path)}")

        # Analyze image with Groq
        voter_data = analyze_image_with_groq(image_path, api_key)

        # Add page information
        page_info = {
            "image_file": os.path.basename(image_path),
            "page_number": i + 2,  # +2 because we skip first 2 pages
            "processed_at": f"Image {i} of {len(image_paths)}"
        }

        # Count voters
        voters_in_page = len(voter_data.get("voters", []))
        total_voters += voters_in_page

        # Attach page info to result
        voter_data["page_info"] = page_info
        voter_data["voter_count"] = voters_in_page

        all_voters_data.append(voter_data)

        print(
            f"Extracted {voters_in_page} voters from this page. Total so far: {total_voters}")
        print(f"Completed processing {os.path.basename(image_path)}")

    print(f"\n=== Processing Complete ===")
    print(f"Total voters extracted: {total_voters}")
    print(f"Data saved in: {OUTPUT_JSON}")

    return {
        "pages": all_voters_data,
        "total_voters": total_voters
    }


def main():
    """Main function to orchestrate the entire process"""
    print("=== Election PDF Analyzer ===")
    print(f"PDF Path: {PDF_PATH}")
    print(f"Language: {LANGUAGE}")
    print(f"Output JSON: {OUTPUT_JSON}")

    # Check if PDF exists
    if not os.path.exists(PDF_PATH):
        print(f"Error: PDF file not found at {PDF_PATH}")
        return

    try:
        # Step 1: Extract metadata from first 2 pages
        print("\n=== Step 1: Extracting Metadata ===")
        metadata = extract_pdf_metadata(PDF_PATH)
        print(f"Constituency: {metadata['constituency']}")
        print(f"District: {metadata['district']}")
        print(f"Total Pages: {metadata['total_pages']}")

        # Save metadata to JSON
        save_to_json({"metadata": metadata}, OUTPUT_JSON)

        # Step 2: Convert PDF pages to images
        print("\n=== Step 2: Converting PDF to Images ===")
        image_paths, output_folder = convert_pdf_to_images(
            PDF_PATH, start_page=3)

        if not image_paths:
            print("No images were created. Exiting.")
            return

        # Step 3: Process all images with Groq API
        print("\n=== Step 3: Processing Images with Groq API ===")
        process_all_images(image_paths, output_folder, GROQ_API_KEY)

        print("\n=== All Done! ===")
        print(f"Check your results in:")
        print(f"- Images folder: {output_folder}")
        print(f"- JSON data: {OUTPUT_JSON}")

    except Exception as e:
        print(f"Error in main process: {e}")


if __name__ == "__main__":
    main()
