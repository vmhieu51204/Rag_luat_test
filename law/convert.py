from pdf2docx import Converter
import os

def convert_pdf_to_docx(pdf_path, docx_path):
    """
    Converts a PDF file to a DOCX file.
    """
    # Check if the input PDF exists
    if not os.path.exists(pdf_path):
        print(f"Error: The file '{pdf_path}' does not exist.")
        return

    try:
        print(f"Starting conversion of '{pdf_path}'...")
        
        # Initialize the Converter object
        cv = Converter(pdf_path)
        
        # Convert all pages (start=0, end=None converts the whole document)
        cv.convert(docx_path, start=0, end=None)
        
        # Close the converter to free up resources
        cv.close()
        
        print(f"Success! File saved as '{docx_path}'")
        
    except Exception as e:
        print(f"An error occurred during conversion: {e}")

# --- Example Usage ---
if __name__ == "__main__":
    # Replace these with your actual file paths
    input_pdf = "2025.pdf"   
    output_docx = "2025.docx"
    
    convert_pdf_to_docx(input_pdf, output_docx)