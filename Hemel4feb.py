import pdfplumber
import pandas as pd
import re

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file, handling errors gracefully."""
    data = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text:
                    lines = text.split("\n")
                    lines = [line.strip() for line in lines if line.strip()]  # Remove empty lines
                    data.extend(lines)
                else:
                    print(f"Warning: Page {page_num} has no text.")
    except Exception as e:
        print(f"Error processing PDF: {e}")
    
    return data

def clean_and_structure_data(data):
    """Cleans and structures extracted text into tabular format."""
    structured_data = []
    temp_row = []
    
    for line in data:
        if is_new_record(line):  
            if temp_row:
                structured_data.append(temp_row)
            temp_row = [line]  # Start a new record
        else:
            temp_row.append(line)  # Continue appending to current record
    
    if temp_row:  
        structured_data.append(temp_row)  # Append last record
    
    return structured_data

def is_new_record(line):
    """
    Determines if a new record starts.
    Example: Lines starting with uppercase words or specific patterns.
    """
    return bool(re.match(r'^[A-Z][A-Za-z0-9\s,:-]+$', line))  # Adjust regex as needed

def save_to_csv(data, output_csv):
    """Saves structured data into a CSV file."""
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False, header=False, encoding='utf-8')
    print(f"CSV file saved successfully: {output_csv}")

if __name__ == "__main__":
    pdf_path = "/home/himel/Desktop/lab7/15_2020-08-20_5f3ec169648f5.pdf"  # Path to the uploaded PDF
    output_csv = "/home/himel/Desktop/lab7/output.csv"  # Save path for the CSV file
    
    raw_data = extract_text_from_pdf(pdf_path)
    
    if not raw_data:
        print("No text extracted from PDF.")
    else:
        structured_data = clean_and_structure_data(raw_data)
        save_to_csv(structured_data, output_csv)

    print("Conversion completed successfully!")
