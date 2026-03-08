import PyPDF2
from io import BytesIO
import re

pdf_reader = PyPDF2.PdfReader("2305.12716.pdf")
text = ""
for i in range(len(pdf_reader.pages)):
    text += pdf_reader.pages[i].extract_text() + "\n"

# Try to find the specific section about closed-form matrix
match = re.search(r'(?i)closed-form.*?(\n.*){1,30}', text)
if match:
    print(match.group(0))
