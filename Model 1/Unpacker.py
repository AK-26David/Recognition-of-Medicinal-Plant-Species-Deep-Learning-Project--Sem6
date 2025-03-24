import mimetypes

file_path = "/content/Run_1.zip"
mime_type, _ = mimetypes.guess_type(file_path)
print(f"Detected MIME type: {mime_type}")

from zipfile import ZipFile

zip_file_path = "/content/Run_1.zip"  # Replace with your uploaded file path
extract_dir = "/content/dataset"  # Target directory for extraction

with ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

print("Extraction complete!")
