import pandas as pd

def read_dataset(file_path: str):
    encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
    
    for encoding in encodings:
        try:
            if file_path.endswith(".csv"):
                return pd.read_csv(file_path, encoding=encoding)
            elif file_path.endswith((".xlsx", ".xls")):
                return pd.read_excel(file_path)
            else:
                raise ValueError("Unsupported file format")
        except UnicodeDecodeError:
            continue
        except Exception as e:
            raise ValueError(f"File reading failed: {str(e)}")
    
    raise ValueError("Could not decode the file with known encodings")
