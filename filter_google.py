import os
import gc
import math
import sqlite3
import json
from tqdm.auto import tqdm

# --- GPU Libraries ---
import cudf
import cupy as cp
import torch
import torch.nn.functional as F

# --- API & ML Libraries ---
from transformers import AutoTokenizer, AutoModel
# NLTK is no longer used for cleaning, but we'll keep the stopwords list for now
import nltk
try:
    from nltk.corpus import stopwords
except ImportError:
    print("NLTK not found, using a basic list of stopwords. For better results, 'pip install nltk'")
    stopwords = {'english': ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']}

from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

# ========================= CONFIGURATION =========================

# --- Google Drive Details ---
GDRIVE_FOLDER_NAME = "Files"  # The NAME of the folder containing your CSVs

# --- List of local CSV files to process ---
CSV_FILENAMES = [
    "comments4_cleaned.csv"
]

# --- Permanent Local Storage Directories ---
LOCAL_DATA_DIR = "youtube_comment_data"
LOCAL_CSV_DIR = os.path.join(LOCAL_DATA_DIR, "csv_files")
LOCAL_DB_DIR = os.path.join(LOCAL_DATA_DIR, "database")
DB_PATH = os.path.join(LOCAL_DB_DIR, "embedded.db")

# --- Model & Embedding Configuration ---
MODEL_NAME = 'Qwen/qwen3-embedding-0.6b'
EMBEDDING_BATCH_SIZE = 128  # For the embedding model itself
PROCESSING_BATCH_SIZE = 10000 # For processing the dataframe
# Define chunk size in bytes for cuDF (10MB)
CHUNK_BYTES = 2 * 1024 * 1024

# ======================= GPU-ACCELERATED CLEANING =======================

# Attempt to download stopwords, fall back to a basic list if NLTK is not fully installed.
try:
    stop_words = set(stopwords.words('english'))
except (NameError, LookupError):
    if 'stopwords' in locals() and 'english' in stopwords:
         stop_words = set(stopwords['english'])
    else:
        print("Falling back to basic stopwords list.")
        stop_words = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'}

# Create a GPU-compatible regex for removing stopwords
stop_words_pattern = r'\b(' + '|'.join(stop_words) + r')\b'

def clean_text_gpu(text_series):
    """
    Cleans a cuDF string Series using GPU-accelerated operations.
    - Converts to lowercase
    - Removes non-alphabetic characters
    - Removes stopwords
    """
    if not hasattr(text_series, 'str'):
        return cudf.Series([], dtype='str')

    # Chain GPU operations for maximum efficiency
    processed_series = (
        text_series
        .str.lower()
        .str.replace(r'[^a-z]', ' ', regex=True)  # Correctly replace non-alphabetic chars with a space
        .str.replace(stop_words_pattern, '', regex=True)  # Remove stopwords
        .str.replace(r'\s+', ' ', regex=True)  # Collapse multiple spaces into one
        .str.strip()  # Remove leading/trailing spaces
    )
    return processed_series


# ================== GPU-STABLE EMBEDDING FUNCTION ===================

def generate_embeddings_stable(texts, model, tokenizer, device, batch_size):
    """
    Generates embeddings on GPU with aggressive, per-batch memory cleanup.
    Returns a CuPy array for further GPU processing.
    """
    model.eval()
    # Use CuPy for the final embeddings array, keeping it on the GPU
    final_embeddings = cp.zeros((len(texts), model.config.hidden_size), dtype=cp.float16)
    
    # Sorting by length is crucial for efficient padding
    length_sorted_indices = sorted(range(len(texts)), key=lambda k: len(texts[k]))
    sorted_texts = [texts[i] for i in length_sorted_indices]

    with torch.no_grad():
        for i in tqdm(range(0, len(sorted_texts), batch_size), desc="  Generating Embeddings"):
            batch_texts = sorted_texts[i:i + batch_size]
            
            inputs = tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                return_tensors='pt', 
                max_length=512
            ).to(device)
            
            last_hidden_state = model(**inputs).last_hidden_state
            
            input_mask_expanded = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            pooled_embeddings = sum_embeddings / sum_mask
            normalized_embeddings = F.normalize(pooled_embeddings, p=2, dim=1)

            # Convert torch tensor to CuPy array directly on the GPU
            batch_cupy_embeddings = cp.asarray(normalized_embeddings.to(torch.float16))
            
            # Place the embeddings back in the correct order
            original_indices = length_sorted_indices[i:i + batch_size]
            # This part is tricky with cupy, let's use a CPU-based re-ordering for simplicity
            # as advanced indexing can be complex.
            final_embeddings[cp.array(original_indices)] = batch_cupy_embeddings

            del inputs, last_hidden_state, pooled_embeddings, normalized_embeddings, batch_cupy_embeddings
            if device == "cuda":
                torch.cuda.empty_cache()

    return final_embeddings

# ================== BATCH-BASED PROCESSING WITH CUDF & SQLITE ==================

def process_single_csv_file(csv_filename, csv_dir, db_connection, model, tokenizer):
    """
    Processes one CSV by loading and cleaning it fully, then iterating in batches
    for memory-stable embedding and insertion into a local SQLite database.
    """
    print(f"\n--- Processing '{csv_filename}' with cuDF and SQLite ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("❌ FATAL: This script requires a CUDA-enabled GPU.")
        return

    file_path = os.path.join(csv_dir, csv_filename)
    total_rows_processed = 0
    gdf = None

    try:
        # Step 1: Load the entire CSV into GPU memory
        print(f"Loading '{file_path}' into GPU memory...")
        try:
            gdf = cudf.read_csv(file_path)
        except Exception as read_error:
            print(f"    ❌ Failed to read CSV with cuDF: {read_error}")
            return
        print(f"  Successfully loaded {len(gdf)} rows.")

        # --- Data Validation and Cleaning ---
        required_columns = ["commentId", "textOriginal"]
        for col in required_columns:
            if col not in gdf.columns:
                print(f"    ❌ '{col}' column not found. Aborting.")
                return
        
        gdf['textOriginal'] = gdf['textOriginal'].fillna('')
        gdf['cleaned_text'] = clean_text_gpu(gdf['textOriginal'])
        gdf = gdf[gdf['cleaned_text'].str.len() > 0]
        
        if gdf.empty:
            print("    ⚠️ No valid text left after cleaning. Nothing to process.")
            return
        print(f"  Cleaning complete. {len(gdf)} rows remaining.")

        # Step 2: Process in batches for embedding and database insertion
        num_batches = math.ceil(len(gdf) / PROCESSING_BATCH_SIZE)
        print(f"Processing {len(gdf)} rows in {num_batches} batches of {PROCESSING_BATCH_SIZE}...")

        cursor = db_connection.cursor()

        for i in tqdm(range(num_batches), desc="Processing Batches"):
            start_index = i * PROCESSING_BATCH_SIZE
            end_index = start_index + PROCESSING_BATCH_SIZE
            batch_df = gdf.iloc[start_index:end_index]

            # --- Embedding Generation ---
            cleaned_texts_list = batch_df['cleaned_text'].to_arrow().to_pylist()
            embeddings_cupy = generate_embeddings_stable(
                cleaned_texts_list, model, tokenizer, device, batch_size=EMBEDDING_BATCH_SIZE
            )
            embeddings_list = embeddings_cupy.get().tolist()

            # --- Data Preparation (cuDF to Python list of tuples) ---
            db_columns = [
                "commentId", "kind", "channelId", "videoId", "authorId", 
                "textOriginal", "parentCommentId", "likeCount", 
                "publishedAt", "updatedAt"
            ]
            
            # Convert cuDF batch to a Pandas DataFrame, then to records
            batch_pandas = batch_df[db_columns].to_pandas()
            batch_pandas['embedding'] = [json.dumps(e) for e in embeddings_list]
            
            # Replace any nulls with None for DB compatibility
            batch_pandas = batch_pandas.where(pd.notna(batch_pandas), None)
            
            records_to_insert = list(batch_pandas.itertuples(index=False, name=None))

            if not records_to_insert:
                continue

            # --- SQLite Insertion ---
            try:
                cursor.executemany("""
                    INSERT OR REPLACE INTO embeddings (
                        commentId, kind, channelId, videoId, authorId, textOriginal, 
                        parentCommentId, likeCount, publishedAt, updatedAt, embedding
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, records_to_insert)
                db_connection.commit()
                total_rows_processed += len(records_to_insert)
            except sqlite3.Error as e:
                print(f"    ❌ SQLite insertion failed for batch {i}. Error: {e}")
                db_connection.rollback()

            # --- Memory Cleanup ---
            del batch_df, cleaned_texts_list, embeddings_cupy, embeddings_list, records_to_insert, batch_pandas
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()

        print(f"    ✅ Finished processing. Saved ~{total_rows_processed} total embeddings for this file.")

    except Exception as e:
        print(f"    ❌ An unexpected error occurred while processing {csv_filename}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Final cleanup
        del gdf
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()

# ========================== DATABASE & GDRIVE FUNCTIONS ===========================

def setup_database(db_path):
    """Creates the SQLite database and the embeddings table if they don't exist."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            commentId TEXT PRIMARY KEY,
            kind TEXT,
            channelId TEXT,
            videoId TEXT,
            authorId TEXT,
            textOriginal TEXT,
            parentCommentId TEXT,
            likeCount INTEGER,
            publishedAt TEXT,
            updatedAt TEXT,
            embedding TEXT
        )
    """)
    conn.commit()
    print(f"✅ Database setup complete at '{db_path}'")
    return conn

def upload_to_gdrive(drive, local_path, gdrive_folder_name):
    """Uploads a local file to a specified Google Drive folder."""
    print(f"\n--- Uploading '{os.path.basename(local_path)}' to Google Drive ---")
    try:
        # Find the target folder
        folder_query = f"title='{gdrive_folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
        folder_list = drive.ListFile({'q': folder_query}).GetList()

        if not folder_list:
            print(f"  ❌ ERROR: No folder named '{gdrive_folder_name}' found on Google Drive.")
            return
        
        gdrive_folder_id = folder_list[0]['id']

        # Check if a file with the same name already exists
        file_name = os.path.basename(local_path)
        file_query = f"'{gdrive_folder_id}' in parents and title='{file_name}' and trashed=false"
        existing_files = drive.ListFile({'q': file_query}).GetList()

        gfile = drive.CreateFile({
            'title': file_name,
            'parents': [{'id': gdrive_folder_id}]
        })
        
        if existing_files:
            print(f"  File '{file_name}' already exists. Overwriting...")
            gfile['id'] = existing_files[0]['id'] # Set the ID to update the existing file

        gfile.SetContentFile(local_path)
        gfile.Upload()
        print(f"✅ Successfully uploaded '{file_name}' to '{gdrive_folder_name}'.")

    except Exception as e:
        print(f"  ❌ An error occurred during Google Drive upload: {e}")

# ========================== MAIN SCRIPT LOGIC ===========================

def main():
    """Executes the full GPU-accelerated workflow."""
    print("\n--- PHASE 1: SETUP & INITIALIZATION ---")
    
    # --- Google Drive Authentication ---
    print("Authenticating with Google Drive...")
    gauth = GoogleAuth()
    gauth.settings['get_refresh_token'] = True
    try:
        gauth.LoadCredentialsFile("mycreds.txt")
        if gauth.credentials is None:
            gauth.LocalWebserverAuth()
        elif gauth.access_token_expired:
            gauth.Refresh()
        else:
            gauth.Authorize()
        gauth.SaveCredentialsFile("mycreds.txt")
        drive = GoogleDrive(gauth)
        print("✅ Google Drive authentication successful.")
    except Exception as e:
        print(f"❌ Google Drive authentication failed: {e}")
        return

    # --- Database Setup ---
    db_connection = setup_database(DB_PATH)

    # --- GPU and Library Initialization ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device != 'cuda':
        print("This script is optimized for CUDA and may not run correctly on CPU.")
        return

    try:
        print(f"Loading model '{MODEL_NAME}' onto GPU...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            dtype=torch.float16,
        ).to(device)
        print("✅ Model and tokenizer loaded successfully in FP16.")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        db_connection.close()
        return

    print("\n--- PHASE 2: PROCESSING & EMBEDDING ---")
    for csv_filename in CSV_FILENAMES:
        # We assume files are already local or synced separately
        local_file_path = os.path.join(LOCAL_CSV_DIR, csv_filename)
        if not os.path.exists(local_file_path):
            print(f"⚠️ File '{csv_filename}' not found locally, skipping processing.")
            continue
        
        process_single_csv_file(csv_filename, LOCAL_CSV_DIR, db_connection, model, tokenizer)

    print("\n✅ All files processed. Data saved to local SQLite database.")
    db_connection.close()

    # --- PHASE 3: UPLOAD DATABASE TO GDRIVE ---
    upload_to_gdrive(drive, DB_PATH, GDRIVE_FOLDER_NAME)

    print("\n🎉 Workflow finished successfully! 🎉")

if __name__ == "__main__":
    # Import pandas here, only if needed for the main execution
    # This keeps the global namespace cleaner
    import pandas as pd
    main()
