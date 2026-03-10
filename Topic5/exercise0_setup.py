# Install dependencies
# On Colab, these install quickly. Locally, you may already have them.
# Use a kernel-aware install when available; fall back to subprocess otherwise.
try:
    ip = get_ipython()
    ip.run_line_magic('pip', 'install -q torch transformers sentence-transformers faiss-cpu pymupdf accelerate ipyfilechooser openai')
except NameError:
    import subprocess, sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'torch', 'transformers', 'sentence-transformers', 'faiss-cpu', 'pymupdf', 'accelerate', 'ipyfilechooser', 'openai'])
# For Exercise 2 (GPT-4o Mini): add 'openai' to the list above if needed


# =============================================================================
# ENVIRONMENT AND DEVICE DETECTION
# =============================================================================
import os
import sys

# Enable MPS fallback for any PyTorch operations not yet implemented on Metal
# This MUST be set before importing torch
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Prevent kernel crash from duplicate OpenMP libraries (PyTorch + FAISS conflict on macOS)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
from typing import Tuple

def detect_environment() -> str:
    """Detect if we're running on Colab or locally."""
    try:
        import google.colab
        return 'colab'
    except ImportError:
        return 'local'

def get_device() -> Tuple[str, torch.dtype]:
    """
    Detect the best available compute device.

    Priority: CUDA > MPS (Apple Silicon) > CPU

    Returns:
        Tuple of (device_string, recommended_dtype)

    Notes:
        - CUDA: Use float16 for memory efficiency (Tensor Cores optimize this)
        - MPS: Use float32 - Apple Silicon doesn't have the same float16
               optimizations as NVIDIA, and float32 is often faster
        - CPU: Use float32 (float16 not well supported on CPU)
    """
    if torch.cuda.is_available():
        device = 'cuda'
        dtype = torch.float16
        device_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✓ Using CUDA GPU: {device_name} ({memory_gb:.1f} GB)")

    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = 'mps'
        dtype = torch.float32  # float32 is often faster on Apple Silicon!
        print("✓ Using Apple Silicon GPU (MPS)")
        print("  Note: Using float32 (faster than float16 on Apple Silicon)")

    else:
        device = 'cpu'
        dtype = torch.float32
        print("⚠ Using CPU (no GPU detected)")
        print("  Tip: For faster processing, use a machine with a GPU")

    return device, dtype

# Detect environment and device
ENVIRONMENT = detect_environment()
DEVICE, DTYPE = get_device()

print(f"\nEnvironment: {ENVIRONMENT.upper()}")
print(f"Device: {DEVICE}, Dtype: {DTYPE}")

# =============================================================================
# CELL 1: SELECT DOCUMENT SOURCE
# =============================================================================
# This cell either:
#   - Shows a folder picker (Local or Colab+Drive) - NON-BLOCKING
#   - Shows an upload dialog (Colab+Upload) - BLOCKING
#
# If a folder picker is shown, SELECT YOUR FOLDER BEFORE running Cell 2.
# The picker widget is non-blocking, so the code continues before you select.
# =============================================================================

from pathlib import Path

# ------------- COLAB USERS: CONFIGURE HERE -------------
USE_GOOGLE_DRIVE = True  # Set to True to use Google Drive instead of uploading
# -------------------------------------------------------

# Default folder: use Corpora from course project (unzip Corpora.zip first).
_folder_default = Path("Corpora/ModelTService")
DOC_FOLDER = str(_folder_default) if _folder_default.exists() else "documents"
folder_chooser = None  # Will hold the picker widget if used

if ENVIRONMENT == 'colab':
    if USE_GOOGLE_DRIVE:
        # ----- COLAB + GOOGLE DRIVE -----
        # Mount Drive first, then show folder picker
        from google.colab import drive
        print("Mounting Google Drive...")
        drive.mount('/content/drive')
        print("✓ Google Drive mounted\n")

        # Now show folder picker for the Drive
        try:
            from ipyfilechooser import FileChooser

            folder_chooser = FileChooser(
                path='/content/drive/MyDrive',
                title='Select your documents folder in Google Drive',
                show_only_dirs=True,
                select_default=True
            )
            print("📁 Select your documents folder below, then run Cell 2:")
            print("   (The picker is non-blocking - select BEFORE running the next cell)")
            display(folder_chooser)

        except ImportError:
            # Fallback: manual path entry
            print("Folder picker not available.")
            print("Edit DOC_FOLDER below with your Google Drive path, then run Cell 2:")
            DOC_FOLDER = '/content/drive/MyDrive/Corpora'  # ← Edit this!
            print(f"  DOC_FOLDER = '{DOC_FOLDER}'")
    else:
        # ----- COLAB + UPLOAD -----
        # Upload dialog blocks until complete, so DOC_FOLDER is ready when done
        from google.colab import files
        os.makedirs(DOC_FOLDER, exist_ok=True)

        print("Upload your documents (PDF, TXT, or MD):")
        print("(This dialog blocks until upload is complete)\n")
        uploaded = files.upload()

        for filename in uploaded.keys():
            os.rename(filename, f'{DOC_FOLDER}/{filename}')
            print(f"  ✓ Saved: {DOC_FOLDER}/{filename}")

        print(f"\n✓ Upload complete. Run Cell 2 to continue.")

else:
    # ----- LOCAL JUPYTER -----
    # Show folder picker
    print("Running locally\n")

    try:
        from ipyfilechooser import FileChooser

        folder_chooser = FileChooser(
            path=str(Path.home()),
            title='Select your documents folder',
            show_only_dirs=True,
            select_default=True
        )
        print("📁 Select your documents folder below, then run Cell 2:")
        print("   (The picker is non-blocking - select BEFORE running the next cell)")
        display(folder_chooser)

    except ImportError:
        # Fallback: manual path entry
        print("Folder picker not available (ipyfilechooser not installed).")
        print(f"\nUsing default folder: {Path(DOC_FOLDER).absolute()}")
        print("\nTo use a different folder, edit DOC_FOLDER in this cell:")
        print("  DOC_FOLDER = '/path/to/your/documents'")
        os.makedirs(DOC_FOLDER, exist_ok=True)


# =============================================================================
# CELL 2: CONFIRM SELECTION AND LIST DOCUMENTS
# =============================================================================
# If you used a folder picker above, make sure you selected a folder
# BEFORE running this cell. The picker is non-blocking.
# =============================================================================

# Read selection from folder picker (if one was used)
if folder_chooser is not None and folder_chooser.selected_path:
    DOC_FOLDER = folder_chooser.selected_path
    print(f"✓ Using selected folder: {DOC_FOLDER}")
elif folder_chooser is not None:
    print("⚠ No folder selected in picker!")
    print("  Please go back to Cell 1, select a folder, then run this cell again.")
else:
    # No picker used (upload or manual path)
    print(f"✓ Using folder: {DOC_FOLDER}")

# Confirm folder (listing skipped for speed)
doc_path = Path(DOC_FOLDER)
if doc_path.exists():
    print(f"✓ Folder set: {doc_path.absolute()}")
    print("  Run the next cells to load, chunk, and index documents.")
else:
    print(f"⚠ Folder not found: {DOC_FOLDER}")
    print("  Please set DOC_FOLDER in the previous cell and run it again.")


# Exercise 1 (and reuse): Official query lists. Reference: CR Jan 13, 20, 21, 23, 2026.
QUERIES_MODEL_T = [
    "How do I adjust the carburetor on a Model T?",
    "What is the correct spark plug gap for a Model T Ford?",
    "How do I fix a slipping transmission band?",
    "What oil should I use in a Model T engine?",
]
QUERIES_CR = [
    "What did Mr. Flood have to say about Mayor David Black in Congress on January 13, 2026?",
    "What mistake did Elise Stefanik make in Congress on January 23, 2026?",
    "What is the purpose of the Main Street Parity Act?",
    "Who in Congress has spoken for and against funding of pregnancy centers?",
]


import fitz  # PyMuPDF
from typing import List, Tuple

def load_text_file(filepath: str) -> str:
    """Load a plain text file."""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()


def load_pdf_file(filepath: str) -> str:
    """
    Extract text from a PDF with embedded text.

    PyMuPDF reads the text layer directly.
    For scanned PDFs without embedded text, you'd need OCR.
    """
    doc = fitz.open(filepath)
    text_parts = []

    for page_num, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            # Add page marker for debugging/citation
            text_parts.append(f"\n[Page {page_num + 1}]\n{text}")

    doc.close()
    return "\n".join(text_parts)


def load_documents(doc_folder: str) -> List[Tuple[str, str]]:
    """Load all documents from a folder. Returns list of (filename, content)."""
    documents = []
    folder = Path(doc_folder)

    for filepath in folder.rglob("*"):
        try:
            if not filepath.is_file():
                continue
        except OSError:
            continue
        if filepath.suffix.lower() not in ('.pdf', '.txt', '.md', '.text'):
            continue
        try:
            if filepath.suffix.lower() == '.pdf':
                content = load_pdf_file(str(filepath))
            elif filepath.suffix.lower() in ['.txt', '.md', '.text']:
                content = load_text_file(str(filepath))
            else:
                continue

            if content.strip():
                documents.append((filepath.name, content))
                print(f"✓ Loaded: {filepath.name} ({len(content):,} chars)")
        except Exception as e:
            print(f"✗ Error loading {filepath}: {e}")

    return documents


# Load your documents
documents = load_documents(DOC_FOLDER)
print(f"\nLoaded {len(documents)} documents")

if len(documents) == 0:
    print("\n⚠ No documents loaded! Please add PDF or TXT files to the documents folder.")


# Inspect a document to verify loading worked
if documents:
    filename, content = documents[0]
    print(f"First document: {filename}")
    print(f"Total length: {len(content):,} characters")
    print(f"\nFirst 1000 characters:\n{'-'*40}")
    print(content[:1000])


from dataclasses import dataclass

@dataclass
class Chunk:
    """A chunk of text with metadata for tracing back to source."""
    text: str
    source_file: str
    chunk_index: int
    start_char: int
    end_char: int


def chunk_text(
    text: str,
    source_file: str,
    chunk_size: int = 512,
    chunk_overlap: int = 128
) -> List[Chunk]:
    """
    Split text into overlapping chunks.

    We try to break at sentence or paragraph boundaries
    to avoid cutting mid-thought.
    """
    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = start + chunk_size

        # Try to break at a good boundary
        if end < len(text):
            # Look for paragraph break first
            para_break = text.rfind('\n\n', start + chunk_size // 2, end)
            if para_break != -1:
                end = para_break + 2
            else:
                # Look for sentence break
                sentence_break = text.rfind('. ', start + chunk_size // 2, end)
                if sentence_break != -1:
                    end = sentence_break + 2

        chunk_text_str = text[start:end].strip()

        if chunk_text_str:
            chunks.append(Chunk(
                text=chunk_text_str,
                source_file=source_file,
                chunk_index=chunk_index,
                start_char=start,
                end_char=end
            ))
            chunk_index += 1

        # Move forward, accounting for overlap
        start = end - chunk_overlap
        if chunks and start <= chunks[-1].start_char:
            start = end  # Safety: ensure progress

    return chunks


# ============================================
# EXPERIMENT: Try different chunk sizes!
# ============================================
CHUNK_SIZE = 512      # Try: 256, 512, 1024
CHUNK_OVERLAP = 128   # Try: 64, 128, 256
# For Ex 7/8 use rebuild_pipeline() — see cell after FAISS index.

# Chunk all documents
all_chunks = []
for filename, content in documents:
    doc_chunks = chunk_text(content, filename, CHUNK_SIZE, CHUNK_OVERLAP)
    all_chunks.extend(doc_chunks)
    print(f"{filename}: {len(doc_chunks)} chunks")

print(f"\nTotal: {len(all_chunks)} chunks")


# Inspect some chunks
if all_chunks:
    print("Sample chunks:")
    indices_to_show = [0, len(all_chunks)//2, -1] if len(all_chunks) > 2 else range(len(all_chunks))
    for i in indices_to_show:
        chunk = all_chunks[i]
        print(f"\n{'='*60}")
        print(f"Chunk {chunk.chunk_index} from {chunk.source_file}")
        print(f"{'='*60}")
        print(chunk.text[:300] + "..." if len(chunk.text) > 300 else chunk.text)


from sentence_transformers import SentenceTransformer
import numpy as np

# Load embedding model
# Options:
# - "sentence-transformers/all-MiniLM-L6-v2": Fast, small (80MB), good quality
# - "BAAI/bge-small-en-v1.5": Better for retrieval, similar size

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

print(f"Loading embedding model: {EMBEDDING_MODEL}")
print(f"Device: {DEVICE}")

# Must explicitly pass device for MPS support!
embed_model = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
EMBEDDING_DIM = embed_model.get_sentence_embedding_dimension()
print(f"Embedding dimension: {EMBEDDING_DIM}")


# DEMO: See how embeddings capture semantic similarity
test_sentences = [
    "The engine needs regular oil changes.",
    "Motor oil should be replaced periodically.",
    "The Senate convened at noon.",
    "Congress began its session at midday."
]

test_embeddings = embed_model.encode(test_sentences)

# Compute cosine similarity matrix
from numpy.linalg import norm

def cosine_sim(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

print("Cosine similarity matrix:")
print("\n" + " " * 40 + "  [0]    [1]    [2]    [3]")
for i, s1 in enumerate(test_sentences):
    sims = [cosine_sim(test_embeddings[i], test_embeddings[j]) for j in range(4)]
    print(f"[{i}] {s1[:35]:35} {sims[0]:.3f}  {sims[1]:.3f}  {sims[2]:.3f}  {sims[3]:.3f}")

print("\n→ Notice: [0]-[1] are similar (both about oil), [2]-[3] are similar (both about Congress)")


# Embed all chunks - this may take a few minutes for large corpora
if all_chunks:
    print(f"Embedding {len(all_chunks)} chunks on {DEVICE}...")
    chunk_texts = [c.text for c in all_chunks]
    chunk_embeddings = embed_model.encode(chunk_texts, show_progress_bar=True)
    chunk_embeddings = chunk_embeddings.astype('float32')  # FAISS wants float32
    print(f"Embeddings shape: {chunk_embeddings.shape}")
else:
    print("No chunks to embed - please load documents first.")


import faiss

# Create FAISS index
# IndexFlatIP = Inner Product (for cosine similarity on normalized vectors)
index = faiss.IndexFlatIP(EMBEDDING_DIM)

if all_chunks:
    # Normalize vectors so inner product = cosine similarity
    faiss.normalize_L2(chunk_embeddings)

    # Add vectors to index
    index.add(chunk_embeddings)
    print(f"Index built with {index.ntotal} vectors")
else:
    print("No embeddings to index - please load and embed documents first.")


# Helper for Exercises 7 & 8: rebuild chunks + index with different chunk_size / chunk_overlap.
def rebuild_pipeline(chunk_size: int = 512, chunk_overlap: int = 128):
    """Re-chunk documents, re-embed, and rebuild FAISS index. Updates global all_chunks and index."""
    global all_chunks, index
    all_chunks = []
    for filename, content in documents:
        all_chunks.extend(chunk_text(content, filename, chunk_size=chunk_size, chunk_overlap=chunk_overlap))
    chunk_embeddings = embed_model.encode([c.text for c in all_chunks], show_progress_bar=True).astype("float32")
    faiss.normalize_L2(chunk_embeddings)
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(chunk_embeddings)
    print(f"Rebuilt: {len(all_chunks)} chunks, chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")


def retrieve(query: str, top_k: int = 5):
    """
    Retrieve the top-k most relevant chunks for a query.

    Returns: List of (chunk, similarity_score) tuples
    """
    # Embed the query
    query_embedding = embed_model.encode([query]).astype('float32')
    faiss.normalize_L2(query_embedding)

    # Search
    scores, indices = index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1:
            results.append((all_chunks[idx], float(score)))

    return results


# Test retrieval
# ============================================
# TRY DIFFERENT QUERIES FOR YOUR CORPUS!
# ============================================
test_query = "What is the procedure for engine maintenance?"  # ← Modify this!

if index.ntotal > 0:
    results = retrieve(test_query, top_k=5)

    print(f"Query: {test_query}\n")
    print("Top 5 retrieved chunks:")
    for i, (chunk, score) in enumerate(results, 1):
        print(f"\n[{i}] Score: {score:.4f} | Source: {chunk.source_file}")
        print(f"    {chunk.text[:200]}...")
else:
    print("Index is empty - please load, chunk, and embed documents first.")


from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================
# CHOOSE YOUR MODEL
# ============================================
LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"  # Or try "Qwen/Qwen2.5-3B-Instruct"

print(f"Loading LLM: {LLM_MODEL}")
print(f"Device: {DEVICE}, Dtype: {DTYPE}")
print("This may take a few minutes on first run...\n")

tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)

# Load with appropriate settings for each device type
if DEVICE == 'cuda':
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        device_map="auto",
        torch_dtype=DTYPE,
        trust_remote_code=True
    )
    print("Model loaded on CUDA")

elif DEVICE == 'mps':
    # For MPS, load to CPU first, then move to MPS
    # (device_map="auto" doesn't work well with MPS)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        torch_dtype=DTYPE,
        trust_remote_code=True
    )
    model = model.to(DEVICE)
    print("Model loaded on MPS (Apple Silicon)")

else:
    # CPU
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        torch_dtype=DTYPE,
        trust_remote_code=True
    )
    print("Model loaded on CPU (this will be slow)")


def generate_response(prompt: str, max_new_tokens: int = 512, temperature: float = 0.3) -> str:
    """
    Generate a response from the LLM.

    Lower temperature = more focused/deterministic
    Higher temperature = more creative/random
    """
    inputs = tokenizer(prompt, return_tensors="pt")

    # Move inputs to the correct device
    if DEVICE == 'cuda':
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    else:
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True if temperature > 0 else False,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode only the new tokens
    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )

    return response.strip()


# The RAG prompt template
PROMPT_TEMPLATE = """You are a helpful assistant that answers questions based on the provided context.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Answer the question based ONLY on the information in the context above
- If the context doesn't contain enough information to answer, say so
- Quote relevant parts of the context to support your answer
- Be concise and direct

ANSWER:"""


def direct_query(question: str, max_new_tokens: int = 512) -> str:
    """Ask the LLM directly with no retrieved context (for RAG vs no-RAG comparison)."""
    prompt = f"""Answer this question:
{question}

Answer:"""
    return generate_response(prompt, max_new_tokens=max_new_tokens)

def rag_query(question: str, top_k: int = 5, show_context: bool = False, prompt_template: str = None) -> str:
    """The complete RAG pipeline. prompt_template: custom template for Exercise 10."""
    # Step 1: Retrieve
    results = retrieve(question, top_k)

    # Format context
    context_parts = []
    for chunk, score in results:
        context_parts.append(f"[Source: {chunk.source_file}, Relevance: {score:.3f}]\n{chunk.text}")
    context = "\n\n---\n\n".join(context_parts)

    if show_context:
        print("=" * 60)
        print("RETRIEVED CONTEXT:")
        print("=" * 60)
        print(context)
        print("=" * 60 + "\n")

    # Step 2: Build prompt (use custom template if provided)
    template = prompt_template if prompt_template is not None else PROMPT_TEMPLATE
    prompt = template.format(context=context, question=question)

    # Step 3: Generate
    answer = generate_response(prompt)

    return answer


# ============================================
# TEST YOUR RAG PIPELINE!
# ============================================

question = "What maintenance is required for the engine?"  # ← Modify for your corpus!

if index.ntotal > 0:
    print(f"Question: {question}\n")
    print("Generating answer...\n")

    answer = rag_query(question, top_k=5, show_context=True)

    print("ANSWER:")
    print(answer)
else:
    print("Pipeline not ready - please complete all previous stages first.")


    