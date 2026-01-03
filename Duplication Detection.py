import streamlit as st
import os
import hashlib
from PIL import Image
import imagehash
from collections import defaultdict
import pandas as pd
import tempfile
import shutil
from pathlib import Path
import base64
import io
import zipfile
import atexit
import traceback
from typing import Dict, List, Tuple, Set
import time

# Set page configuration
st.set_page_config(
    page_title="Duplicate Image Finder",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with your color scheme
st.markdown("""
<style>
    /* Main Colors */
    :root {
        --primary: #008571;
        --primary-dark: #1E5050;
        --white: #FFFFFF;
        --gray: #4D4D4D;
        --accent-yellow: #EDB500;
        --accent-green: #B8D124;
        --danger: #FF5722;
        --warning: #FF9800;
    }
    
    /* Main container */
    .main {
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Progress bars */
    .stProgress .st-bo {
        background-color: var(--primary);
    }
    
    /* Duplicate group styling */
    .duplicate-group {
        background-color: rgba(0, 133, 113, 0.05);
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 20px;
        border-left: 6px solid var(--primary);
        border: 2px solid rgba(0, 133, 113, 0.1);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    
    /* Unique group styling */
    .unique-group {
        background-color: rgba(184, 209, 36, 0.05);
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 20px;
        border-left: 6px solid var(--accent-green);
        border: 2px solid rgba(184, 209, 36, 0.1);
    }
    
    /* Image grid */
    .image-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
        gap: 15px;
        margin-top: 15px;
    }
    
    /* Image container */
    .image-container {
        position: relative;
        border: 3px solid #E0E0E0;
        border-radius: 10px;
        overflow: hidden;
        transition: all 0.3s ease;
        background: var(--white);
        box-shadow: 0 3px 5px rgba(0, 0, 0, 0.1);
    }
    
    .image-container:hover {
        transform: translateY(-5px);
        border-color: var(--primary);
        box-shadow: 0 8px 15px rgba(0, 133, 113, 0.2);
    }
    
    .image-container.selected {
        border-color: var(--accent-yellow);
        border-width: 4px;
        background: rgba(237, 181, 0, 0.05);
    }
    
    /* Match badge */
    .match-badge {
        position: absolute;
        top: 10px;
        left: 10px;
        background: linear-gradient(135deg, var(--primary), var(--primary-dark));
        color: var(--white);
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.75em;
        font-weight: bold;
        z-index: 10;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    /* File size label */
    .file-size {
        font-size: 0.75em;
        color: var(--gray);
        padding: 3px 8px;
        background: rgba(255, 255, 255, 0.9);
        border-radius: 4px;
        display: inline-block;
        margin-top: 5px;
        border: 1px solid rgba(77, 77, 77, 0.1);
    }
    
    /* Similarity meter */
    .match-meter {
        height: 12px;
        background: #E8E8E8;
        border-radius: 6px;
        margin: 8px 0;
        overflow: hidden;
    }
    
    .match-fill {
        height: 100%;
        background: linear-gradient(90deg, var(--accent-green), var(--primary));
        border-radius: 6px;
    }
    
    /* Similarity info box */
    .similarity-info {
        background: linear-gradient(135deg, rgba(0, 133, 113, 0.05), rgba(184, 209, 36, 0.05));
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(0, 133, 113, 0.2);
        margin: 15px 0;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary), var(--primary-dark));
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, var(--primary-dark), var(--primary));
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 133, 113, 0.3);
    }
    
    /* Danger button */
    .danger-button > button {
        background: linear-gradient(135deg, var(--danger), #D32F2F);
    }
    
    .danger-button > button:hover {
        background: linear-gradient(135deg, #D32F2F, #B71C1C);
        box-shadow: 0 5px 15px rgba(255, 87, 34, 0.3);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, var(--primary), var(--primary-dark));
        color: white;
        padding: 25px;
        border-radius: 15px;
        margin-bottom: 25px;
        text-align: center;
        box-shadow: 0 6px 12px rgba(0, 133, 113, 0.3);
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        border: 2px solid rgba(0, 133, 113, 0.1);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05);
        text-align: center;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, rgba(0, 133, 113, 0.05) 0%, rgba(255, 255, 255, 1) 100%);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(0, 133, 113, 0.1);
        border-radius: 8px 8px 0 0;
        border: 1px solid rgba(0, 133, 113, 0.2);
        padding: 10px 20px;
        font-weight: 600;
        color: var(--primary-dark);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--primary), var(--primary-dark));
        color: white !important;
    }
    
    /* Upload area styling */
    .upload-area {
        border: 3px dashed var(--primary);
        border-radius: 15px;
        padding: 40px;
        text-align: center;
        background: rgba(0, 133, 113, 0.05);
        margin: 20px 0;
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        background: rgba(0, 133, 113, 0.1);
        border-color: var(--primary-dark);
    }
    
    /* Custom badge colors based on similarity */
    .badge-exact { background: linear-gradient(135deg, var(--accent-green), #5CB85C); }
    .badge-high { background: linear-gradient(135deg, var(--primary), var(--primary-dark)); }
    .badge-medium { background: linear-gradient(135deg, var(--accent-yellow), #FFA726); }
    .badge-low { background: linear-gradient(135deg, #FF9800, #FF5722); }
    
    /* Warning and info boxes */
    .warning-box {
        background: rgba(255, 152, 0, 0.1);
        border-left: 5px solid var(--warning);
        padding: 15px;
        border-radius: 8px;
        margin: 15px 0;
    }
    
    .info-box {
        background: rgba(0, 133, 113, 0.1);
        border-left: 5px solid var(--primary);
        padding: 15px;
        border-radius: 8px;
        margin: 15px 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        margin-top: 40px;
        padding: 20px;
        color: var(--gray);
        font-size: 0.9em;
        border-top: 1px solid rgba(0, 133, 113, 0.2);
    }
    
    /* Step indicator */
    .step-indicator {
        display: flex;
        justify-content: space-between;
        margin: 30px 0;
        position: relative;
    }
    
    .step-indicator::before {
        content: '';
        position: absolute;
        top: 15px;
        left: 10%;
        right: 10%;
        height: 3px;
        background: rgba(0, 133, 113, 0.2);
        z-index: 1;
    }
    
    .step {
        background: white;
        border: 3px solid rgba(0, 133, 113, 0.3);
        border-radius: 50%;
        width: 35px;
        height: 35px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        color: var(--primary);
        position: relative;
        z-index: 2;
    }
    
    .step.active {
        background: var(--primary);
        color: white;
        border-color: var(--primary-dark);
    }
    
    .step-label {
        position: absolute;
        top: 40px;
        left: 50%;
        transform: translateX(-50%);
        white-space: nowrap;
        font-size: 0.8em;
        color: var(--gray);
    }
    
    /* Feature cards */
    .feature-card {
        background: white;
        padding: 25px;
        border-radius: 15px;
        border: 2px solid rgba(0, 133, 113, 0.1);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05);
        text-align: center;
        transition: transform 0.3s ease;
        height: 100%;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        border-color: var(--primary);
    }
    
    /* Welcome card */
    .welcome-card {
        background: linear-gradient(135deg, rgba(0, 133, 113, 0.1), rgba(184, 209, 36, 0.1));
        padding: 30px;
        border-radius: 20px;
        border: 3px solid var(--primary);
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# Configuration constants
MAX_FILES = 1000
MAX_ZIP_SIZE_MB = 500  # Maximum zip file size
MAX_TOTAL_IMAGES = 2000  # Maximum total images to process
SUPPORTED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp', '.jfif', '.heic', '.avif')

class DuplicateImageFinder:
    def __init__(self, folder_path, recursive=True):
        self.folder_path = folder_path
        self.recursive = recursive
        self.image_extensions = SUPPORTED_EXTENSIONS
    
    def get_image_files(self):
        """Get all image files in the folder, optionally recursive"""
        image_files = []
        
        if self.recursive:
            # Recursive search through all subfolders
            for root, dirs, files in os.walk(self.folder_path):
                for filename in files:
                    if filename.lower().endswith(self.image_extensions):
                        # Store relative path from main folder
                        rel_path = os.path.relpath(os.path.join(root, filename), self.folder_path)
                        image_files.append(rel_path)
        else:
            # Non-recursive
            for filename in os.listdir(self.folder_path):
                if filename.lower().endswith(self.image_extensions):
                    filepath = os.path.join(self.folder_path, filename)
                    if os.path.isfile(filepath):
                        image_files.append(filename)
        
        return sorted(image_files)
    
    def get_full_path(self, filename):
        """Get full absolute path for a file"""
        return os.path.join(self.folder_path, filename)
    
    def calculate_hash_similarity(self, hash1: imagehash.ImageHash, hash2: imagehash.ImageHash, method: str = 'phash') -> float:
        """
        Calculate similarity percentage between two image hashes
        """
        max_differences = {
            'phash': 64,
            'average_hash': 64,
            'dhash': 64,
            'whash': 64
        }
        
        max_diff = max_differences.get(method, 64)
        diff = hash1 - hash2
        
        similarity = 100 * (1 - (diff / max_diff))
        return max(0, min(100, similarity))
    
    def find_duplicates_with_similarity(self, method='phash', threshold=5, similarity_threshold=80.0):
        """
        Find duplicate/similar images with similarity percentages
        """
        image_files = self.get_image_files()
        
        if len(image_files) == 0:
            return {}, {}, {}, {}
        
        if method == 'md5':
            return self._find_exact_duplicates_with_similarity(image_files)
        else:
            return self._find_similar_duplicates_with_similarity(
                image_files, method, threshold, similarity_threshold
            )
    
    def _find_exact_duplicates_with_similarity(self, image_files):
        """Find exact duplicates using MD5 hash"""
        hash_dict = {}
        duplicates = defaultdict(list)
        hash_values = {}
        similarity_scores = defaultdict(dict)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, filename in enumerate(image_files):
            status_text.text(f"Processing: {filename} ({idx+1}/{len(image_files)})")
            progress_bar.progress((idx + 1) / len(image_files))
            
            filepath = self.get_full_path(filename)
            
            try:
                with open(filepath, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                    hash_values[filename] = file_hash
                    
                    if file_hash in hash_dict:
                        original = hash_dict[file_hash]
                        duplicates[original].append(filename)
                        similarity_scores[original][filename] = 100.0
                    else:
                        hash_dict[file_hash] = filename
            except Exception as e:
                st.warning(f"Error processing {filename}: {e}")
        
        progress_bar.empty()
        status_text.empty()
        
        return dict(duplicates), hash_values, dict(similarity_scores), {}
    
    def _find_similar_duplicates_with_similarity(self, image_files, method='phash', threshold=5, similarity_threshold=80.0):
        """Find similar images using perceptual hashing with similarity scores"""
        hashes = {}
        file_hashes = {}
        hash_values = {}
        duplicates = defaultdict(list)
        similarity_scores = defaultdict(dict)
        
        hash_methods = {
            'phash': imagehash.phash,
            'average_hash': imagehash.average_hash,
            'dhash': imagehash.dhash,
            'whash': imagehash.whash
        }
        
        hash_func = hash_methods.get(method, imagehash.phash)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # First pass: calculate all hashes
        st.info("📊 Calculating image hashes...")
        for idx, filename in enumerate(image_files):
            status_text.text(f"Calculating: {filename[:50]}... ({idx+1}/{len(image_files)})")
            progress_bar.progress((idx + 1) / len(image_files) * 0.5)
            
            filepath = self.get_full_path(filename)
            
            try:
                with Image.open(filepath) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    current_hash = hash_func(img)
                    file_hashes[filename] = current_hash
                    hash_values[filename] = str(current_hash)
            except Exception as e:
                st.warning(f"⚠️ Error processing {filename}: {e}")
        
        # Second pass: find duplicates
        st.info("🔍 Finding similar images...")
        
        for idx, (filename, current_hash) in enumerate(file_hashes.items()):
            status_text.text(f"Comparing: {filename[:50]}... ({idx+1}/{len(file_hashes)})")
            progress_bar.progress(0.5 + (idx + 1) / len(file_hashes) * 0.5)
            
            best_match = None
            best_similarity = 0
            
            for existing_hash, existing_file in hashes.items():
                similarity = self.calculate_hash_similarity(current_hash, existing_hash, method)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = existing_file
            
            if best_match and best_similarity >= similarity_threshold:
                duplicates[best_match].append(filename)
                similarity_scores[best_match][filename] = best_similarity
            else:
                hashes[current_hash] = filename
        
        progress_bar.empty()
        status_text.empty()
        
        # Find best match for each file
        best_matches = {}
        for filename in image_files:
            if filename in file_hashes:
                best_match_file = None
                best_match_score = 0
                
                for other_file, other_hash in file_hashes.items():
                    if filename != other_file:
                        similarity = self.calculate_hash_similarity(
                            file_hashes[filename], 
                            other_hash, 
                            method
                        )
                        
                        if similarity > best_match_score:
                            best_match_score = similarity
                            best_match_file = other_file
                
                if best_match_file:
                    best_matches[filename] = {
                        'best_match': best_match_file,
                        'similarity': best_match_score
                    }
        
        return dict(duplicates), hash_values, dict(similarity_scores), best_matches

def extract_zip_file(zip_file, extract_to):
    """Extract zip file to temporary directory"""
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        # Check total extracted size
        total_size = sum(zinfo.file_size for zinfo in zip_ref.infolist())
        if total_size > MAX_ZIP_SIZE_MB * 1024 * 1024:
            raise ValueError(f"Zip file too large. Maximum size: {MAX_ZIP_SIZE_MB}MB")
        
        # Extract files
        zip_ref.extractall(extract_to)
        
        # Count extracted images
        image_count = 0
        for root, dirs, files in os.walk(extract_to):
            for file in files:
                if file.lower().endswith(SUPPORTED_EXTENSIONS):
                    image_count += 1
        
        return image_count

def validate_zip_file(zip_file):
    """Validate the uploaded zip file"""
    # Check file size
    file_size = len(zip_file.getvalue())
    if file_size > MAX_ZIP_SIZE_MB * 1024 * 1024:
        return False, f"Zip file too large. Maximum size: {MAX_ZIP_SIZE_MB}MB"
    
    # Try to read zip file
    try:
        with zipfile.ZipFile(io.BytesIO(zip_file.getvalue()), 'r') as zip_ref:
            # Check number of files
            file_count = len(zip_ref.namelist())
            if file_count > MAX_FILES:
                return False, f"Too many files in zip. Maximum: {MAX_FILES}"
            
            # Check for any image files
            has_images = any(
                name.lower().endswith(SUPPORTED_EXTENSIONS) 
                for name in zip_ref.namelist()
            )
            
            if not has_images:
                return False, "No image files found in zip archive"
            
            return True, f"Valid zip file with {file_count} files"
    except zipfile.BadZipFile:
        return False, "Invalid zip file format"
    except Exception as e:
        return False, f"Error reading zip file: {str(e)}"

def get_image_base64(image_path, max_size=(200, 200)):
    """Convert image to base64 for display"""
    try:
        img = Image.open(image_path)
        img.thumbnail(max_size)
        
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return img_str
    except Exception:
        return None

def get_file_size(filepath):
    """Get file size in human readable format"""
    try:
        size = os.path.getsize(filepath)
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"
    except:
        return "N/A"

def create_similarity_meter(similarity):
    """Create HTML for similarity meter with new colors"""
    width = min(100, max(0, similarity))
    
    # Choose color based on similarity
    if similarity >= 90:
        color = "#B8D124"  # Exact/High
        badge_class = "badge-exact"
    elif similarity >= 70:
        color = "#008571"  # Medium
        badge_class = "badge-high"
    elif similarity >= 50:
        color = "#EDB500"  # Low
        badge_class = "badge-medium"
    else:
        color = "#FF5722"  # Very Low
        badge_class = "badge-low"
    
    return f"""
    <div style="display: flex; align-items: center; gap: 15px; margin: 8px 0;">
        <div style="width: 120px; height: 12px; background: #E8E8E8; border-radius: 6px; overflow: hidden;">
            <div style="width: {width}%; height: 100%; background: {color}; border-radius: 6px;"></div>
        </div>
        <span style="font-weight: bold; color: {color}; min-width: 60px;">{similarity:.1f}%</span>
        <span style="font-size: 0.85em; color: #4D4D4D;">
            {'Exact' if similarity >= 95 else 
              'Very High' if similarity >= 90 else 
              'High' if similarity >= 80 else 
              'Medium' if similarity >= 70 else 
              'Low' if similarity >= 50 else 'Very Low'}
        </span>
    </div>
    """, badge_class

def get_badge_color(similarity):
    """Get badge color based on similarity"""
    if similarity >= 90:
        return "#B8D124"
    elif similarity >= 70:
        return "#008571"
    elif similarity >= 50:
        return "#EDB500"
    else:
        return "#FF5722"

def cleanup_temp_directory(temp_dir):
    """Clean up temporary directory"""
    if temp_dir and os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass

def register_session_cleanup():
    """Register cleanup for session state"""
    if 'temp_dir' in st.session_state:
        atexit.register(cleanup_temp_directory, st.session_state.temp_dir)

def create_step_indicator(current_step):
    """Create visual step indicator"""
    steps = [
        {"number": 1, "label": "Upload Zip", "description": "Upload your images zip file"},
        {"number": 2, "label": "Configure", "description": "Set detection settings"},
        {"number": 3, "label": "Scan", "description": "Find duplicate images"},
        {"number": 4, "label": "Manage", "description": "Review & export results"}
    ]
    
    html = '<div class="step-indicator">'
    for step in steps:
        active_class = "active" if step["number"] == current_step else ""
        html += f'''
        <div style="position: relative; text-align: center; flex: 1;">
            <div class="step {active_class}">{step["number"]}</div>
            <div class="step-label"><strong>{step["label"]}</strong></div>
            <div style="margin-top: 60px; font-size: 0.85em; color: var(--gray);">{step["description"]}</div>
        </div>
        '''
    html += '</div>'
    return html

def main():
    # Initialize session state
    if 'duplicates' not in st.session_state:
        st.session_state.duplicates = {}
    if 'hash_values' not in st.session_state:
        st.session_state.hash_values = {}
    if 'similarity_scores' not in st.session_state:
        st.session_state.similarity_scores = {}
    if 'best_matches' not in st.session_state:
        st.session_state.best_matches = {}
    if 'temp_dir' not in st.session_state:
        st.session_state.temp_dir = None
    if 'files_to_delete' not in st.session_state:
        st.session_state.files_to_delete = set()
    if 'recursive_search' not in st.session_state:
        st.session_state.recursive_search = True
    if 'scan_complete' not in st.session_state:
        st.session_state.scan_complete = False
    if 'zip_filename' not in st.session_state:
        st.session_state.zip_filename = None
    if 'image_count' not in st.session_state:
        st.session_state.image_count = 0
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 1
    
    # Custom header - Single unified header
    st.markdown("""
    <div class="main-header">
        <h1 style="margin: 0; font-size: 2.5em; display: flex; align-items: center; justify-content: center; gap: 15px;">
            <span>🔍</span>
            <span>Smart Duplicate Image Finder</span>
        </h1>
        <p style="margin: 10px 0 0 0; font-size: 1.2em; opacity: 0.9;">
            Upload a zip file of images to find duplicates with similarity analysis
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # Step indicator in sidebar
        current_step = st.session_state.current_step
        if current_step == 1:
            st.markdown("#### 📤 Step 1: Upload Zip File")
        elif current_step == 2:
            st.markdown("#### ⚙️ Step 2: Configure Settings")
        elif current_step == 3:
            st.markdown("#### 🔍 Step 3: Scan & Analyze")
        else:
            st.markdown("#### 📊 Step 4: Results & Export")
        
        # Zip file upload - Only show in step 1
        if current_step == 1 or not st.session_state.temp_dir:
            st.markdown('<div class="upload-area">', unsafe_allow_html=True)
            zip_file = st.file_uploader(
                "📦 Choose or drag & drop a zip file",
                type=['zip'],
                help="Upload a zip file containing your images",
                label_visibility="collapsed"
            )
            st.markdown('<p style="font-size: 0.9em; color: var(--gray); margin-top: 10px;">'
                       '💡 Tip: Zip your image folder before uploading</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            if zip_file:
                is_valid, message = validate_zip_file(zip_file)
                
                if is_valid:
                    st.success(f"✅ {message}")
                    
                    if st.button("📂 Extract Images", use_container_width=True, type="primary"):
                        with st.spinner("Extracting zip file..."):
                            # Create temp directory
                            temp_dir = tempfile.mkdtemp()
                            st.session_state.temp_dir = temp_dir
                            
                            # Save zip file
                            zip_path = os.path.join(temp_dir, "uploaded.zip")
                            with open(zip_path, "wb") as f:
                                f.write(zip_file.getvalue())
                            
                            # Extract zip
                            try:
                                image_count = extract_zip_file(zip_path, temp_dir)
                                st.session_state.image_count = image_count
                                st.session_state.zip_filename = zip_file.name
                                
                                if image_count == 0:
                                    st.error("❌ No images found in zip file!")
                                    cleanup_temp_directory(temp_dir)
                                    st.session_state.temp_dir = None
                                elif image_count > MAX_TOTAL_IMAGES:
                                    st.error(f"❌ Too many images ({image_count}). Maximum: {MAX_TOTAL_IMAGES}")
                                    cleanup_temp_directory(temp_dir)
                                    st.session_state.temp_dir = None
                                else:
                                    st.success(f"✅ Extracted {image_count} images")
                                    st.session_state.scan_complete = False
                                    st.session_state.current_step = 2
                                    st.rerun()
                            except Exception as e:
                                st.error(f"❌ Error extracting zip: {str(e)}")
                                cleanup_temp_directory(temp_dir)
                                st.session_state.temp_dir = None
                else:
                    st.error(f"❌ {message}")
        
        # Detection settings - Show in step 2 or later
        if st.session_state.temp_dir and (current_step >= 2 or st.session_state.image_count > 0):
            st.markdown("#### 🔍 Detection Settings")
            method = st.selectbox(
                "Detection Method:",
                options=['phash', 'md5', 'average_hash', 'dhash'],
                index=0,
                help="• phash: Best for similar images (recommended)\n• md5: Exact duplicates only\n• average_hash: Faster but less accurate\n• dhash: Good for resized images"
            )
            
            if method != 'md5':
                col1, col2 = st.columns(2)
                with col1:
                    threshold = st.slider(
                        "Hash Threshold:",
                        min_value=0,
                        max_value=64,
                        value=5,
                        help="Lower = more strict matching"
                    )
                with col2:
                    similarity_threshold = st.slider(
                        "Similarity (%):",
                        min_value=50,
                        max_value=100,
                        value=80,
                        help="Minimum similarity to mark as duplicate"
                    )
            else:
                threshold = 0
                similarity_threshold = 100
            
            # Action buttons
            st.markdown("#### 🚀 Actions")
            
            if not st.session_state.scan_complete:
                if st.button("🔍 Start Scan", use_container_width=True, type="primary"):
                    st.session_state.current_step = 3
                    with st.spinner("🚀 Scanning for duplicates..."):
                        finder = DuplicateImageFinder(
                            st.session_state.temp_dir,
                            recursive=True
                        )
                        
                        if method == 'md5':
                            duplicates, hash_values, similarity_scores, best_matches = finder.find_duplicates_with_similarity(
                                method=method
                            )
                        else:
                            duplicates, hash_values, similarity_scores, best_matches = finder.find_duplicates_with_similarity(
                                method=method,
                                threshold=threshold,
                                similarity_threshold=similarity_threshold
                            )
                        
                        st.session_state.duplicates = duplicates
                        st.session_state.hash_values = hash_values
                        st.session_state.similarity_scores = similarity_scores
                        st.session_state.best_matches = best_matches
                        st.session_state.scan_complete = True
                        st.session_state.current_step = 4
                    
                    st.success("✅ Scan completed!")
                    st.rerun()
            
            # Clear button
            if st.session_state.scan_complete or st.session_state.image_count > 0:
                if st.button("🗑️ Clear All", use_container_width=True):
                    # Clean up
                    if st.session_state.temp_dir:
                        cleanup_temp_directory(st.session_state.temp_dir)
                    
                    # Reset session state
                    for key in ['duplicates', 'hash_values', 'similarity_scores', 'best_matches',
                               'files_to_delete', 'temp_dir', 'zip_filename', 'image_count', 'current_step']:
                        if key in st.session_state:
                            st.session_state[key] = None if key == 'temp_dir' else {} if key == 'files_to_delete' else set() if key == 'files_to_delete' else 1 if key == 'current_step' else 0 if key == 'image_count' else {}
                    
                    st.session_state.scan_complete = False
                    st.rerun()
        
        # Statistics
        if st.session_state.image_count > 0:
            st.markdown("#### 📊 Statistics")
            
            if st.session_state.duplicates:
                total_duplicates = sum(len(dup_list) for dup_list in st.session_state.duplicates.values())
                avg_similarity = 0
                count = 0
                
                for original, scores in st.session_state.similarity_scores.items():
                    for dup, score in scores.items():
                        avg_similarity += score
                        count += 1
                
                if count > 0:
                    avg_similarity = avg_similarity / count
                
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size: 0.9em; color: #4D4D4D;">Total Images</div>
                    <div style="font-size: 1.8em; font-weight: bold; color: #008571;">{st.session_state.image_count}</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size: 0.9em; color: #4D4D4D;">Total Duplicates</div>
                    <div style="font-size: 1.8em; font-weight: bold; color: #1E5050;">{total_duplicates}</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size: 0.9em; color: #4D4D4D;">Avg Similarity</div>
                    <div style="font-size: 1.8em; font-weight: bold; color: #B8D124;">{avg_similarity:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size: 0.9em; color: #4D4D4D;">Images Ready</div>
                    <div style="font-size: 1.8em; font-weight: bold; color: #008571;">{st.session_state.image_count}</div>
                    <div style="font-size: 0.8em; color: var(--gray); margin-top: 5px;">
                        Click "Start Scan" to begin analysis
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Info box
        st.markdown("""
        <div class="info-box">
            <strong>📋 Quick Guide:</strong>
            <ol style="margin: 10px 0; padding-left: 20px; font-size: 0.9em;">
                <li>Upload zip with images</li>
                <li>Configure detection settings</li>
                <li>Start scan</li>
                <li>Review & export results</li>
            </ol>
            <div style="font-size: 0.85em; color: var(--gray); margin-top: 10px;">
                <strong>📁 Supported formats:</strong> PNG, JPG, JPEG, WEBP, GIF, BMP, TIFF
                <br><strong>📦 Max zip size:</strong> {MAX_ZIP_SIZE_MB}MB
                <br><strong>🖼️ Max images:</strong> {MAX_TOTAL_IMAGES}
            </div>
        </div>
        """.format(MAX_ZIP_SIZE_MB=MAX_ZIP_SIZE_MB, MAX_TOTAL_IMAGES=MAX_TOTAL_IMAGES), unsafe_allow_html=True)
    
    # Main content area
    # Show step indicator
    if st.session_state.current_step > 1:
        st.markdown(create_step_indicator(st.session_state.current_step), unsafe_allow_html=True)
    
    if st.session_state.temp_dir and os.path.exists(st.session_state.temp_dir):
        # Show current session info
        if st.session_state.current_step >= 2:
            st.markdown(f"""
            <div style="background: rgba(0, 133, 113, 0.1); padding: 20px; border-radius: 12px; margin: 20px 0; border-left: 5px solid #008571;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <strong style="font-size: 1.1em;">📁 Current Session:</strong>
                        <div style="margin-top: 8px;">
                            <span style="color: var(--primary);">📦 Archive:</span> <code>{st.session_state.zip_filename}</code>
                            <br><span style="color: var(--primary);">🖼️ Images:</span> {st.session_state.image_count} images found
                            <br><span style="color: var(--primary);">⚙️ Method:</span> {'MD5 (Exact)' if 'method' not in locals() or method == 'md5' else 'Perceptual Hash'}
                        </div>
                    </div>
                    <div>
                        <button onclick="window.location.reload();" style="background: rgba(255, 255, 255, 0.9); border: 2px solid var(--primary); color: var(--primary-dark); padding: 8px 16px; border-radius: 6px; cursor: pointer; font-weight: bold; transition: all 0.3s;">
                            🔄 New Session
                        </button>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Display results or ready state
        if st.session_state.duplicates:
            total_duplicates = sum(len(dup_list) for dup_list in st.session_state.duplicates.values())
            
            st.markdown(f"""
            <div style="text-align: center; margin: 25px 0;">
                <h2 style="color: #1E5050;">🎯 Analysis Complete!</h2>
                <p style="font-size: 1.2em;">
                    Found <span style="color: #008571; font-weight: bold;">{total_duplicates}</span> duplicate files in 
                    <span style="color: #1E5050; font-weight: bold;">{len(st.session_state.duplicates)}</span> groups
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Create tabs
            tab1, tab2, tab3, tab4 = st.tabs(["📸 Visual Groups", "📊 Similarity Analysis", "🏆 Best Matches", "📥 Export Results"])
            
            with tab1:
                # Display each duplicate group
                if len(st.session_state.duplicates) == 0:
                    st.markdown("""
                    <div style="text-align: center; padding: 50px; background: rgba(184, 209, 36, 0.1); border-radius: 15px; margin: 20px 0;">
                        <h3 style="color: #1E5050;">🎉 No Duplicates Found!</h3>
                        <p style="font-size: 1.1em; color: #4D4D4D;">
                            Your image collection appears to be clean with no duplicate images.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    for idx, (original, duplicates_list) in enumerate(st.session_state.duplicates.items()):
                        with st.container():
                            st.markdown(f'<div class="duplicate-group">', unsafe_allow_html=True)
                            
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.markdown(f"### 🏷️ Group {idx+1}")
                                st.markdown(f"**Original:** `{original}`")
                                st.markdown(f"*{len(duplicates_list)} duplicate(s) found*")
                            
                            with col2:
                                original_path = os.path.join(st.session_state.temp_dir, original)
                                original_size = get_file_size(original_path)
                                st.markdown(f"""
                                <div style="text-align: right;">
                                    <div style="font-size: 0.9em; color: #4D4D4D;">File Size</div>
                                    <div style="font-size: 1.2em; font-weight: bold; color: #008571;">{original_size}</div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Display similarity scores
                            if original in st.session_state.similarity_scores:
                                st.markdown("**Similarity Scores:**")
                                for dup in duplicates_list:
                                    similarity = st.session_state.similarity_scores[original].get(dup, 0)
                                    meter_html, _ = create_similarity_meter(similarity)
                                    st.markdown(meter_html, unsafe_allow_html=True)
                            
                            # Display images in a grid
                            all_files = [original] + duplicates_list
                            st.markdown('<div class="image-grid">', unsafe_allow_html=True)
                            
                            cols = st.columns(min(len(all_files), 5))
                            for i, filename in enumerate(all_files):
                                filepath = os.path.join(st.session_state.temp_dir, filename)
                                img_base64 = get_image_base64(filepath)
                                
                                with cols[i % len(cols)]:
                                    if img_base64:
                                        # Show similarity badge for duplicates
                                        badge_html = ""
                                        if i > 0:  # Not the original
                                            similarity = st.session_state.similarity_scores.get(original, {}).get(filename, 0)
                                            if similarity > 0:
                                                badge_color = get_badge_color(similarity)
                                                badge_html = f'<div class="match-badge" style="background: {badge_color};">{similarity:.0f}%</div>'
                                        
                                        st.markdown(f"""
                                        <div class="image-container">
                                            {badge_html if badge_html else ""}
                                            <img src="data:image/png;base64,{img_base64}" 
                                                 style="width:100%; height:auto; border-radius:8px;">
                                            <div style="padding:10px; font-size:0.8em;">
                                                <div style="color: {'#008571' if i==0 else '#FF5722'}; font-weight: bold; margin-bottom: 5px;">
                                                    {'🟢 Original' if i==0 else '🔴 Duplicate'}
                                                </div>
                                                <div style="word-break: break-all; font-size: 0.75em; margin-bottom: 5px;">{filename[-30:]}</div>
                                                <span class="file-size">{get_file_size(filepath)}</span>
                                            </div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    
                                    # Checkbox for selection
                                    if filename != original:
                                        checkbox_key = f"del_{hash(filename)}_{idx}"
                                        is_checked = st.checkbox(
                                            f"Select {filename[:20]}...", 
                                            key=checkbox_key,
                                            help=f"Select to mark {filename} for removal"
                                        )
                                        
                                        if is_checked:
                                            st.session_state.files_to_delete.add(filename)
                                        elif filename in st.session_state.files_to_delete:
                                            st.session_state.files_to_delete.remove(filename)
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)
            
            with tab2:
                # Similarity analysis
                st.markdown("### 📈 Similarity Analysis")
                
                data = []
                for original, duplicates_list in st.session_state.duplicates.items():
                    for dup in duplicates_list:
                        original_path = os.path.join(st.session_state.temp_dir, original)
                        dup_path = os.path.join(st.session_state.temp_dir, dup)
                        
                        similarity = st.session_state.similarity_scores.get(original, {}).get(dup, 0)
                        
                        data.append({
                            'Original File': original,
                            'Duplicate File': dup,
                            'Similarity (%)': similarity,
                            'Match Level': (
                                'Exact' if similarity >= 99 else
                                'Very High' if similarity >= 90 else
                                'High' if similarity >= 80 else
                                'Medium' if similarity >= 70 else
                                'Low' if similarity >= 60 else
                                'Very Low'
                            ),
                            'Original Size': get_file_size(original_path),
                            'Duplicate Size': get_file_size(dup_path),
                            'Hash': st.session_state.hash_values.get(dup, 'N/A')[:20] + '...'
                        })
                
                if data:
                    df = pd.DataFrame(data)
                    df = df.sort_values('Similarity (%)', ascending=False)
                    
                    # Display with styling
                    def color_similarity(val):
                        if isinstance(val, (int, float)):
                            if val >= 90:
                                return 'background-color: rgba(184, 209, 36, 0.2); color: #1E5050; font-weight: bold;'
                            elif val >= 80:
                                return 'background-color: rgba(0, 133, 113, 0.2); color: #1E5050;'
                            elif val >= 70:
                                return 'background-color: rgba(237, 181, 0, 0.2); color: #4D4D4D;'
                            else:
                                return 'background-color: rgba(255, 87, 34, 0.1); color: #4D4D4D;'
                        return ''
                    
                    styled_df = df.style.map(color_similarity, subset=['Similarity (%)'])
                    st.dataframe(styled_df, use_container_width=True, hide_index=True,
                                column_config={
                                    "Similarity (%)": st.column_config.NumberColumn(
                                        format="%.1f%%"
                                    )
                                })
                    
                    # Export
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Similarity Report (CSV)",
                        data=csv,
                        file_name="similarity_report.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            with tab3:
                # Best matches analysis
                st.markdown("### 🏆 Best Match Analysis")
                
                if st.session_state.best_matches:
                    best_match_data = []
                    for filename, match_info in st.session_state.best_matches.items():
                        filepath = os.path.join(st.session_state.temp_dir, filename)
                        
                        best_match_data.append({
                            'Image': filename,
                            'Best Match': match_info['best_match'],
                            'Similarity (%)': match_info['similarity'],
                            'Is Duplicate?': 'Yes' if filename in [dup for dups in st.session_state.duplicates.values() for dup in dups] else 'No',
                            'File Size': get_file_size(filepath)
                        })
                    
                    if best_match_data:
                        df_best = pd.DataFrame(best_match_data)
                        df_best = df_best.sort_values('Similarity (%)', ascending=False)
                        
                        st.dataframe(df_best, use_container_width=True, hide_index=True,
                                    column_config={
                                        "Similarity (%)": st.column_config.NumberColumn(
                                            format="%.1f%%"
                                        )
                                    })
                        
                        # Export
                        csv_best = df_best.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Best Matches (CSV)",
                            data=csv_best,
                            file_name="best_matches.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
            
            with tab4:
                # File management and export
                st.markdown("### 📥 Export Results")
                
                if st.session_state.files_to_delete:
                    st.warning(f"**🗑️ {len(st.session_state.files_to_delete)} files selected for removal**")
                    
                    # Show selected files
                    for filename in sorted(st.session_state.files_to_delete):
                        filepath = os.path.join(st.session_state.temp_dir, filename)
                        
                        # Find original and similarity
                        original_file = None
                        similarity_score = 0
                        for orig, dups in st.session_state.duplicates.items():
                            if filename in dups:
                                original_file = orig
                                similarity_score = st.session_state.similarity_scores.get(orig, {}).get(filename, 0)
                                break
                        
                        if original_file:
                            st.markdown(f"""
                            <div style="background: rgba(255, 87, 34, 0.1); padding: 10px; border-radius: 8px; margin: 5px 0; border-left: 4px solid #FF5722;">
                                <strong>{filename}</strong><br>
                                <small>Size: {get_file_size(filepath)} | Matches: {original_file} ({similarity_score:.1f}%)</small>
                            </div>
                            """, unsafe_allow_html=True)
                
                # Export options
                st.markdown("#### 🚀 Export Options")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Clean archive download
                    if st.button("📦 Download Clean Archive", use_container_width=True):
                        with st.spinner("Creating clean zip file..."):
                            # Create new zip without deleted files
                            clean_zip_path = os.path.join(st.session_state.temp_dir, "clean_images.zip")
                            with zipfile.ZipFile(clean_zip_path, 'w') as zipf:
                                for root, dirs, files in os.walk(st.session_state.temp_dir):
                                    for file in files:
                                        if file.lower().endswith(SUPPORTED_EXTENSIONS):
                                            file_rel_path = os.path.relpath(os.path.join(root, file), st.session_state.temp_dir)
                                            if file_rel_path not in st.session_state.files_to_delete:
                                                zipf.write(
                                                    os.path.join(root, file),
                                                    file_rel_path
                                                )
                            
                            # Offer download
                            with open(clean_zip_path, 'rb') as f:
                                st.download_button(
                                    label="⬇️ Download Clean Archive",
                                    data=f,
                                    file_name="clean_images.zip",
                                    mime="application/zip",
                                    use_container_width=True
                                )
                
                with col2:
                    # Clear selection
                    if st.session_state.files_to_delete:
                        if st.button("🗑️ Clear Selection", use_container_width=True):
                            st.session_state.files_to_delete.clear()
                            st.rerun()
                
                # Full report export
                st.markdown("---")
                st.markdown("#### 📄 Full Report Export")
                
                report_col1, report_col2 = st.columns(2)
                
                with report_col1:
                    # Export all data as JSON
                    import json
                    report_data = {
                        "archive_name": st.session_state.zip_filename,
                        "image_count": st.session_state.image_count,
                        "duplicate_groups": st.session_state.duplicates,
                        "similarity_scores": st.session_state.similarity_scores,
                        "best_matches": st.session_state.best_matches,
                        "selected_for_removal": list(st.session_state.files_to_delete)
                    }
                    
                    json_str = json.dumps(report_data, indent=2, default=str)
                    st.download_button(
                        label="📊 Download Full Report (JSON)",
                        data=json_str,
                        file_name="full_report.json",
                        mime="application/json",
                        use_container_width=True
                    )
        
        elif st.session_state.scan_complete and not st.session_state.duplicates:
            st.markdown("""
            <div style="text-align: center; padding: 50px; background: rgba(184, 209, 36, 0.1); border-radius: 15px; margin: 20px 0;">
                <h2 style="color: #1E5050;">🎉 No Duplicates Found!</h2>
                <p style="font-size: 1.2em; color: #4D4D4D;">
                    Your image collection appears to be clean with no duplicate images.
                </p>
                <div style="margin-top: 20px;">
                    <p style="color: #4D4D4D;">You can still download your images or start a new session.</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Export option even if no duplicates
            if st.button("📦 Download Original Archive", use_container_width=True):
                with st.spinner("Preparing download..."):
                    clean_zip_path = os.path.join(st.session_state.temp_dir, "original_images.zip")
                    with zipfile.ZipFile(clean_zip_path, 'w') as zipf:
                        for root, dirs, files in os.walk(st.session_state.temp_dir):
                            for file in files:
                                if file.lower().endswith(SUPPORTED_EXTENSIONS):
                                    file_rel_path = os.path.relpath(os.path.join(root, file), st.session_state.temp_dir)
                                    zipf.write(
                                        os.path.join(root, file),
                                        file_rel_path
                                    )
                    
                    with open(clean_zip_path, 'rb') as f:
                        st.download_button(
                            label="⬇️ Download Archive",
                            data=f,
                            file_name="original_images.zip",
                            mime="application/zip",
                            use_container_width=True
                        )
        else:
            # Ready to scan
            st.markdown("""
            <div class="welcome-card">
                <h3 style="color: #1E5050; text-align: center;">🚀 Ready to Scan</h3>
                <p style="color: #4D4D4D; margin-bottom: 20px; text-align: center;">
                    You have <strong style="color: #008571;">{image_count}</strong> images ready for analysis.
                    <br>Click <strong>"Start Scan"</strong> in the sidebar to begin!
                </p>
                <div style="display: flex; justify-content: center; gap: 15px; margin-top: 20px;">
                    <div style="text-align: center;">
                        <div style="font-size: 2em;">📦</div>
                        <div style="font-weight: bold; color: #1E5050;">Archive</div>
                        <div style="font-size: 0.9em; color: #4D4D4D;">{zip_filename}</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 2em;">🖼️</div>
                        <div style="font-weight: bold; color: #1E5050;">Images</div>
                        <div style="font-size: 0.9em; color: #4D4D4d;">{image_count} found</div>
                    </div>
                </div>
            </div>
            """.format(
                image_count=st.session_state.image_count,
                zip_filename=st.session_state.zip_filename[:30] + "..." if len(st.session_state.zip_filename) > 30 else st.session_state.zip_filename
            ), unsafe_allow_html=True)
    
    else:
        # Welcome screen - User-friendly introduction
        st.markdown("""
        <div style="text-align: center; padding: 40px; background: linear-gradient(135deg, #008571, #1E5050); 
                 color: white; border-radius: 20px; margin: 20px 0;">
            <h1 style="font-size: 2.8em; margin-bottom: 10px;">🔍 Smart Duplicate Image Finder</h1>
            <p style="font-size: 1.3em; opacity: 0.9;">Find and remove duplicate images in seconds</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature cards
        st.markdown("### 🌟 Why Use Our Tool?")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="feature-card">
                <div style="font-size: 3em; margin-bottom: 15px;">📦</div>
                <h3 style="color: #1E5050; margin-bottom: 10px;">Easy Zip Upload</h3>
                <p style="color: #4D4D4D; font-size: 0.95em;">
                    Just zip your image folder and upload. No need to select individual files.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="feature-card">
                <div style="font-size: 3em; margin-bottom: 15px;">🎯</div>
                <h3 style="color: #1E5050; margin-bottom: 10px;">Smart Detection</h3>
                <p style="color: #4D4D4D; font-size: 0.95em;">
                    Finds similar images, not just exact copies. Get similarity percentages for each match.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="feature-card">
                <div style="font-size: 3em; margin-bottom: 15px;">🛡️</div>
                <h3 style="color: #1E5050; margin-bottom: 10px;">Privacy First</h3>
                <p style="color: #4D4D4D; font-size: 0.95em;">
                    All files are processed temporarily and automatically deleted. Your privacy is protected.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # How it works section
        st.markdown("""
        <div style="background: white; padding: 30px; border-radius: 15px; border: 3px solid #008571; margin: 30px 0;">
            <h3 style="color: #1E5050; text-align: center;">📝 How It Works</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-top: 20px;">
                <div style="text-align: center; padding: 20px;">
                    <div style="background: rgba(0, 133, 113, 0.1); width: 60px; height: 60px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 15px; font-size: 1.5em;">
                        1️⃣
                    </div>
                    <h4 style="color: #1E5050;">Compress & Upload</h4>
                    <p>Zip your image folder and upload it using the sidebar</p>
                </div>
                <div style="text-align: center; padding: 20px;">
                    <div style="background: rgba(0, 133, 113, 0.1); width: 60px; height: 60px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 15px; font-size: 1.5em;">
                        2️⃣
                    </div>
                    <h4 style="color: #1E5050;">Configure Settings</h4>
                    <p>Choose detection method and similarity threshold</p>
                </div>
                <div style="text-align: center; padding: 20px;">
                    <div style="background: rgba(0, 133, 113, 0.1); width: 60px; height: 60px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 15px; font-size: 1.5em;">
                        3️⃣
                    </div>
                    <h4 style="color: #1E5050;">Scan & Analyze</h4>
                    <p>Let our algorithm find duplicate and similar images</p>
                </div>
                <div style="text-align: center; padding: 20px;">
                    <div style="background: rgba(0, 133, 113, 0.1); width: 60px; height: 60px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 15px; font-size: 1.5em;">
                        4️⃣
                    </div>
                    <h4 style="color: #1E5050;">Review & Export</h4>
                    <p>View results and download clean archive</p>
                </div>
            </div>
            
            <div style="text-align: center; margin-top: 30px;">
                <p style="font-size: 1.1em; color: #4D4D4D;">
                    <strong>Ready to start?</strong> Use the uploader in the sidebar to begin!
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>🛡️ <strong>Privacy Notice:</strong> All uploaded files are processed temporarily and automatically deleted after your session ends.</p>
        <p style="font-size: 0.8em; margin-top: 10px;">Smart Duplicate Image Finder v2.0 | Made with ❤️ for organizing your photos</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Register cleanup
    register_session_cleanup()

if __name__ == "__main__":
    try:
        # Check for required packages
        import PIL
        import imagehash
        import pandas
        import numpy
        
        main()
    except ImportError as e:
        st.error(f"❌ Missing required package: {e}")
        st.info("💡 Install required packages with:")
        st.code("pip install pillow imagehash streamlit pandas numpy", language="bash")
    except Exception as e:
        st.error("❌ An unexpected error occurred")
        
        # User-friendly error message
        st.markdown("""
        <div class="warning-box">
            <strong>🛠️ Troubleshooting Tips:</strong>
            <ul>
                <li><strong>Refresh the page</strong> and try again</li>
                <li>Ensure your zip file is not corrupted</li>
                <li>Try with a smaller zip file (under 100MB)</li>
                <li>Check that images are in supported formats</li>
                <li>Ensure zip file contains actual images</li>
            </ul>
            <p style="margin-top: 10px;">If the problem persists, please try uploading a different zip file.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Optional: Show technical details in expander
        with st.expander("Technical Details (for debugging)"):
            st.exception(e)