# CAD Material Tracker

AI-powered web application for extracting material tags from CAD drawings using Roboflow detection and OCR.

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.122-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Features

- **AI-Powered Detection**: Uses Roboflow to automatically detect CAD drawings on PDF pages
- **Real-time Processing**: Live WebSocket updates showing progress as pages are processed
- **OCR Extraction**: Extracts material tags (e.g., `BR-1`, `FCL-2`, `MC-1`) from detected drawings
- **Beautiful UI**: Modern dark-themed interface with live previews and statistics
- **Export Results**: Download extracted material tags as CSV
- **Fast & Efficient**: Processes only detected drawing regions, not entire pages

## Quick Start

### Prerequisites

Before starting, ensure you have:
- **Python 3.12** specifically ([Download Python 3.12](https://www.python.org/downloads/release/python-3120/))
  - **Why Python 3.12?** The Roboflow `inference-sdk` package (required for AI detection) only works with Python 3.12. This is a hard requirement, not optional.
- Git (if cloning from repository)
- Tesseract OCR ([Download here](https://github.com/UB-Mannheim/tesseract/wiki))
- Roboflow API key ([Get one here](https://roboflow.com))

### Installation Steps

#### Step 1: Get the Project

**Option A: Clone from Git repository**
```bash
git clone <your-repo-url>
cd Archive
```

**Option B: Download and extract ZIP**
```bash
# Extract the project folder
cd Archive
```

#### Step 2: Create Virtual Environment with Python 3.12

**Important:** This project requires Python 3.12 specifically because the Roboflow `inference-sdk` package (used for AI-powered CAD drawing detection) only supports Python 3.12. Create the virtual environment using Python 3.12:

**Windows:**
```powershell
# Using Python Launcher (recommended)
py -3.12 -m venv venv312

# Or if python3.12 is in your PATH
python3.12 -m venv venv312

# Activate virtual environment
venv312\Scripts\Activate.ps1
```

**Mac/Linux:**
```bash
# Create virtual environment with Python 3.12
python3.12 -m venv venv312

# Activate virtual environment
source venv312/bin/activate
```

**Note:** If you have multiple Python versions installed, you can check which versions are available:
- Windows: `py -0` (lists all installed Python versions)
- Mac/Linux: `python3.12 --version` (verify Python 3.12 is installed)

You should see `(venv312)` in your terminal prompt when activated.

#### Step 3: Install Python Dependencies

```bash
pip install -r requirements.txt
```

This will install all required packages including FastAPI, pdfplumber, Pillow, and others.

#### Step 4: Install Tesseract OCR

**Windows:**
```bash
# Using winget (recommended)
winget install --id UB-Mannheim.TesseractOCR

# Or download installer from:
# https://github.com/UB-Mannheim/tesseract/wiki
```

**Mac:**
```bash
brew install tesseract
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
```

**Verify Tesseract installation:**
```bash
tesseract --version
```

#### Step 5: Configure Environment Variables

Create a `.env` file in the project root directory:

**Windows (PowerShell):**
```powershell
New-Item -Path .env -ItemType File
```

**Mac/Linux:**
```bash
touch .env
```

Edit the `.env` file and add your configuration:
```env
ROBOFLOW_API_KEY=your_roboflow_api_key_here
ROBOFLOW_MODEL_ID=cad-drawing-iy9tc/11
```

Replace `your_roboflow_api_key_here` with your actual Roboflow API key.

#### Step 6: Verify Installation

Check that everything is set up correctly:

```bash
# Verify Python packages
python -c "import fastapi, pdfplumber, PIL; print('All packages installed')"

# Verify Tesseract
tesseract --version

# Verify .env file exists
# Windows:
if (Test-Path .env) { echo ".env file found" }

# Mac/Linux:
test -f .env && echo ".env file found"
```

#### Step 7: Run the Application

**Make sure your virtual environment is activated first!** You should see `(venv312)` in your terminal prompt.

**Option A: Using uvicorn (recommended)**
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

**Option B: Using Python directly**
```bash
python app.py
```

**Quick command to activate and run (Windows PowerShell):**
```powershell
venv312\Scripts\Activate.ps1; uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

You should see output like:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

**Note:** The `--reload` flag enables auto-reload on code changes (useful for development).

#### Step 8: Open in Browser

Open your web browser and navigate to:
```
http://localhost:8000
```

You should see the CAD Material Tracker interface.

### First Run Checklist

- [ ] Python 3.12 installed and verified
- [ ] Virtual environment `venv312` created with Python 3.12
- [ ] Virtual environment activated (you see `(venv312)` in terminal)
- [ ] All Python packages installed (`pip install -r requirements.txt`)
- [ ] Tesseract OCR installed and verified
- [ ] `.env` file created with `ROBOFLOW_API_KEY` and `ROBOFLOW_MODEL_ID`
- [ ] Application starts without errors
- [ ] Web interface loads at http://localhost:8000

## Usage

1. **Upload PDF**: Drag and drop your CAD drawings PDF or click to browse
2. **Configure Pages** (optional): Set start and end page numbers, or leave empty to process all pages
3. **Start Processing**: Click "Start Processing" to begin extraction
4. **Monitor Progress**: 
   - Left panel shows full page preview with detected drawings (red boxes)
   - Right panel shows each detected drawing region with extracted tags
   - Live logs show real-time progress
5. **Download Results**: Once complete, download the CSV file with all extracted material tags

## Project Structure

```
Archive/
├── app.py                      # FastAPI backend application
├── templates/
│   └── index.html             # Web UI template
├── material_descriptions.json  # Material tag descriptions
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
├── uploads/                    # Uploaded PDF files (auto-created)
├── output_images/              # Processed images (auto-created)
├── results/                    # Generated CSV results (auto-created)
└── predicted_images_web/       # Annotated images (auto-created)
```

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `ROBOFLOW_API_KEY` | Your Roboflow API key | Yes |
| `ROBOFLOW_MODEL_ID` | Roboflow model ID | Yes |

### Material Descriptions

The `material_descriptions.json` file contains mappings of material tags to their descriptions. Format:

```json
{
  "BR-1": "SOUTH & NORTH GROUND FLOOR - FACE BRICK 1",
  "FCL-2": "RESIDENTIAL LEVELS - FIBER CEMENT LAPSIDING 2",
  ...
}
```

## Tech Stack

- **Backend**: FastAPI, WebSockets
- **AI/ML**: Roboflow (object detection), Tesseract OCR
- **Frontend**: HTML5, CSS3, JavaScript, Lucide Icons
- **Image Processing**: PIL/Pillow
- **PDF Processing**: pdfplumber

## API Endpoints

- `GET /` - Main web interface
- `POST /upload` - Upload PDF file
- `POST /process` - Start processing (via WebSocket)
- `WebSocket /ws` - Real-time updates
- `GET /download` - Download results CSV

## Troubleshooting

### Tesseract not found
- Ensure Tesseract is installed and in your PATH
- On Windows, the app auto-detects common installation paths
- Verify installation: `tesseract --version`

### Roboflow errors
- Check your API key in `.env` file
- Verify model ID is correct
- Check your Roboflow account has API access

### No drawings detected
- Ensure PDF contains CAD drawings
- Try different pages (some pages may not have drawings)
- Check Roboflow model confidence threshold (default: 0.8)


