# CAD Material Tracker

AI-powered web application for extracting material tags from CAD drawings using Roboflow detection and OCR.

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.122-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Features

- **AI-Powered Detection**: Uses Roboflow to automatically detect CAD drawings on PDF pages
- **Real-time Processing**: Live WebSocket updates showing progress as pages are processed
- **OCR Extraction**: Extracts material tags (e.g., `BR-1`, `FCL-2`, `MC-1`) using **EasyOCR** (Deep Learning) for superior accuracy on noisy CAD drawings
- **Advanced Deduplication**: Intelligent algorithms to prevent duplicate tags within drawings and overlapping detection regions
- **Cost Optimization**: Smart processing only triggers expensive API calls when valid tags are found
- **Tag Management**: Add, remove, and manage detection tags through a modern web interface
- **CSV Import/Export**: Bulk upload tags via CSV or download current tags as template
- **Beautiful UI**: Modern dark-themed interface with live previews and statistics
- **Export Results**: Download extracted material tags as CSV
- **Fast & Efficient**: Processes only detected drawing regions, not entire pages

## Quick Start

### Prerequisites

Before starting, ensure you have:
- **Python 3.12** specifically ([Download Python 3.12](https://www.python.org/downloads/release/python-3120/))
  - **Why Python 3.12?** The Roboflow `inference-sdk` package (required for AI detection) only works with Python 3.12. This is a hard requirement, not optional.
- Git (if cloning from repository)
- EasyOCR (automatically installed via requirements.txt)
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

#### Step 4: Verify EasyOCR
EasyOCR installs automatically with `pip`. On the first run, it will download necessary model files (~100MB). No manual system installation is required unlike Tesseract.

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

### Managing Tags

Before processing PDFs, you can manage which tags the system detects:

1. **Access Tag Management**: Click the "Manage Tags" button in the header
2. **View Tags**: See all currently configured tags in a sortable table
3. **Add Single Tag**: 
   - Enter tag ID (e.g., `GL-10`, `BR-1`, `MT-05`)
   - Enter description (e.g., "Tempered Glass Window")
   - Click "Add Tag"
4. **Bulk Upload via CSV**:
   - Click "Download Template" to get the CSV format
   - Edit CSV with your tags: `tag,description`
   - Click "Upload CSV File" and select your file
   - System validates and imports all tags
5. **Remove Tags**: Click the "Delete" button next to any tag to remove it
6. **Download Tags**: Click "Download Template" to export all current tags as CSV

> **Note**: Tags are automatically reloaded before each PDF processing job, so changes take effect immediately.

### Processing PDFs

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
│   ├── index.html             # Main web UI
│   └── tag-management.html    # Tag management interface
├── material_descriptions.json  # Material tag descriptions (managed via UI)
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

### Material Tags

Tags are now managed through the web interface at `/tag-management`. The system stores tags in `material_descriptions.json`.

**Tag Format Rules:**
- 1-3 uppercase letters
- Optional dash (`-`) or underscore (`_`)
- 1-2 digits
- Examples: `A1`, `BR-1`, `MT-05`, `FCL-2`, `GL_10`

You can also manually edit `material_descriptions.json` if needed:

```json
{
  "BR-1": "SOUTH & NORTH GROUND FLOOR - FACE BRICK 1",
  "FCL-2": "RESIDENTIAL LEVELS - FIBER CEMENT LAPSIDING 2",
  ...
}
```

## Tech Stack

- **Backend**: FastAPI, WebSockets
- **AI/ML**: Roboflow (object detection), EasyOCR (text extraction)
- **Frontend**: HTML5, CSS3, JavaScript, Lucide Icons
- **Image Processing**: PIL/Pillow
- **PDF Processing**: pdfplumber

## API Endpoints

### Main Application
- `GET /` - Main web interface
- `GET /tag-management` - Tag management interface
- `POST /upload` - Upload PDF file
- `POST /process` - Start processing (via WebSocket)
- `WebSocket /ws` - Real-time updates
- `GET /download` - Download results CSV

### Tag Management API
- `GET /api/tags` - List all tags
- `POST /api/tags` - Add a single tag
- `DELETE /api/tags/{tag}` - Remove a tag
- `POST /api/tags/upload-csv` - Bulk upload tags from CSV
- `GET /api/tags/download-csv` - Download current tags as CSV

## AWS EC2 Deployment (CI/CD)

This project includes a fully automated CI/CD pipeline that deploys to AWS EC2 on every push to `main`.

### GitHub Secrets Required

| Secret | Description | Example |
|--------|-------------|---------|
| `EC2_HOST` | EC2 public IP address | `13.216.238.153` |
| `EC2_SSH_KEY` | Private SSH key (PEM format) | `-----BEGIN RSA PRIVATE KEY-----...` |
| `ROBOFLOW_API_KEY` | Your Roboflow API key | `abc123...` |
| `ROBOFLOW_MODEL_ID` | Roboflow model ID | `cad-drawing-iy9tc/11` |

### What the CI/CD Does Automatically

1. **System Setup**: Configures kernel buffers for large file uploads
2. **Python 3.12**: Installs Python 3.12 and creates venv
3. **Tesseract OCR**: Builds from source (Amazon Linux 2023 compatibility)
4. **Nginx**: Configures reverse proxy with WebSocket support
5. **Systemd Service**: Sets up auto-restart on crash
6. **Large File Support**: 2GB upload limit, 2-hour timeouts
7. **WebSocket**: 24-hour keepalive for long processing jobs

### EC2 Instance Requirements

- **OS**: Amazon Linux 2023 (recommended)
- **Instance Type**: `t3.medium` or larger (for processing PDFs)
- **Storage**: 20GB+ EBS volume
- **Security Group**: Allow ports 22 (SSH), 80 (HTTP)

### Manual Deployment Commands

If you need to deploy manually:

```bash
# SSH into EC2
ssh -i your-key.pem ec2-user@your-ec2-ip

# Check service status
sudo systemctl status cad-tracker

# View logs
sudo journalctl -u cad-tracker -f

# Restart service
sudo systemctl restart cad-tracker

# Check Nginx
sudo nginx -t
sudo systemctl restart nginx
```

## Troubleshooting

### EasyOCR Slow Startup
- On the very first run, EasyOCR downloads model files. This can take 1-2 minutes. Subsequent runs are faster.
- Ensure you have a stable internet connection for the first run.

### OCR returns empty results
- Check lighting/contrast of the drawing. EasyOCR is robust but very blurry text may fail.
- Check debug logs to see if tags are being detected but filtered out due to low confidence.

### Service crashes repeatedly
- Check logs: `sudo journalctl -u cad-tracker -n 100`
- Verify Python packages: `source venv312/bin/activate && pip list`
- Check disk space and memory: `df -h && free -m`


