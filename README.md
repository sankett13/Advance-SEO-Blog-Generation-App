# SEO Blog Generation App - Installation Guide

## Prerequisites

- Python 3.11 or higher
- pip (Python package manager)

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/sankett13/Advance-SEO-Blog-Generation-App.git
cd Advance-SEO-Blog-Generation-App
```

### 2. Create Virtual Environment

```bash
python -m venv env

# On macOS/Linux:
source env/bin/activate

# On Windows:
env\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Environment Setup

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env file with your API keys:
# - SERP_API_KEY: Get from https://serpapi.com/
# - GOOGLE_GEMINI_API_KEY: Get from https://makersuite.google.com/app/apikey
```

### 5. Database Setup

```bash
cd django_blog_generator
python manage.py migrate
```

### 6. Run the Application

```bash
python manage.py runserver
```

The application will be available at: http://127.0.0.1:8000/

## Features

- ✅ SEO-optimized blog post generation
- ✅ Competitor analysis
- ✅ AI-powered content creation with Google Gemini
- ✅ Word document export
- ✅ Optional fields for custom outlines and secondary keywords
- ✅ Web-based interface

## API Keys Required

1. **SERP API** - For competitor research and search results
2. **Google Gemini API** - For AI content generation

## Usage

1. Enter your blog title and primary keywords
2. Optionally add secondary keywords, custom outline, or target length
3. Click "Generate Blog Post"
4. Download the generated content as a Word document
