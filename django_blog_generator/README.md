# Django SEO Blog Generator

A powerful web application that automatically generates SEO-optimized blog posts using AI. Built with Django and integrated with your existing SEO content automation script.

## Features

🤖 **AI-Powered Content Generation**

- Analyzes top competitors automatically
- Generates comprehensive blog outlines
- Creates SEO-optimized content using Google Gemini AI
- Addresses content gaps in the market

📊 **Competitor Analysis**

- Scrapes and analyzes top-ranking competitor blogs
- Identifies content gaps and opportunities
- Extracts successful keyword strategies
- Provides People Also Ask (PAA) question coverage

📄 **Document Generation**

- Automatically saves generated blogs as Word documents (.docx)
- Preserves formatting and structure
- Easy download and sharing

🎨 **Modern Web Interface**

- Clean, responsive Bootstrap-based UI
- Real-time generation status tracking
- Blog history and management
- Admin interface for advanced management

## Prerequisites

- Python 3.8 or higher
- Django 5.1+
- API keys for:
  - SERP API (for competitor research)
  - Google Gemini AI (for content generation)

## Installation

### 1. Clone and Navigate

```bash
cd django_blog_generator
```

### 2. Set Up Virtual Environment (Recommended)

```bash
# If not already activated
source ../env/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the parent directory with your API keys:

```bash
# In /Users/sanketpatel/Desktop/NetopDigital/Content_Generation_Automation/.env
SERP_API_KEY=your_serp_api_key_here
GOOGLE_GEMINI_API_KEY=your_google_gemini_api_key_here
```

### 5. Run Setup Script

```bash
python setup.py
```

This script will:

- Install all Python packages
- Create database migrations
- Apply migrations
- Create necessary directories
- Optionally create a Django superuser

### 6. Start Development Server

```bash
python manage.py runserver
```

Visit `http://127.0.0.1:8000` to start using the application!

## Usage

### Generating a Blog Post

1. **Enter Blog Information**

   - Blog title or main topic
   - Primary keywords (comma-separated)
   - Number of competitors to analyze (1-10)

2. **Start Generation**

   - Click "Generate Blog Post"
   - The system will analyze competitors and generate content
   - Process typically takes 2-5 minutes

3. **Download Results**
   - View the generated blog content
   - Download as a Word document
   - Access competitor analysis data

### Admin Interface

Access `http://127.0.0.1:8000/admin` to:

- Manage blog generations
- View detailed analytics
- Monitor system performance
- Access raw API responses

## Project Structure

```
django_blog_generator/
├── blog_automation/                 # Main Django app
│   ├── models.py                   # Database models
│   ├── views.py                    # Web views and logic
│   ├── forms.py                    # Web forms
│   ├── services.py                 # SEO automation integration
│   ├── admin.py                    # Django admin configuration
│   ├── urls.py                     # URL routing
│   ├── templates/                  # HTML templates
│   └── templatetags/               # Custom template filters
├── blog_generator/                 # Django project settings
│   ├── settings.py                 # Project configuration
│   ├── urls.py                     # Main URL routing
│   └── wsgi.py                     # WSGI configuration
├── media/                          # Generated files storage
│   └── generated_blogs/            # Word documents
├── manage.py                       # Django management script
├── requirements.txt                # Python dependencies
├── setup.py                       # Setup automation script
└── README.md                      # This file
```

## API Configuration

### SERP API

- Used for competitor research and search results
- Get your key from [serpapi.com](https://serpapi.com)
- Required for competitor analysis

### Google Gemini AI

- Powers content generation and analysis
- Get your key from [Google AI Studio](https://makersuite.google.com/app/apikey)
- Required for blog content generation

## Features in Detail

### Content Generation Process

1. **Competitor Discovery**

   - Searches Google for top-ranking pages
   - Filters out social media and generic sites
   - Selects diverse, high-quality sources

2. **Content Analysis**

   - Scrapes competitor blog content
   - Analyzes headings, structure, and keywords
   - Identifies content gaps and opportunities

3. **AI-Powered Generation**

   - Creates comprehensive blog outlines
   - Generates full blog posts with SEO optimization
   - Ensures E-E-A-T (Experience, Expertise, Authoritativeness, Trustworthiness) compliance

4. **Document Creation**
   - Formats content for publication
   - Saves as Word document with proper formatting
   - Includes meta descriptions and SEO elements

### Database Schema

The application uses a simple but comprehensive database schema:

- **BlogGeneration Model**: Stores all blog generation requests and results
  - Basic info (title, keywords, settings)
  - Generation status and timestamps
  - Generated content and analysis data
  - File paths and error handling

## Troubleshooting

### Common Issues

**1. API Keys Not Working**

- Ensure `.env` file is in the correct location
- Check API key validity and quotas
- Verify environment variable names match exactly

**2. Blog Generation Fails**

- Check internet connectivity
- Verify API key permissions
- Review error logs in Django admin or console

**3. File Download Issues**

- Ensure media directory has write permissions
- Check if Word document was created successfully
- Verify file path in database matches actual file location

### Development Tips

**Running in Debug Mode**

- Set `DEBUG = True` in settings.py for development
- Use `python manage.py runserver` with `--settings` flag for custom settings

**Database Management**

```bash
# Create new migrations after model changes
python manage.py makemigrations

# Apply migrations
python manage.py migrate

# Create superuser
python manage.py createsuperuser
```

**Viewing Logs**

- Check `blog_generator.log` for application logs
- Use Django admin to view detailed generation data
- Enable console logging for real-time debugging

## Integration with Original Script

This Django application integrates seamlessly with your existing `seo_content_automation.py` script:

- **Service Layer**: `services.py` wraps original functions
- **Import System**: Dynamically imports from parent directory
- **Error Handling**: Graceful fallbacks if original script unavailable
- **Data Persistence**: Stores all results in database for future reference

## Production Deployment

For production deployment, consider:

1. **Environment Variables**

   - Use proper environment variable management
   - Keep API keys secure and never commit to version control

2. **Database**

   - Use PostgreSQL or MySQL instead of SQLite
   - Configure proper database backups

3. **Static Files**

   - Configure static file serving with nginx or Apache
   - Use cloud storage for media files

4. **Security**
   - Set `DEBUG = False`
   - Configure `ALLOWED_HOSTS`
   - Use HTTPS in production
   - Implement proper authentication for admin access

## License

This project is part of the NetopDigital Content Generation Automation system.

## Support

For issues or questions:

1. Check the troubleshooting section above
2. Review Django and API documentation
3. Examine application logs for error details
4. Ensure all prerequisites are properly configured
