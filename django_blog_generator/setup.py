#!/usr/bin/env python3
"""
Setup script for Django SEO Blog Generator
"""
import os
import sys
import subprocess
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False

def main():
    print("🚀 Setting up Django SEO Blog Generator")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("manage.py").exists():
        print("❌ Error: manage.py not found. Please run this script from the Django project directory.")
        sys.exit(1)
    
    # Check if virtual environment is activated
    if not os.getenv('VIRTUAL_ENV'):
        print("⚠️  Warning: No virtual environment detected. It's recommended to use a virtual environment.")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("Please activate your virtual environment and try again.")
            sys.exit(1)
    
    # Install requirements
    if not run_command("pip install -r requirements.txt", "Installing Python packages"):
        print("❌ Failed to install requirements. Please check your internet connection and try again.")
        sys.exit(1)
    
    # Create migrations
    if not run_command("python manage.py makemigrations", "Creating database migrations"):
        print("❌ Failed to create migrations.")
        sys.exit(1)
    
    # Apply migrations
    if not run_command("python manage.py migrate", "Applying database migrations"):
        print("❌ Failed to apply migrations.")
        sys.exit(1)
    
    # Create media directory
    media_dir = Path("media/generated_blogs")
    media_dir.mkdir(parents=True, exist_ok=True)
    print("✅ Created media directories")
    
    # Check for .env file
    env_file = Path("../.env")  # Look for .env in parent directory
    if not env_file.exists():
        print("\n📝 IMPORTANT: API Configuration Required")
        print("=" * 40)
        print("Create a .env file in the parent directory with the following keys:")
        print("SERP_API_KEY=your_serp_api_key_here")
        print("GOOGLE_GEMINI_API_KEY=your_google_gemini_api_key_here")
        print("\nWithout these API keys, the blog generation will not work.")
    else:
        print("✅ Found .env file in parent directory")
    
    # Ask about creating superuser
    create_superuser = input("\n🔐 Create Django superuser for admin access? (y/N): ")
    if create_superuser.lower() == 'y':
        run_command("python manage.py createsuperuser", "Creating superuser")
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Make sure your API keys are configured in the .env file")
    print("2. Run 'python manage.py runserver' to start the development server")
    print("3. Open http://127.0.0.1:8000 in your browser")
    print("4. Access admin at http://127.0.0.1:8000/admin (if you created a superuser)")

if __name__ == "__main__":
    main()