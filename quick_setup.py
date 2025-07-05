#!/usr/bin/env python3
"""
Quick Setup Script for SmolVLM-GeoEye Development Environment
============================================================

This script automates the setup of the development environment
for the SmolVLM-GeoEye geotechnical engineering workflow.

Author: SmolVLM-GeoEye Team
Version: 1.0.0
"""

import os
import sys
import subprocess
import platform
import venv
import shutil
from pathlib import Path
from typing import List, Tuple, Optional
import json

class QuickSetup:
    """Handles quick setup of development environment"""
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.venv_name = "geotechnical_env"
        self.venv_path = self.project_root / self.venv_name
        self.requirements_file = self.project_root / "requirements.txt"
        self.env_file = self.project_root / ".env"
        self.env_example_file = self.project_root / ".env.example"
        self.system = platform.system()
        self.python_executable = sys.executable
        
    def print_header(self):
        """Print setup header"""
        print("="*60)
        print("🏗️  SmolVLM-GeoEye Quick Setup")
        print("="*60)
        print(f"System: {self.system}")
        print(f"Python: {sys.version.split()[0]}")
        print(f"Project Root: {self.project_root}")
        print("="*60)
    
    def check_python_version(self) -> bool:
        """Check if Python version is compatible"""
        print("\n📌 Checking Python version...")
        
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            print(f"❌ Python {version.major}.{version.minor} detected.")
            print("   Python 3.8+ is required.")
            return False
        
        print(f"✅ Python {version.major}.{version.minor} is compatible")
        return True
    
    def create_virtual_environment(self) -> bool:
        """Create virtual environment"""
        print(f"\n📌 Creating virtual environment '{self.venv_name}'...")
        
        try:
            if self.venv_path.exists():
                print(f"   Virtual environment already exists at {self.venv_path}")
                response = input("   Do you want to recreate it? (y/N): ").strip().lower()
                if response == 'y':
                    print("   Removing existing environment...")
                    shutil.rmtree(self.venv_path)
                else:
                    print("   Using existing environment")
                    return True
            
            # Create virtual environment
            venv.create(self.venv_path, with_pip=True)
            print(f"✅ Virtual environment created at {self.venv_path}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to create virtual environment: {str(e)}")
            return False
    
    def get_pip_command(self) -> List[str]:
        """Get the pip command for the virtual environment"""
        if self.system == "Windows":
            pip_path = self.venv_path / "Scripts" / "pip.exe"
        else:
            pip_path = self.venv_path / "bin" / "pip"
        
        return [str(pip_path)]
    
    def get_python_command(self) -> str:
        """Get the python command for the virtual environment"""
        if self.system == "Windows":
            return str(self.venv_path / "Scripts" / "python.exe")
        else:
            return str(self.venv_path / "bin" / "python")
    
    def upgrade_pip(self) -> bool:
        """Upgrade pip in virtual environment"""
        print("\n📌 Upgrading pip...")
        
        try:
            pip_cmd = self.get_pip_command()
            subprocess.run(
                pip_cmd + ["install", "--upgrade", "pip"],
                check=True,
                capture_output=True,
                text=True
            )
            print("✅ Pip upgraded successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to upgrade pip: {e.stderr}")
            return False
    
    def install_requirements(self) -> bool:
        """Install requirements from requirements.txt"""
        print("\n📌 Installing requirements...")
        
        if not self.requirements_file.exists():
            print(f"❌ Requirements file not found: {self.requirements_file}")
            return False
        
        try:
            pip_cmd = self.get_pip_command()
            
            # Install requirements
            print("   Installing from requirements.txt...")
            result = subprocess.run(
                pip_cmd + ["install", "-r", str(self.requirements_file)],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print("✅ All requirements installed successfully")
                return True
            else:
                print(f"❌ Failed to install requirements:")
                print(result.stderr)
                return False
                
        except Exception as e:
            print(f"❌ Error installing requirements: {str(e)}")
            return False
    
    def create_env_file(self) -> bool:
        """Create .env file from .env.example"""
        print("\n📌 Setting up environment variables...")
        
        # Create .env.example if it doesn't exist
        if not self.env_example_file.exists():
            print("   Creating .env.example...")
            example_content = """# RunPod Configuration (Required)
RUNPOD_API_KEY=your_runpod_api_key_here
RUNPOD_ENDPOINT_ID=your_runpod_endpoint_id_here

# HuggingFace Token (Optional - for SmolAgent)
HF_TOKEN=your_huggingface_token_here

# Environment Settings
ENVIRONMENT=development
DEBUG=true
"""
            self.env_example_file.write_text(example_content)
            print("   ✅ Created .env.example")
        
        # Check if .env exists
        if self.env_file.exists():
            print("   .env file already exists")
            return True
        
        # Copy .env.example to .env
        try:
            shutil.copy(self.env_example_file, self.env_file)
            print("✅ Created .env file from .env.example")
            print("\n   ⚠️  IMPORTANT: Edit .env file and add your credentials:")
            print("      - RUNPOD_API_KEY: Get from https://runpod.io/console/user/settings")
            print("      - RUNPOD_ENDPOINT_ID: Get from https://runpod.io/console/serverless")
            print("      - HF_TOKEN: Get from https://huggingface.co/settings/tokens")
            return True
        except Exception as e:
            print(f"❌ Failed to create .env file: {str(e)}")
            return False
    
    def create_directories(self) -> bool:
        """Create necessary directories"""
        print("\n📌 Creating project directories...")
        
        directories = [
            "data",
            "data/uploads",
            "data/processed",
            "data/cache",
            "logs",
            "models",
            "outputs"
        ]
        
        try:
            for dir_name in directories:
                dir_path = self.project_root / dir_name
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"   ✅ {dir_name}/")
            
            # Create .gitkeep files to preserve empty directories
            for dir_name in directories:
                gitkeep_path = self.project_root / dir_name / ".gitkeep"
                gitkeep_path.touch(exist_ok=True)
            
            return True
        except Exception as e:
            print(f"❌ Failed to create directories: {str(e)}")
            return False
    
    def test_imports(self) -> bool:
        """Test critical imports"""
        print("\n📌 Testing critical imports...")
        
        python_cmd = self.get_python_command()
        
        test_script = """
import sys
try:
    import streamlit
    print("✅ Streamlit")
except ImportError as e:
    print(f"❌ Streamlit: {e}")
    sys.exit(1)

try:
    import pandas
    print("✅ Pandas")
except ImportError as e:
    print(f"❌ Pandas: {e}")
    sys.exit(1)

try:
    import plotly
    print("✅ Plotly")
except ImportError as e:
    print(f"❌ Plotly: {e}")
    sys.exit(1)

try:
    from smolagents import tool, ToolCallingAgent, HfApiModel
    print("✅ SmolAgents")
except ImportError as e:
    print(f"❌ SmolAgents: {e}")
    sys.exit(1)

try:
    import PyPDF2
    print("✅ PyPDF2")
except ImportError as e:
    print(f"❌ PyPDF2: {e}")
    sys.exit(1)

print("\\n✅ All critical imports successful!")
"""
        
        try:
            result = subprocess.run(
                [python_cmd, "-c", test_script],
                capture_output=True,
                text=True
            )
            
            print(result.stdout)
            return result.returncode == 0
            
        except Exception as e:
            print(f"❌ Import test failed: {str(e)}")
            return False
    
    def print_activation_instructions(self):
        """Print instructions for activating the environment"""
        print("\n" + "="*60)
        print("🎉 Setup Complete!")
        print("="*60)
        
        print("\n📝 Next Steps:")
        print("\n1. Activate the virtual environment:")
        
        if self.system == "Windows":
            print(f"   {self.venv_name}\\Scripts\\activate")
        else:
            print(f"   source {self.venv_name}/bin/activate")
        
        print("\n2. Configure your RunPod credentials:")
        print(f"   Edit the .env file and add your API keys")
        
        print("\n3. Validate your configuration:")
        print("   python validate_runpod.py")
        
        print("\n4. Run the application:")
        print("   streamlit run app.py")
        
        print("\n5. (Optional) Deploy to RunPod:")
        print("   ./deploy-runpod.sh all")
        
        print("\n📚 Documentation:")
        print("   - README.md: General overview and features")
        print("   - RUNPOD_DEPLOYMENT.md: Detailed deployment guide")
        print("   - CODE_CHANGES_GUIDE.md: VLM integration details")
        
        print("\n🔗 Useful Links:")
        print("   - RunPod Console: https://runpod.io/console")
        print("   - HuggingFace: https://huggingface.co")
        print("   - Project Repo: https://github.com/kilickursat/SmolVLM-GeoEye")
        
        print("\n" + "="*60)
    
    def run(self) -> bool:
        """Run the complete setup process"""
        self.print_header()
        
        # Check Python version
        if not self.check_python_version():
            return False
        
        # Create virtual environment
        if not self.create_virtual_environment():
            return False
        
        # Upgrade pip
        if not self.upgrade_pip():
            return False
        
        # Install requirements
        if not self.install_requirements():
            return False
        
        # Create .env file
        if not self.create_env_file():
            return False
        
        # Create directories
        if not self.create_directories():
            return False
        
        # Test imports
        if not self.test_imports():
            print("\n⚠️  Some imports failed. You may need to install additional packages.")
        
        # Print activation instructions
        self.print_activation_instructions()
        
        return True

def main():
    """Main setup function"""
    setup = QuickSetup()
    
    try:
        success = setup.run()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n❌ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Setup failed with error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()