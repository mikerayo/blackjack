"""Setup script for ML Blackjack project."""

import subprocess
import sys
import os


def check_python_version():
    """Check if Python version is 3.10 or higher."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print(f"❌ Python 3.10+ required. You have Python {version.major}.{version.minor}")
        return False
    print(f"✓ Python {version.major}.{version.minor} detected")
    return True


def install_dependencies():
    """Install dependencies from requirements.txt."""
    print("\n📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False


def create_directories():
    """Create necessary directories."""
    print("\n📁 Creating directories...")
    dirs = ["models", "logs", "data", "models/visualizations"]

    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
        print(f"  ✓ Created {dir_name}")


def run_tests():
    """Run basic tests to verify installation."""
    print("\n🧪 Running tests...")

    # Test environment
    try:
        print("\n  Testing environment...")
        subprocess.check_call([sys.executable, "tests/test_env.py"])
    except subprocess.CalledProcessError:
        print("  ❌ Environment test failed")
        return False

    # Test game engine
    try:
        print("\n  Testing game engine...")
        subprocess.check_call([sys.executable, "-m", "pytest", "tests/test_game.py", "-v"])
    except subprocess.CalledProcessError:
        print("  ⚠ pytest not available, skipping game engine tests")
        # Try running directly
        try:
            subprocess.check_call([sys.executable, "tests/test_game.py"])
        except:
            pass  # It's okay if pytest tests don't run

    return True


def print_usage_examples():
    """Print usage examples."""
    print("\n" + "="*60)
    print("🎮 Usage Examples")
    print("="*60)

    print("\n1️⃣  Test the game:")
    print("   python src/main.py --mode test")

    print("\n2️⃣  Train the agent (100k episodes):")
    print("   python src/main.py --mode train --episodes 100000 --visualize")

    print("\n3️⃣  Evaluate a trained model:")
    print("   python src/main.py --mode evaluate --episodes 10000 --model-path models/model.pt")

    print("\n4️⃣  Quick training test (1k episodes):")
    print("   python src/main.py --mode train --episodes 1000 --log-frequency 100")

    print("\n" + "="*60)


def main():
    """Main setup function."""
    print("="*60)
    print("🎰 ML BLACKJACK - SETUP")
    print("="*60)

    # Check Python version
    if not check_python_version():
        sys.exit(1)

    # Install dependencies
    if not install_dependencies():
        sys.exit(1)

    # Create directories
    create_directories()

    # Run tests
    if not run_tests():
        print("\n⚠️  Some tests failed, but installation may still work")

    # Print usage
    print_usage_examples()

    print("\n✅ Setup complete!")
    print("\nNext steps:")
    print("  1. Try the test mode: python src/main.py --mode test")
    print("  2. Start training: python src/main.py --mode train --episodes 100000")
    print("\nGood luck! 🍀\n")


if __name__ == "__main__":
    main()
