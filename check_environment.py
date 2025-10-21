import sys
import os
import time

def check_environment():
    print("🔍 DIAGNOSTIC ENVIRONNEMENT")
    print("=" * 50)
    
    # Informations Python
    print(f"Python exe: {sys.executable}")
    print(f"Python version: {sys.version}")
    print(f"Architecture: {sys.platform}")
    
    # Chemin
    print(f"Current dir: {os.getcwd()}")
    print(f"Python path: {sys.path[:3]}")  # 3 premiers
    
    # Mémoire
    import psutil
    process = psutil.Process()
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB")
    
    print("=" * 50)

if __name__ == "__main__":
    check_environment()