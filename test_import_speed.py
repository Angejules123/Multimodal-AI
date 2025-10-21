import time
import sys

def test_import_speed():
    print("🧪 TEST DE VITESSE DES IMPORTS")
    print("=" * 40)
    
    modules_to_test = [
        'numpy',
        'pandas', 
        'matplotlib',
        'seaborn',
        'pathlib'
    ]
    
    for module_name in modules_to_test:
        start_time = time.time()
        try:
            __import__(module_name)
            end_time = time.time()
            status = "✅"
        except ImportError:
            end_time = time.time()
            status = "❌"
        
        print(f"{status} {module_name:15} {end_time - start_time:.3f}s")
    
    print("=" * 40)

if __name__ == "__main__":
    test_import_speed()