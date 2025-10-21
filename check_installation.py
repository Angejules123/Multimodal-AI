import importlib

required_packages = [
    'numpy', 'scipy', 'pandas', 'matplotlib', 'seaborn',
    'mne', 'sklearn', 'torch', 'pywt', 'antropy'
]

print("Vérification des packages installés:")
print("=" * 40)

for package in required_packages:
    try:
        mod = importlib.import_module(package)
        version = getattr(mod, '__version__', 'Version inconnue')
        print(f"✓ {package:20} {version}")
    except ImportError:
        print(f"✗ {package:20} NON INSTALLÉ")

print("=" * 40)

# Test MNE spécifique
try:
    import mne
    print(f"\nMNE version: {mne.__version__}")
    print("Fonctionnalités MNE disponibles:")
    print(f"- Filtering: {hasattr(mne, 'filter')}")
    print(f"- ICA: {hasattr(mne, 'ICA')}")
    print(f"- Epochs: {hasattr(mne, 'Epochs')}")
except ImportError as e:
    print(f"Erreur MNE: {e}")