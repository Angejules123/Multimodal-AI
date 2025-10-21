"""
Pipeline de Prétraitement EEG pour le Diagnostic Précoce des Troubles Cognitifs
Conforme au cahier des charges - Section F2.1

Auteurs: ANGE JULES & MAHAMADOU SAADOU ADANANE
"""

import numpy as np
import pandas as pd
import mne
from scipy import signal
from scipy.fft import fft, fftfreq
from sklearn.decomposition import FastICA
import os
import zipfile
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class EEGPreprocessor:
    """
    Classe principale pour le prétraitement des données EEG
    Implémente les spécifications F2.1 du cahier des charges
    """
    
    def __init__(self, base_path, output_path='preprocessed_data'):
        """
        Initialise le préprocesseur
        
        Args:
            base_path: Chemin vers les données brutes
            output_path: Chemin de sortie pour les données prétraitées
        """
        self.base_path = Path(base_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)
        
        # Paramètres selon cahier des charges
        self.params = {
            'freq_bands': {
                'delta': (0.5, 4),
                'theta': (4, 8),
                'alpha': (8, 13),
                'beta': (13, 30),
                'gamma': (30, 50)
            },
            'filter_low': 0.5,
            'filter_high': 50,
            'sfreq': 250,  # À ajuster selon vos données
            'ica_n_components': 15
        }
        
        self.quality_report = []
        
    def extract_participant_data(self, dataset_id, participant_id):
        """
        Extrait les données d'un participant depuis le zip
        
        Args:
            dataset_id: ID du dataset (4917218 ou 5055046)
            participant_id: ID du participant (P01-P15)
        """
        zip_path = self.base_path / str(dataset_id) / f"{participant_id}.zip"
        
        if not zip_path.exists():
            raise FileNotFoundError(f"Fichier non trouvé: {zip_path}")
        
        extract_path = self.output_path / 'temp' / dataset_id / participant_id
        extract_path.mkdir(parents=True, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        
        return extract_path
    
    def load_eeg_data(self, data_path):
        """
        Charge les données EEG selon le format disponible
        Supporte: .fif, .edf, .bdf, .set
        """
        data_files = list(Path(data_path).rglob('*.fif')) + \
                     list(Path(data_path).rglob('*.edf')) + \
                     list(Path(data_path).rglob('*.bdf')) + \
                     list(Path(data_path).rglob('*.set'))
        
        if not data_files:
            raise FileNotFoundError("Aucun fichier EEG trouvé")
        
        file_path = data_files[0]
        
        # Détection format et chargement approprié
        if file_path.suffix == '.fif':
            raw = mne.io.read_raw_fif(file_path, preload=True, verbose=False)
        elif file_path.suffix in ['.edf', '.bdf']:
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        elif file_path.suffix == '.set':
            raw = mne.io.read_raw_eeglab(file_path, preload=True, verbose=False)
        else:
            raise ValueError(f"Format non supporté: {file_path.suffix}")
        
        return raw
    
    def quality_check(self, raw, participant_id):
        """
        F1.3 - Validation qualité données
        Détecte les artéfacts et vérifie l'intégrité
        """
        quality_metrics = {
            'participant': participant_id,
            'n_channels': len(raw.ch_names),
            'duration': raw.times[-1],
            'sampling_freq': raw.info['sfreq'],
            'bad_channels': [],
            'artifacts_detected': {}
        }
        
        # Détection canaux défectueux (variance excessive)
        data = raw.get_data()
        channel_std = np.std(data, axis=1)
        threshold = np.median(channel_std) * 3
        bad_idx = np.where(channel_std > threshold)[0]
        quality_metrics['bad_channels'] = [raw.ch_names[i] for i in bad_idx]
        
        # Détection saturation
        saturation = np.sum(np.abs(data) > np.percentile(np.abs(data), 99.9), axis=1)
        quality_metrics['artifacts_detected']['saturation'] = int(np.sum(saturation > 10))
        
        # Détection pics (bruit électrique)
        spikes = np.sum(np.abs(np.diff(data, axis=1)) > 50e-6, axis=1)
        quality_metrics['artifacts_detected']['spikes'] = int(np.sum(spikes > 5))
        
        self.quality_report.append(quality_metrics)
        
        return quality_metrics
    
    def apply_bandpass_filter(self, raw):
        """
        F2.1 - Filtrage passe-bande (0.5-50 Hz)
        """
        raw_filtered = raw.copy()
        raw_filtered.filter(
            l_freq=self.params['filter_low'],
            h_freq=self.params['filter_high'],
            fir_design='firwin',
            verbose=False
        )
        return raw_filtered
    
    def remove_artifacts_ica(self, raw):
        """
        F2.1 - Suppression artéfacts (ICA pour clignements, ECG)
        """
        # Configuration ICA
        ica = FastICA(
            n_components=self.params['ica_n_components'],
            random_state=42,
            max_iter=500
        )
        
        # Application ICA
        data = raw.get_data()
        ica_components = ica.fit_transform(data.T).T
        
        # Détection composantes artéfactuelles
        # Clignements (corrélation avec électrodes frontales)
        frontal_channels = [i for i, name in enumerate(raw.ch_names) 
                           if 'Fp' in name or 'AF' in name]
        
        artifact_components = []
        if frontal_channels:
            frontal_data = data[frontal_channels].mean(axis=0)
            for i in range(ica_components.shape[0]):
                corr = np.corrcoef(ica_components[i], frontal_data)[0, 1]
                if abs(corr) > 0.7:  # Seuil de corrélation
                    artifact_components.append(i)
        
        # Reconstruction sans artéfacts
        ica_components[artifact_components] = 0
        clean_data = ica.inverse_transform(ica_components.T).T
        
        # Création nouveau objet Raw
        info = raw.info
        raw_clean = mne.io.RawArray(clean_data, info, verbose=False)
        
        return raw_clean, len(artifact_components)
    
    def spectral_decomposition(self, raw):
        """
        F2.1 - Décomposition spectrale (delta, theta, alpha, beta, gamma)
        F3.2 - Extraction puissance spectrale par bande
        """
        data = raw.get_data()
        sfreq = raw.info['sfreq']
        
        band_powers = {}
        
        for band_name, (low, high) in self.params['freq_bands'].items():
            # Filtrage par bande
            sos = signal.butter(4, [low, high], btype='bandpass', 
                               fs=sfreq, output='sos')
            filtered = signal.sosfiltfilt(sos, data, axis=1)
            
            # Calcul puissance (moyenne du carré du signal)
            power = np.mean(filtered ** 2, axis=1)
            band_powers[band_name] = power
        
        return band_powers
    
    def compute_connectivity(self, raw):
        """
        F3.2 - Cohérence inter-électrodes
        """
        data = raw.get_data()
        n_channels = data.shape[0]
        
        # Matrice de connectivité (corrélation de Pearson)
        connectivity_matrix = np.corrcoef(data)
        
        # Métriques de connectivité
        connectivity_features = {
            'mean_connectivity': np.mean(connectivity_matrix[np.triu_indices(n_channels, k=1)]),
            'max_connectivity': np.max(connectivity_matrix[np.triu_indices(n_channels, k=1)]),
            'global_efficiency': self._compute_global_efficiency(connectivity_matrix)
        }
        
        return connectivity_matrix, connectivity_features
    
    def _compute_global_efficiency(self, conn_matrix):
        """Calcule l'efficacité globale du réseau"""
        # Inverse des distances (1/correlation)
        with np.errstate(divide='ignore', invalid='ignore'):
            dist_matrix = 1 / (conn_matrix + 1e-10)
            dist_matrix[np.isinf(dist_matrix)] = 0
        
        n = conn_matrix.shape[0]
        efficiency = 0
        
        for i in range(n):
            for j in range(i+1, n):
                if dist_matrix[i, j] > 0:
                    efficiency += 1 / dist_matrix[i, j]
        
        return 2 * efficiency / (n * (n - 1)) if n > 1 else 0
    
    def extract_features(self, raw, band_powers, connectivity_features):
        """
        F3.1 et F3.2 - Extraction complète des features
        """
        features = {}
        
        # Features spectrales
        for band, powers in band_powers.items():
            features[f'{band}_mean'] = np.mean(powers)
            features[f'{band}_std'] = np.std(powers)
            features[f'{band}_max'] = np.max(powers)
        
        # Features de connectivité
        features.update(connectivity_features)
        
        # Features statistiques globales
        data = raw.get_data()
        features['signal_mean'] = np.mean(data)
        features['signal_std'] = np.std(data)
        features['signal_kurtosis'] = np.mean([
            np.mean((data[i] - np.mean(data[i]))**4) / (np.std(data[i])**4)
            for i in range(data.shape[0])
        ])
        features['signal_skewness'] = np.mean([
            np.mean((data[i] - np.mean(data[i]))**3) / (np.std(data[i])**3)
            for i in range(data.shape[0])
        ])
        
        return features
    
    def preprocess_participant(self, dataset_id, participant_id):
        """
        Pipeline complet de prétraitement pour un participant
        """
        print(f"\n{'='*60}")
        print(f"Prétraitement: Dataset {dataset_id} - {participant_id}")
        print(f"{'='*60}")
        
        try:
            # 1. Extraction données
            print("→ Extraction des données...")
            data_path = self.extract_participant_data(dataset_id, participant_id)
            
            # 2. Chargement EEG
            print("→ Chargement EEG...")
            raw = self.load_eeg_data(data_path)
            
            # 3. Contrôle qualité
            print("→ Contrôle qualité...")
            quality = self.quality_check(raw, participant_id)
            print(f"  ✓ {quality['n_channels']} canaux, {quality['duration']:.1f}s")
            print(f"  ✓ Canaux défectueux: {len(quality['bad_channels'])}")
            
            # 4. Filtrage passe-bande
            print("→ Filtrage passe-bande (0.5-50 Hz)...")
            raw_filtered = self.apply_bandpass_filter(raw)
            
            # 5. Suppression artéfacts ICA
            print("→ Suppression artéfacts (ICA)...")
            raw_clean, n_artifacts = self.remove_artifacts_ica(raw_filtered)
            print(f"  ✓ {n_artifacts} composantes artéfactuelles supprimées")
            
            # 6. Décomposition spectrale
            print("→ Décomposition spectrale...")
            band_powers = self.spectral_decomposition(raw_clean)
            
            # 7. Analyse connectivité
            print("→ Analyse de connectivité...")
            conn_matrix, conn_features = self.compute_connectivity(raw_clean)
            
            # 8. Extraction features
            print("→ Extraction des features...")
            features = self.extract_features(raw_clean, band_powers, conn_features)
            
            # 9. Sauvegarde
            print("→ Sauvegarde des résultats...")
            output_dir = self.output_path / dataset_id / participant_id
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Sauvegarde données nettoyées
            raw_clean.save(output_dir / 'clean_raw.fif', overwrite=True, verbose=False)
            
            # Sauvegarde features
            pd.DataFrame([features]).to_csv(output_dir / 'features.csv', index=False)
            
            # Sauvegarde connectivité
            np.save(output_dir / 'connectivity_matrix.npy', conn_matrix)
            
            # Sauvegarde puissances spectrales
            np.savez(output_dir / 'band_powers.npz', **band_powers)
            
            print(f"✅ Prétraitement terminé: {output_dir}")
            
            return {
                'status': 'success',
                'participant': participant_id,
                'features': features,
                'quality': quality
            }
            
        except Exception as e:
            print(f"❌ Erreur: {str(e)}")
            return {
                'status': 'error',
                'participant': participant_id,
                'error': str(e)
            }
    
    def preprocess_all_datasets(self):
        """
        Prétraite tous les participants de tous les datasets
        """
        datasets = ['4917218', '5055046']
        participants = [f'P{i:02d}' for i in range(1, 16)]
        
        results = []
        
        for dataset_id in datasets:
            for participant_id in participants:
                result = self.preprocess_participant(dataset_id, participant_id)
                results.append(result)
        
        # Génération rapport global
        self.generate_quality_report()
        
        return results
    
    def generate_quality_report(self):
        """
        F1.3 - Rapport qualité automatisé
        """
        if not self.quality_report:
            print("Aucune donnée de qualité disponible")
            return
        
        df_quality = pd.DataFrame(self.quality_report)
        
        report_path = self.output_path / 'quality_report.csv'
        df_quality.to_csv(report_path, index=False)
        
        print(f"\n{'='*60}")
        print("RAPPORT DE QUALITÉ GLOBAL")
        print(f"{'='*60}")
        print(f"Total participants: {len(df_quality)}")
        print(f"Canaux moyens: {df_quality['n_channels'].mean():.1f}")
        print(f"Durée moyenne: {df_quality['duration'].mean():.1f}s")
        print(f"Participants avec canaux défectueux: {sum(df_quality['bad_channels'].apply(len) > 0)}")
        print(f"\nRapport sauvegardé: {report_path}")


# Fonction principale d'utilisation
def main():
    """
    Script principal d'exécution
    """
    # Configuration des chemins
    BASE_PATH = "e:/Master data science/MPDS3_2025/projet federal/"
    OUTPUT_PATH = "e:/Master data science/MPDS3_2025/projet federal/preprocessed_data"
    
    # Initialisation du préprocesseur
    preprocessor = EEGPreprocessor(BASE_PATH, OUTPUT_PATH)
    
    # Option 1: Prétraiter un seul participant (pour test)
    print("TEST: Prétraitement d'un participant...")
    result = preprocessor.preprocess_participant('4917218', 'P01')
    
    # Option 2: Prétraiter tous les participants (décommenter pour utilisation complète)
    # print("Prétraitement de tous les participants...")
    # results = preprocessor.preprocess_all_datasets()
    
    print("\n✅ Pipeline de prétraitement terminé!")


if __name__ == "__main__":
    main()