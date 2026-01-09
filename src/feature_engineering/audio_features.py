"""
Audio feature extraction for bird stress analysis
"""

import numpy as np
import librosa
import soundfile as sf
from typing import Dict, Optional, Tuple, List
from pathlib import Path
from loguru import logger
from scipy import signal
from scipy.stats import entropy


class AudioFeatureExtractor:
    """Extract acoustic features from bird audio recordings"""
    
    def __init__(
        self,
        sample_rate: int = 22050,
        n_mfcc: int = 40,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        fmin: int = 0,
        fmax: int = 11025
    ):
        """
        Initialize audio feature extractor
        
        Args:
            sample_rate: Target sample rate for audio
            n_mfcc: Number of MFCCs to extract
            n_fft: FFT window size
            hop_length: Hop length for STFT
            n_mels: Number of Mel bands
            fmin: Minimum frequency
            fmax: Maximum frequency
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        
        logger.info(f"Initialized AudioFeatureExtractor with sr={sample_rate}, n_mfcc={n_mfcc}")
    
    def load_audio(
        self,
        audio_path: Path,
        duration: Optional[float] = None,
        offset: float = 0.0
    ) -> Tuple[np.ndarray, int]:
        """
        Load audio file
        
        Args:
            audio_path: Path to audio file
            duration: Duration to load in seconds (None = load all)
            offset: Start offset in seconds
            
        Returns:
            Audio array and sample rate
        """
        try:
            audio, sr = librosa.load(
                audio_path,
                sr=self.sample_rate,
                duration=duration,
                offset=offset,
                mono=True
            )
            
            logger.debug(f"Loaded audio: {audio_path.name}, length={len(audio)/sr:.2f}s")
            return audio, sr
            
        except Exception as e:
            logger.error(f"Failed to load audio {audio_path}: {str(e)}")
            raise
    
    def extract_mfcc(
        self,
        audio: np.ndarray,
        n_mfcc: Optional[int] = None
    ) -> np.ndarray:
        """
        Extract Mel-frequency cepstral coefficients (MFCCs)
        
        Args:
            audio: Audio array
            n_mfcc: Number of MFCCs (overrides default)
            
        Returns:
            MFCC array of shape (n_mfcc, time_steps)
        """
        n_mfcc = n_mfcc or self.n_mfcc
        
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax
        )
        
        # Delta and delta-delta features
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        # Concatenate
        mfcc_features = np.concatenate([mfcc, mfcc_delta, mfcc_delta2], axis=0)
        
        logger.debug(f"Extracted MFCC features: shape={mfcc_features.shape}")
        return mfcc_features
    
    def extract_spectral_centroid(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract spectral centroid (center of mass of spectrum)
        
        Args:
            audio: Audio array
            
        Returns:
            Spectral centroid array
        """
        centroid = librosa.feature.spectral_centroid(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        return centroid[0]
    
    def extract_spectral_bandwidth(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract spectral bandwidth
        
        Args:
            audio: Audio array
            
        Returns:
            Spectral bandwidth array
        """
        bandwidth = librosa.feature.spectral_bandwidth(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        return bandwidth[0]
    
    def extract_spectral_rolloff(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract spectral rolloff (frequency below which X% of energy is contained)
        
        Args:
            audio: Audio array
            
        Returns:
            Spectral rolloff array
        """
        rolloff = librosa.feature.spectral_rolloff(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            roll_percent=0.85
        )
        
        return rolloff[0]
    
    def extract_zero_crossing_rate(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract zero-crossing rate
        
        Args:
            audio: Audio array
            
        Returns:
            Zero-crossing rate array
        """
        zcr = librosa.feature.zero_crossing_rate(
            y=audio,
            frame_length=self.n_fft,
            hop_length=self.hop_length
        )
        
        return zcr[0]
    
    def extract_spectral_entropy(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract spectral entropy (measure of spectral complexity)
        
        Args:
            audio: Audio array
            
        Returns:
            Spectral entropy array
        """
        # Compute spectrogram
        spec = np.abs(librosa.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        ))
        
        # Normalize each frame
        spec_norm = spec / (np.sum(spec, axis=0, keepdims=True) + 1e-10)
        
        # Compute entropy for each time frame
        spectral_entropy = np.apply_along_axis(
            lambda x: entropy(x + 1e-10),
            0,
            spec_norm
        )
        
        return spectral_entropy
    
    def extract_spectral_flatness(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract spectral flatness (measure of tonality vs. noise)
        
        Args:
            audio: Audio array
            
        Returns:
            Spectral flatness array
        """
        flatness = librosa.feature.spectral_flatness(
            y=audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        return flatness[0]
    
    def extract_chroma(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract chromagram features
        
        Args:
            audio: Audio array
            
        Returns:
            Chroma features of shape (12, time_steps)
        """
        chroma = librosa.feature.chroma_stft(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        return chroma
    
    def extract_rms_energy(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract RMS energy
        
        Args:
            audio: Audio array
            
        Returns:
            RMS energy array
        """
        rms = librosa.feature.rms(
            y=audio,
            frame_length=self.n_fft,
            hop_length=self.hop_length
        )
        
        return rms[0]
    
    def extract_all_features(
        self,
        audio: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Extract all audio features
        
        Args:
            audio: Audio array
            
        Returns:
            Dictionary of all extracted features
        """
        logger.info("Extracting all audio features...")
        
        features = {
            'mfcc': self.extract_mfcc(audio),
            'spectral_centroid': self.extract_spectral_centroid(audio),
            'spectral_bandwidth': self.extract_spectral_bandwidth(audio),
            'spectral_rolloff': self.extract_spectral_rolloff(audio),
            'zero_crossing_rate': self.extract_zero_crossing_rate(audio),
            'spectral_entropy': self.extract_spectral_entropy(audio),
            'spectral_flatness': self.extract_spectral_flatness(audio),
            'chroma': self.extract_chroma(audio),
            'rms_energy': self.extract_rms_energy(audio)
        }
        
        logger.info(f"Extracted {len(features)} feature types")
        return features
    
    def get_feature_statistics(
        self,
        features: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        Compute statistical summary of features
        
        Args:
            features: Dictionary of feature arrays
            
        Returns:
            Dictionary of feature statistics
        """
        stats = {}
        
        for feature_name, feature_array in features.items():
            if feature_name == 'mfcc':
                # For MFCC, compute stats for each coefficient
                for i in range(feature_array.shape[0]):
                    stats[f'{feature_name}_{i}_mean'] = np.mean(feature_array[i])
                    stats[f'{feature_name}_{i}_std'] = np.std(feature_array[i])
            elif feature_name == 'chroma':
                # For chroma, compute stats for each pitch class
                for i in range(12):
                    stats[f'{feature_name}_{i}_mean'] = np.mean(feature_array[i])
            else:
                # For other features, compute basic statistics
                stats[f'{feature_name}_mean'] = np.mean(feature_array)
                stats[f'{feature_name}_std'] = np.std(feature_array)
                stats[f'{feature_name}_min'] = np.min(feature_array)
                stats[f'{feature_name}_max'] = np.max(feature_array)
                stats[f'{feature_name}_median'] = np.median(feature_array)
        
        return stats
    
    def segment_audio(
        self,
        audio: np.ndarray,
        segment_length: float = 5.0,
        overlap: float = 0.5
    ) -> List[np.ndarray]:
        """
        Segment audio into overlapping windows
        
        Args:
            audio: Audio array
            segment_length: Length of each segment in seconds
            overlap: Overlap between segments (0.0 to 1.0)
            
        Returns:
            List of audio segments
        """
        segment_samples = int(segment_length * self.sample_rate)
        hop_samples = int(segment_samples * (1 - overlap))
        
        segments = []
        for start in range(0, len(audio) - segment_samples + 1, hop_samples):
            end = start + segment_samples
            segments.append(audio[start:end])
        
        logger.debug(f"Segmented audio into {len(segments)} segments")
        return segments
    
    def process_audio_file(
        self,
        audio_path: Path,
        segment_length: float = 5.0,
        overlap: float = 0.5
    ) -> List[Dict[str, np.ndarray]]:
        """
        Process audio file and extract features from segments
        
        Args:
            audio_path: Path to audio file
            segment_length: Length of each segment in seconds
            overlap: Overlap between segments
            
        Returns:
            List of feature dictionaries for each segment
        """
        # Load audio
        audio, _ = self.load_audio(audio_path)
        
        # Segment audio
        segments = self.segment_audio(audio, segment_length, overlap)
        
        # Extract features from each segment
        all_features = []
        for i, segment in enumerate(segments):
            logger.debug(f"Processing segment {i+1}/{len(segments)}")
            features = self.extract_all_features(segment)
            all_features.append(features)
        
        logger.info(f"Processed {len(all_features)} segments from {audio_path.name}")
        return all_features
