"""
Time Series Dataset Creator with Disaster Labeling
Creates 168-hour sequences for LSTM training
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional
from loguru import logger
from tqdm import tqdm


class TimeSeriesDatasetCreator:
    """Creates time series sequences with disaster labels"""
    
    def __init__(
        self,
        sequence_hours: int = 168,  # 7 days lookback
        feature_columns: Optional[List[str]] = None
    ):
        """
        Initialize dataset creator
        
        Args:
            sequence_hours: Number of hours in each sequence
            feature_columns: List of feature column names
        """
        self.sequence_hours = sequence_hours
        self.feature_columns = feature_columns
        logger.info(f"TimeSeriesDatasetCreator initialized with {sequence_hours}h sequences")
    
    def load_features(
        self,
        features_path: Path
    ) -> pd.DataFrame:
        """
        Load processed features from CSV
        
        Args:
            features_path: Path to features CSV
            
        Returns:
            DataFrame with features
        """
        try:
            df = pd.read_csv(features_path)
            
            # Convert timestamp to datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            logger.info(f"Loaded {len(df)} feature records from {features_path}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load features: {str(e)}")
            return pd.DataFrame()
    
    def load_disasters(
        self,
        disasters_path: Path
    ) -> pd.DataFrame:
        """
        Load disaster records
        
        Args:
            disasters_path: Path to disaster CSV
            
        Returns:
            DataFrame with disaster records
        """
        try:
            df = pd.read_csv(disasters_path)
            
            # Convert timestamps
            for col in ['disaster_timestamp', 'window_start', 'window_end']:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
            
            logger.info(f"Loaded {len(df)} disaster records from {disasters_path}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load disasters: {str(e)}")
            return pd.DataFrame()
    
    def calculate_stress_label(
        self,
        hours_before_disaster: float,
        pressure_drop_6h: float = 0,
        pressure_drop_24h: float = 0
    ) -> float:
        """
        Calculate stress label based on time before disaster
        
        Args:
            hours_before_disaster: Hours until disaster
            pressure_drop_6h: Pressure drop in last 6 hours
            pressure_drop_24h: Pressure drop in last 24 hours
            
        Returns:
            Stress score (0.0-1.0)
        """
        if hours_before_disaster < 0:  # After disaster
            base_stress = 1.0
        elif hours_before_disaster <= 24:  # 0-24h before
            base_stress = 0.7 + (0.3 * (24 - hours_before_disaster) / 24)
        elif hours_before_disaster <= 48:  # 24-48h before
            base_stress = 0.5 + (0.2 * (48 - hours_before_disaster) / 24)
        elif hours_before_disaster <= 72:  # 48-72h before
            base_stress = 0.3 + (0.2 * (72 - hours_before_disaster) / 24)
        elif hours_before_disaster <= 168:  # 3-7 days before
            base_stress = 0.1 + (0.2 * (168 - hours_before_disaster) / 96)
        else:  # Normal day
            base_stress = 0.1
        
        # Weather-based adjustments
        weather_adjustment = 0.0
        if pressure_drop_6h < -3:  # Rapid pressure drop
            weather_adjustment += 0.1
        if pressure_drop_24h < -5:  # Very rapid pressure drop
            weather_adjustment += 0.2
        
        # Final stress (clamped to 0-1)
        stress = min(1.0, base_stress + weather_adjustment)
        
        return stress
    
    def create_sequences_for_disaster(
        self,
        features_df: pd.DataFrame,
        disaster_row: pd.Series,
        location_threshold_km: float = 100
    ) -> List[Tuple[np.ndarray, float, Dict]]:
        """
        Create training sequences for a single disaster event
        
        Args:
            features_df: DataFrame with features
            disaster_row: Single disaster record
            location_threshold_km: Maximum distance from disaster
            
        Returns:
            List of (sequence, label, metadata) tuples
        """
        sequences = []
        
        disaster_time = disaster_row['disaster_timestamp']
        disaster_lat = disaster_row['latitude']
        disaster_lng = disaster_row['longitude']
        window_start = disaster_row['window_start']
        
        # Filter features by location and time
        # Rough distance filter (1 degree ≈ 111 km)
        lat_delta = location_threshold_km / 111.0
        lng_delta = location_threshold_km / 111.0
        
        location_mask = (
            (features_df['latitude'] >= disaster_lat - lat_delta) &
            (features_df['latitude'] <= disaster_lat + lat_delta) &
            (features_df['longitude'] >= disaster_lng - lng_delta) &
            (features_df['longitude'] <= disaster_lng + lng_delta)
        )
        
        time_mask = (
            (features_df['timestamp'] >= window_start) &
            (features_df['timestamp'] <= disaster_time)
        )
        
        relevant_features = features_df[location_mask & time_mask].sort_values('timestamp')
        
        if len(relevant_features) < self.sequence_hours:
            logger.warning(f"Not enough data for disaster at {disaster_time}")
            return sequences
        
        # Create sliding window sequences
        for i in range(len(relevant_features) - self.sequence_hours + 1):
            sequence_data = relevant_features.iloc[i:i + self.sequence_hours]
            
            # Get feature columns
            if self.feature_columns:
                feature_cols = [c for c in self.feature_columns if c in sequence_data.columns]
            else:
                # Exclude non-feature columns
                exclude_cols = ['timestamp', 'latitude', 'longitude', 'location', 'disaster_id']
                feature_cols = [c for c in sequence_data.columns if c not in exclude_cols]
            
            # Extract features as numpy array
            X = sequence_data[feature_cols].values
            
            # Calculate label (stress at end of sequence)
            end_time = sequence_data.iloc[-1]['timestamp']
            hours_before = (disaster_time - end_time).total_seconds() / 3600
            
            # Get weather features for adjustment
            pressure_drop_6h = sequence_data.iloc[-1].get('pressure_drop_6h', 0)
            pressure_drop_24h = sequence_data.iloc[-1].get('pressure_drop_24h', 0)
            
            y = self.calculate_stress_label(
                hours_before,
                pressure_drop_6h,
                pressure_drop_24h
            )
            
            # Metadata
            metadata = {
                'disaster_id': disaster_row['disaster_id'],
                'disaster_type': disaster_row['disaster_type'],
                'disaster_time': disaster_time,
                'sequence_end': end_time,
                'hours_before_disaster': hours_before,
                'magnitude': disaster_row.get('magnitude', 0),
                'location': disaster_row.get('place', 'Unknown')
            }
            
            sequences.append((X, y, metadata))
        
        return sequences
    
    def create_normal_sequences(
        self,
        features_df: pd.DataFrame,
        disasters_df: pd.DataFrame,
        num_samples: int = 1000,
        min_distance_hours: int = 240  # 10 days from any disaster
    ) -> List[Tuple[np.ndarray, float, Dict]]:
        """
        Create sequences from normal (non-disaster) periods
        
        Args:
            features_df: DataFrame with features
            disasters_df: DataFrame with disasters (to avoid)
            num_samples: Number of normal samples to create
            min_distance_hours: Minimum hours from any disaster
            
        Returns:
            List of (sequence, label, metadata) tuples
        """
        sequences = []
        
        # Sort features by time
        features_df = features_df.sort_values('timestamp').reset_index(drop=True)
        
        # Create mask for normal periods (far from disasters)
        normal_mask = np.ones(len(features_df), dtype=bool)
        
        for _, disaster in disasters_df.iterrows():
            disaster_time = disaster['disaster_timestamp']
            min_delta = timedelta(hours=min_distance_hours)
            
            time_mask = (
                (features_df['timestamp'] >= disaster_time - min_delta) &
                (features_df['timestamp'] <= disaster_time + min_delta)
            )
            normal_mask &= ~time_mask
        
        normal_features = features_df[normal_mask]
        
        if len(normal_features) < self.sequence_hours:
            logger.warning("Not enough normal data")
            return sequences
        
        # Randomly sample starting points
        max_start = len(normal_features) - self.sequence_hours
        if max_start <= 0:
            return sequences
        
        sample_starts = np.random.choice(max_start, min(num_samples, max_start), replace=False)
        
        # Get feature columns
        if self.feature_columns:
            feature_cols = [c for c in self.feature_columns if c in normal_features.columns]
        else:
            exclude_cols = ['timestamp', 'latitude', 'longitude', 'location', 'disaster_id']
            feature_cols = [c for c in normal_features.columns if c not in exclude_cols]
        
        for start_idx in sample_starts:
            sequence_data = normal_features.iloc[start_idx:start_idx + self.sequence_hours]
            
            X = sequence_data[feature_cols].values
            y = 0.1  # Baseline stress for normal periods
            
            metadata = {
                'disaster_id': 'normal',
                'disaster_type': 'none',
                'sequence_end': sequence_data.iloc[-1]['timestamp'],
                'hours_before_disaster': float('inf'),
                'location': 'normal_period'
            }
            
            sequences.append((X, y, metadata))
        
        return sequences
    
    def create_full_dataset(
        self,
        features_path: Path,
        disasters_path: Path,
        output_dir: Path,
        normal_ratio: float = 0.5
    ) -> Dict[str, Path]:
        """
        Create complete training dataset
        
        Args:
            features_path: Path to features CSV
            disasters_path: Path to disasters CSV
            output_dir: Output directory
            normal_ratio: Ratio of normal to disaster samples
            
        Returns:
            Dictionary with paths to created files
        """
        logger.info("Creating full time series dataset...")
        
        # Load data
        features_df = self.load_features(features_path)
        disasters_df = self.load_disasters(disasters_path)
        
        if features_df.empty or disasters_df.empty:
            logger.error("Cannot create dataset with empty data")
            return {}
        
        # Create sequences for each disaster
        disaster_sequences = []
        logger.info(f"Creating sequences for {len(disasters_df)} disasters...")
        
        for idx, disaster_row in tqdm(disasters_df.iterrows(), total=len(disasters_df)):
            seqs = self.create_sequences_for_disaster(features_df, disaster_row)
            disaster_sequences.extend(seqs)
        
        logger.info(f"Created {len(disaster_sequences)} disaster sequences")
        
        # Create normal sequences
        num_normal = int(len(disaster_sequences) * normal_ratio)
        logger.info(f"Creating {num_normal} normal sequences...")
        normal_sequences = self.create_normal_sequences(features_df, disasters_df, num_normal)
        
        logger.info(f"Created {len(normal_sequences)} normal sequences")
        
        # Combine all sequences
        all_sequences = disaster_sequences + normal_sequences
        
        # Shuffle
        np.random.shuffle(all_sequences)
        
        # Split into train/val/test
        n_total = len(all_sequences)
        n_train = int(0.7 * n_total)
        n_val = int(0.15 * n_total)
        
        train_seqs = all_sequences[:n_train]
        val_seqs = all_sequences[n_train:n_train + n_val]
        test_seqs = all_sequences[n_train + n_val:]
        
        # Save datasets
        output_dir.mkdir(parents=True, exist_ok=True)
        
        paths = {}
        for split_name, seqs in [('train', train_seqs), ('val', val_seqs), ('test', test_seqs)]:
            X_list = [s[0] for s in seqs]
            y_list = [s[1] for s in seqs]
            meta_list = [s[2] for s in seqs]
            
            # Convert to arrays
            X = np.array(X_list)  # Shape: (n_samples, sequence_hours, n_features)
            y = np.array(y_list)  # Shape: (n_samples,)
            
            # Save as compressed numpy files
            output_file = output_dir / f"{split_name}_dataset.npz"
            np.savez_compressed(
                output_file,
                X=X,
                y=y,
                metadata=meta_list
            )
            
            paths[split_name] = output_file
            logger.info(f"Saved {split_name}: {len(seqs)} samples, shape {X.shape} -> {output_file}")
        
        # Save metadata
        metadata_df = pd.DataFrame([s[2] for s in all_sequences])
        metadata_path = output_dir / "dataset_metadata.csv"
        metadata_df.to_csv(metadata_path, index=False)
        paths['metadata'] = metadata_path
        
        logger.info(f"Dataset creation complete! Total: {n_total} samples")
        logger.info(f"Train: {len(train_seqs)}, Val: {len(val_seqs)}, Test: {len(test_seqs)}")
        
        return paths


def create_dataset_from_raw_data(
    features_path: Path,
    disasters_path: Path,
    output_dir: Path
) -> Dict[str, Path]:
    """
    Convenience function to create dataset
    
    Args:
        features_path: Path to features CSV
        disasters_path: Path to disasters CSV
        output_dir: Output directory
        
    Returns:
        Dictionary with paths to dataset files
    """
    creator = TimeSeriesDatasetCreator(sequence_hours=168)
    return creator.create_full_dataset(features_path, disasters_path, output_dir)
