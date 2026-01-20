"""
Module 1 & 2: Data Collection, Cleaning and Preprocessing
Week 1-2 Implementation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """
    Handles data loading, cleaning, and preprocessing for energy consumption analysis
    """
    
    def __init__(self, data_path):
        """
        Initialize preprocessor with dataset path
        
        Args:
            data_path: Path to the CSV dataset
        """
        self.data_path = data_path
        self.df = None
        self.df_clean = None
        
    def load_data(self):
        """Load dataset from CSV file"""
        print("[INFO] Loading dataset...")
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"[OK] Dataset loaded successfully!")
            print(f"  Shape: {self.df.shape}")
            print(f"  Columns: {len(self.df.columns)}")
            print(f"  Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            return self.df
        except Exception as e:
            print(f"[ERROR] Error loading dataset: {e}")
            return None
    
    def explore_data(self):
        """Perform exploratory data analysis"""
        if self.df is None:
            print("[ERROR] Please load data first!")
            return
        
        print("\n" + "="*60)
        print(" EXPLORATORY DATA ANALYSIS")
        print("="*60)
        
        # Basic info
        print("\n1. Dataset Information:")
        print(f"   - Total Records: {len(self.df):,}")
        print(f"   - Total Features: {len(self.df.columns)}")
        print(f"   - Date Range: {self.df['time'].min()} to {self.df['time'].max()}")
        
        # Data types
        print("\n2. Data Types:")
        print(self.df.dtypes.value_counts())
        
        # Missing values
        print("\n3. Missing Values:")
        missing = self.df.isnull().sum()
        if missing.sum() == 0:
            print("   [OK] No missing values found!")
        else:
            print(missing[missing > 0])
        
        # Statistical summary
        print("\n4. Statistical Summary:")
        print(self.df.describe().T)
        
        return self.df.describe()
    
    def clean_data(self):
        """Clean and preprocess the dataset"""
        if self.df is None:
            print("[ERROR] Please load data first!")
            return
        
        print("\n" + "="*60)
        print("[INFO] DATA CLEANING")
        print("="*60)
        
        # Create a copy
        self.df_clean = self.df.copy()
        
        # 1. Convert time column to datetime
        print("\n1. Converting timestamps...")
        self.df_clean['time'] = pd.to_datetime(self.df_clean['time'])
        self.df_clean.set_index('time', inplace=True)
        print("   [OK] Timestamps converted and set as index")
        
        # 2. Handle missing values
        print("\n2. Handling missing values...")
        missing_before = self.df_clean.isnull().sum().sum()
        
        # Forward fill for small gaps
        self.df_clean.fillna(method='ffill', limit=5, inplace=True)
        # Backward fill for remaining
        self.df_clean.fillna(method='bfill', limit=5, inplace=True)
        # Fill any remaining with 0 (assuming device was off)
        self.df_clean.fillna(0, inplace=True)
        
        missing_after = self.df_clean.isnull().sum().sum()
        print(f"   [OK] Missing values: {missing_before} -> {missing_after}")
        
        # 3. Remove duplicates
        print("\n3. Removing duplicates...")
        duplicates_before = self.df_clean.duplicated().sum()
        self.df_clean.drop_duplicates(inplace=True)
        duplicates_after = self.df_clean.duplicated().sum()
        print(f"   [OK] Duplicates: {duplicates_before} -> {duplicates_after}")
        
        # 4. Handle outliers (using IQR method)
        print("\n4. Handling outliers...")
        numeric_cols = self.df_clean.select_dtypes(include=[np.number]).columns
        outliers_removed = 0
        
        for col in numeric_cols:
            Q1 = self.df_clean[col].quantile(0.25)
            Q3 = self.df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier bounds
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            # Cap outliers instead of removing
            outliers_count = ((self.df_clean[col] < lower_bound) | 
                            (self.df_clean[col] > upper_bound)).sum()
            
            self.df_clean[col] = self.df_clean[col].clip(lower=lower_bound, 
                                                          upper=upper_bound)
            outliers_removed += outliers_count
        
        print(f"   [OK] Outliers capped: {outliers_removed}")
        
        # 5. Drop unnecessary columns
        print("\n5. Dropping unnecessary columns...")
        cols_to_drop = ['Unnamed: 0'] if 'Unnamed: 0' in self.df_clean.columns else []
        if cols_to_drop:
            self.df_clean.drop(columns=cols_to_drop, inplace=True)
            print(f"   [OK] Dropped columns: {cols_to_drop}")
        
        print(f"\n[OK] Data cleaning complete!")
        print(f"  Final shape: {self.df_clean.shape}")
        
        return self.df_clean
    
    def resample_data(self, freq='H'):
        """
        Resample data to different time frequencies
        
        Args:
            freq: Resampling frequency ('H'=hourly, 'D'=daily, 'W'=weekly, 'M'=monthly)
        """
        if self.df_clean is None:
            print("[ERROR] Please clean data first!")
            return
        
        print(f"\n[INFO] Resampling data to {freq} frequency...")
        
        # Resample numeric columns
        df_resampled = self.df_clean.select_dtypes(include=[np.number]).resample(freq).mean()
        
        print(f"   [OK] Resampled shape: {df_resampled.shape}")
        
        return df_resampled
    
    def create_aggregations(self):
        """Create useful aggregations for analysis"""
        if self.df_clean is None:
            print("[ERROR] Please clean data first!")
            return
        
        print("\n[INFO] Creating aggregations...")
        
        aggregations = {}
        
        # Hourly aggregation
        aggregations['hourly'] = self.resample_data('H')
        print("   [OK] Hourly aggregation created")
        
        # Daily aggregation
        aggregations['daily'] = self.resample_data('D')
        print("   [OK] Daily aggregation created")
        
        # Weekly aggregation
        aggregations['weekly'] = self.resample_data('W')
        print("   [OK] Weekly aggregation created")
        
        # Monthly aggregation
        aggregations['monthly'] = self.resample_data('M')
        print("   [OK] Monthly aggregation created")
        
        return aggregations
    
    def save_processed_data(self, output_path):
        """Save processed data to CSV"""
        if self.df_clean is None:
            print("[ERROR] No processed data to save!")
            return
        
        print(f"\n[INFO] Saving processed data to {output_path}...")
        self.df_clean.to_csv(output_path)
        print("   [OK] Data saved successfully!")
    
    def visualize_data_quality(self):
        """Visualize data quality metrics"""
        if self.df is None:
            print("[ERROR] Please load data first!")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Missing values heatmap
        axes[0, 0].set_title('Missing Values by Column')
        missing_data = self.df.isnull().sum()
        missing_data[missing_data > 0].plot(kind='bar', ax=axes[0, 0], color='coral')
        axes[0, 0].set_ylabel('Count')
        
        # Data types distribution
        axes[0, 1].set_title('Data Types Distribution')
        self.df.dtypes.value_counts().plot(kind='pie', ax=axes[0, 1], autopct='%1.1f%%')
        
        # Record count over time
        axes[1, 0].set_title('Records Over Time')
        self.df.set_index(pd.to_datetime(self.df['time'])).resample('D').size().plot(ax=axes[1, 0])
        axes[1, 0].set_ylabel('Records per Day')
        
        # Summary statistics
        axes[1, 1].set_title('Numeric Columns Summary')
        axes[1, 1].axis('off')
        summary_text = f"Total Records: {len(self.df):,}\n"
        summary_text += f"Numeric Columns: {len(self.df.select_dtypes(include=[np.number]).columns)}\n"
        summary_text += f"Categorical Columns: {len(self.df.select_dtypes(include=['object']).columns)}\n"
        summary_text += f"Missing Values: {self.df.isnull().sum().sum():,}\n"
        summary_text += f"Duplicates: {self.df.duplicated().sum():,}"
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=14, verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig('reports/figures/data_quality.png', dpi=100, bbox_inches='tight')
        print("[OK] Data quality visualization saved to reports/figures/data_quality.png")
        plt.show()


def main():
    """Main execution function"""
    print("="*60)
    print("SMART ENERGY ANALYSIS - DATA PREPROCESSING")
    print("="*60)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor('HomeC_augmented.csv')
    
    # Load data
    df = preprocessor.load_data()
    
    if df is not None:
        # Explore data
        preprocessor.explore_data()
        
        # Clean data
        df_clean = preprocessor.clean_data()
        
        # Create aggregations
        aggregations = preprocessor.create_aggregations()
        
        # Save processed data
        preprocessor.save_processed_data('data/processed/energy_data_clean.csv')
        
        # Visualize data quality
        # preprocessor.visualize_data_quality()
        
        print("\n" + "="*60)
        print("[OK] DATA PREPROCESSING COMPLETE!")
        print("="*60)
        
        return preprocessor, aggregations
    
    return None, None


if __name__ == "__main__":
    preprocessor, aggregations = main()
