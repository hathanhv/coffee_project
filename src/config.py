"""
Configuration file for Coffee Project

Contains all project paths and settings.
"""

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CLEANED_DATA_DIR = PROCESSED_DATA_DIR / "cleaned"
DICTIONARY_DIR = DATA_DIR / "dictionary"

# Raw data files
RAW_FILES = {
    'sa_var': RAW_DATA_DIR / 'SA#var.csv',
    'needstate': RAW_DATA_DIR / 'NeedstateDaydaypart.csv',
    'brandhealth': RAW_DATA_DIR / 'Brandhealth.csv',
    'brand_image': RAW_DATA_DIR / 'Brand_Image.csv',
    'segmentation': RAW_DATA_DIR / '2017Segmenttation3685case.csv'
}

# Processed data files
MERGED_DATA_FILE = PROCESSED_DATA_DIR / "merged_full.csv"

# Cleaned data files
CLEANED_FILES = {
    'customer_seg': CLEANED_DATA_DIR / 'customer_segmentation_clean.csv',
    'brand_image': CLEANED_DATA_DIR / 'brand_image_clean.csv',
    'brandhealth': CLEANED_DATA_DIR / 'brandhealth_clean.csv',
    'sa_var': CLEANED_DATA_DIR / 'sa_var_clean.csv',
    'needstate': CLEANED_DATA_DIR / 'needstate_clean.csv'
}

# Models directory
MODELS_DIR = PROJECT_ROOT / "results"

# Reports directory
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
SLIDES_DIR = REPORTS_DIR / "slides"

# Model settings
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Logging configuration
LOG_LEVEL = "INFO"
