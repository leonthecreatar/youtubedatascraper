import os
import sys
from pathlib import Path
from typing import List, Dict
import yaml
from tqdm import tqdm
import logging
from datetime import datetime
import pandas as pd

from youtube_api import YouTubeDataScraper
from data_processor import YouTubeDataProcessor

def setup_logging() -> logging.Logger:
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

# Set up logger
logger = setup_logging()

def load_channels(config_path: str) -> List[Dict]:
    """Load channel configurations from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            return config.get('channels', [])
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file: {e}")
        raise

def get_channel_id(scraper: YouTubeDataScraper, channel_config: Dict) -> str:
    """Get channel ID from either handle or ID in channel config."""
    if 'id' in channel_config:
        return channel_config['id']
    elif 'handle' in channel_config:
        channel_id = scraper.get_channel_id_from_handle(channel_config['handle'])
        if not channel_id:
            raise ValueError(f"Could not find channel ID for handle: {channel_config['handle']}")
        return channel_id
    else:
        raise ValueError("Channel config must contain either 'id' or 'handle'")

def main():
    """Main function to run the YouTube channel analysis."""
    logger = setup_logging()
    logger.info("Starting YouTube channel analysis")
    
    try:
        # Initialize components
        scraper = YouTubeDataScraper()
        processor = YouTubeDataProcessor()
        
        # Load channel configurations
        channels = load_channels("config/config.yaml")
        if not channels:
            logger.error("No channels configured for analysis")
            return
        
        # Extract channel IDs
        channel_ids = [channel['id'] for channel in channels]
        logger.info(f"Processing {len(channel_ids)} channels")
        
        # Generate analysis report using batch processing
        processor.generate_analysis_report(channel_ids)
        
        logger.info("Analysis complete")
        
    except Exception as e:
        logger.error(f"An error occurred during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main() 