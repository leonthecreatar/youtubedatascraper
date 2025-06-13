import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
import logging
import re
import concurrent.futures
import numpy as np
from tqdm import tqdm
from youtube_api import YouTubeDataScraper
import time

class YouTubeDataProcessor:
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the data processor with configuration."""
        self.config = self._load_config(config_path)
        self.youtube_api = YouTubeDataScraper(config_path)
        self._setup_directories()
        # Set up logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        self._max_workers = min(32, (os.cpu_count() or 1) + 4)  # Optimal number of workers
        self._chunk_size = 1000  # Process videos in chunks of 1000

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _setup_directories(self):
        """Create necessary directories for data storage."""
        directories = [
            self.config['storage']['raw_data_dir'],
            self.config['storage']['processed_data_dir']
        ]
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    def save_raw_data(self, channel_id: str, data: Dict):
        """Save raw channel data to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{channel_id}_{timestamp}.json"
        filepath = os.path.join(self.config['storage']['raw_data_dir'], filename)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def save_processed_data(self, data: pd.DataFrame, filename: str):
        """Save processed data to CSV file."""
        filepath = os.path.join(self.config['storage']['processed_data_dir'], filename)
        data.to_csv(filepath, index=False)

    def process_channel_data(self, channel_data: Dict) -> pd.DataFrame:
        """Process raw channel data into a structured DataFrame."""
        channel_info = channel_data['channel_info']
        metrics = channel_data['metrics']

        # Create a row for each time period
        rows = []
        for period, period_metrics in metrics.items():
            row = {
                'channel_id': channel_info['channel_id'],
                'channel_name': channel_info['title'],
                'period': period,
                'subscriber_count': channel_info['subscriber_count'],
                'total_videos': channel_info['video_count'],
                'total_views': channel_info['view_count'],
                **period_metrics
            }
            rows.append(row)

        return pd.DataFrame(rows)

    def _parse_duration(self, duration: str) -> float:
        """Convert ISO 8601 duration format to minutes."""
        try:
            # Debug log the input duration
            self.logger.debug(f"Parsing duration: {duration}")
            
            if not duration or duration == 'PT0S':  # Handle empty or 0 duration case
                self.logger.debug("Empty or zero duration")
                return 0.0
                
            # Extract hours, minutes, and seconds using regex
            hours_match = re.search(r'(\d+)H', duration)
            minutes_match = re.search(r'(\d+)M', duration)
            seconds_match = re.search(r'(\d+)S', duration)
            
            hours = int(hours_match.group(1)) if hours_match else 0
            minutes = int(minutes_match.group(1)) if minutes_match else 0
            seconds = int(seconds_match.group(1)) if seconds_match else 0
            
            total_minutes = hours * 60 + minutes + seconds / 60
            
            # Debug log the parsed values
            self.logger.debug(f"Parsed duration - Hours: {hours}, Minutes: {minutes}, Seconds: {seconds}, Total Minutes: {total_minutes}")
            
            return total_minutes
            
        except Exception as e:
            self.logger.error(f"Error parsing duration '{duration}': {str(e)}")
            return 0.0

    def generate_analysis_report(self, channel_ids: List[str]) -> None:
        """
        Generate a comprehensive analysis report for the given channel IDs using parallel processing.
        
        Args:
            channel_ids: List of YouTube channel IDs to analyze
        """
        try:
            self.logger.info(f"Starting analysis report generation for {len(channel_ids)} channels")
            
            # Get all channel data in one batch
            channels_data = self.youtube_api.get_channel_info(channel_ids)
            if not channels_data:
                self.logger.error("No channel data retrieved")
                return
            
            # Get all video data in one batch
            videos_data = self.youtube_api.get_channel_videos(channel_ids)
            if not videos_data:
                self.logger.error("No video data retrieved")
                return
            
            # Process channel data
            channel_df = self.process_comprehensive_channel_data(channels_data, videos_data)
            if channel_df.empty:
                self.logger.error("No channel data was successfully processed")
                return
            
            # Process video data for all channels
            all_videos_data = []
            video_info = {}  # Dictionary to store video metadata for comment processing
            
            for channel_id, channel_videos in videos_data.items():
                if 'videos' in channel_videos and channel_videos['videos']:
                    video_df = self.process_comprehensive_video_data(channel_videos)
                    if not video_df.empty:
                        all_videos_data.append(video_df)
                        # Store video metadata for comment processing
                        for _, row in video_df.iterrows():
                            video_info[row['id']] = {
                                'title': row['title'],
                                'channel_title': row['channel_title'],
                                'channel_id': row['channel_id']
                            }
            
            # Save combined data files
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = self.config['storage']['processed_data_dir']
            
            # Save channel data
            channel_df.to_csv(
                os.path.join(output_dir, f"channel_data_{timestamp}.csv"),
                index=False,
                encoding='utf-8'
            )
            self.logger.info(f"\nChannel data saved to: {os.path.join(output_dir, f'channel_data_{timestamp}.csv')}")
            
            # Process and save video data in chunks
            if all_videos_data:
                combined_videos_df = pd.DataFrame()
                chunk_size = self._chunk_size
                total_videos = sum(len(df) for df in all_videos_data)
                
                # Process each chunk
                for i in range(0, total_videos, chunk_size):
                    chunk_dfs = []
                    remaining = chunk_size
                    
                    # Get chunks from each video DataFrame
                    for df in all_videos_data:
                        if len(df) == 0:
                            continue
                        if remaining <= 0:
                            break
                        chunk_size_this_df = min(remaining, len(df))
                        chunk_dfs.append(df.iloc[:chunk_size_this_df])
                        df.drop(df.index[:chunk_size_this_df], inplace=True)
                        remaining -= chunk_size_this_df
                    
                    if chunk_dfs:
                        # Combine chunks and optimize memory
                        chunk_df = pd.concat(chunk_dfs, ignore_index=True)
                        chunk_df = self._optimize_dataframe_memory(chunk_df)
                        
                        # Append to main DataFrame
                        if combined_videos_df.empty:
                            combined_videos_df = chunk_df
                        else:
                            combined_videos_df = pd.concat([combined_videos_df, chunk_df], ignore_index=True)
                        
                        # Save chunk to CSV
                        mode = 'w' if i == 0 else 'a'
                        header = i == 0
                        chunk_df.to_csv(
                            os.path.join(output_dir, f"combined_video_data_{timestamp}.csv"),
                            mode=mode,
                            header=header,
                            index=False,
                            encoding='utf-8'
                        )
                        
                        # Clear memory
                        del chunk_df
                        del chunk_dfs
                
                self.logger.info(f"\nCombined video data saved to: {os.path.join(output_dir, f'combined_video_data_{timestamp}.csv')}")
                
                # Log combined video statistics
                self._log_video_statistics(combined_videos_df)
                
                # Process comments for all videos
                video_ids = list(video_info.keys())
                self.process_video_comments(video_ids, video_info)
            
            self.logger.info("\nAnalysis report generation completed")
            
        except Exception as e:
            self.logger.error(f"Error generating analysis report: {str(e)}")
            raise

    def process_comprehensive_channel_data(self, channels_data: List[Dict], videos_data: Optional[Dict[str, dict]] = None) -> pd.DataFrame:
        """
        Process comprehensive channel data into a structured DataFrame.
        
        Args:
            channels_data: List of raw channel data from YouTube API
            videos_data: Optional dictionary mapping channel IDs to their video data
            
        Returns:
            pd.DataFrame: Processed channel data
        """
        try:
            all_channels_info = []
            
            for channel_data in channels_data:
                channel_id = channel_data['id']
                
                # Extract all relevant fields
                channel_info = {
                    'channel_id': channel_data['id'],
                    'etag': channel_data['etag'],
                    
                    # Snippet information
                    'title': channel_data['snippet']['title'],
                    'description': channel_data['snippet']['description'],
                    'custom_url': channel_data['snippet'].get('customUrl', ''),
                    'published_at': channel_data['snippet']['publishedAt'],
                    'country': channel_data['snippet'].get('country', ''),
                    'default_language': channel_data['snippet'].get('defaultLanguage', ''),
                    
                    # Content Details
                    'uploads_playlist_id': channel_data['contentDetails']['relatedPlaylists'].get('uploads', ''),
                    'likes_playlist_id': channel_data['contentDetails']['relatedPlaylists'].get('likes', ''),
                    'favorites_playlist_id': channel_data['contentDetails']['relatedPlaylists'].get('favorites', ''),
                    
                    # Statistics
                    'view_count': int(channel_data['statistics'].get('viewCount', 0)),
                    'subscriber_count': int(channel_data['statistics'].get('subscriberCount', 0)),
                    'hidden_subscriber_count': channel_data['statistics'].get('hiddenSubscriberCount', False),
                    'video_count': int(channel_data['statistics'].get('videoCount', 0)),
                    
                    # Topic Details
                    'topic_ids': ','.join(channel_data.get('topicDetails', {}).get('topicIds', [])),
                    'topic_categories': ','.join(channel_data.get('topicDetails', {}).get('topicCategories', [])),
                    
                    # Branding Settings
                    'branding_title': channel_data.get('brandingSettings', {}).get('channel', {}).get('title', ''),
                    'branding_description': channel_data.get('brandingSettings', {}).get('channel', {}).get('description', ''),
                    'branding_keywords': channel_data.get('brandingSettings', {}).get('channel', {}).get('keywords', ''),
                    'branding_tracking_analytics_account_id': channel_data.get('brandingSettings', {}).get('channel', {}).get('trackingAnalyticsAccountId', ''),
                    'branding_unsubscribed_trailer': channel_data.get('brandingSettings', {}).get('channel', {}).get('unsubscribedTrailer', ''),
                    'branding_default_language': channel_data.get('brandingSettings', {}).get('channel', {}).get('defaultLanguage', ''),
                    'branding_country': channel_data.get('brandingSettings', {}).get('channel', {}).get('country', ''),
                    
                    # Status
                    'privacy_status': channel_data.get('status', {}).get('privacyStatus', ''),
                    'is_linked': channel_data.get('status', {}).get('isLinked', False),
                    'long_uploads_status': channel_data.get('status', {}).get('longUploadsStatus', ''),
                    'made_for_kids': channel_data.get('status', {}).get('madeForKids', False),
                    'self_declared_made_for_kids': channel_data.get('status', {}).get('selfDeclaredMadeForKids', False),
                    
                    # Content Owner Details
                    'content_owner': channel_data.get('contentOwnerDetails', {}).get('contentOwner', ''),
                    'time_linked': channel_data.get('contentOwnerDetails', {}).get('timeLinked', '')
                }
                
                # Calculate additional metrics if video data is available
                if videos_data and channel_id in videos_data and 'videos' in videos_data[channel_id]:
                    videos = videos_data[channel_id]['videos']
                    video_df = pd.DataFrame(videos)
                    
                    # Calculate video metrics
                    channel_info.update({
                        # View metrics
                        'avg_views_per_video': video_df['view_count'].mean(),
                        'median_views_per_video': video_df['view_count'].median(),
                        'highest_viewed_video': video_df['view_count'].max(),
                        'highest_viewed_video_title': video_df.loc[video_df['view_count'].idxmax(), 'title'] if not video_df.empty else '',
                        
                        # Engagement metrics
                        'avg_engagement_rate_per_view': (
                            (video_df['like_count'] + video_df['comment_count']).sum() / 
                            video_df['view_count'].sum() * 100 if video_df['view_count'].sum() > 0 else 0
                        ),
                        'avg_engagement_rate_per_subscriber': (
                            (video_df['like_count'] + video_df['comment_count']).sum() / 
                            channel_info['subscriber_count'] * 100 if channel_info['subscriber_count'] > 0 else 0
                        ),
                        
                        # Content metrics
                        'avg_video_duration': video_df['duration_seconds'].mean() / 60,  # Convert to minutes
                        'median_video_duration': video_df['duration_seconds'].median() / 60,  # Convert to minutes
                        
                        # Upload frequency
                        'upload_frequency': self._calculate_upload_frequency(video_df)
                    })
                else:
                    # Set default values if no video data
                    channel_info.update({
                        'avg_views_per_video': 0,
                        'median_views_per_video': 0,
                        'highest_viewed_video': 0,
                        'highest_viewed_video_title': '',
                        'avg_engagement_rate_per_view': 0,
                        'avg_engagement_rate_per_subscriber': 0,
                        'avg_video_duration': 0,
                        'median_video_duration': 0,
                        'upload_frequency': 0
                    })
                
                all_channels_info.append(channel_info)
            
            # Convert to DataFrame
            df = pd.DataFrame(all_channels_info)
            
            # Log some basic statistics
            self.logger.info("\nChannel Data Statistics:")
            self.logger.info(f"Total channels processed: {len(df)}")
            self.logger.info(f"Total subscribers across all channels: {df['subscriber_count'].sum():,}")
            self.logger.info(f"Total views across all channels: {df['view_count'].sum():,}")
            self.logger.info(f"Average subscribers per channel: {df['subscriber_count'].mean():,.0f}")
            self.logger.info(f"Average views per channel: {df['view_count'].mean():,.0f}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error processing comprehensive channel data: {str(e)}")
            raise

    def _calculate_upload_frequency(self, video_df: pd.DataFrame) -> float:
        """
        Calculate the average number of days between video uploads.
        
        Args:
            video_df: DataFrame containing video data with 'published_at' column
            
        Returns:
            float: Average days between uploads
        """
        try:
            if len(video_df) < 2:
                return 0.0
                
            # Convert published_at to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(video_df['published_at']):
                video_df['published_at'] = pd.to_datetime(video_df['published_at'])
            
            # Sort by published date
            video_df = video_df.sort_values('published_at')
            
            # Calculate time differences between consecutive videos
            time_diffs = video_df['published_at'].diff().dropna()
            
            # Convert to days and calculate mean
            avg_days = time_diffs.dt.total_seconds().mean() / (24 * 3600)
            
            return avg_days
            
        except Exception as e:
            self.logger.error(f"Error calculating upload frequency: {str(e)}")
            return 0.0

    def process_comprehensive_video_data(self, videos_data: Dict) -> pd.DataFrame:
        """
        Process comprehensive video data into a structured DataFrame.
        
        Args:
            videos_data: Dictionary containing video data from YouTube API
                Expected keys: 'videos' (list of video data), 'categories' (dict of category mappings)
            
        Returns:
            pd.DataFrame: Processed video data
        """
        try:
            if not videos_data or 'videos' not in videos_data:
                self.logger.warning("No video data provided")
                return pd.DataFrame()
            
            videos = videos_data['videos']
            if not videos:
                self.logger.warning("Empty video list")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(videos)
            
            # Ensure all numeric columns are properly typed
            numeric_columns = [
                'view_count', 'like_count', 'dislike_count', 'favorite_count', 'comment_count',
                'duration_seconds', 'duration_ms', 'bitrate_bps', 'concurrent_viewers',
                'video_width', 'video_height', 'video_frame_rate', 'video_aspect_ratio',
                'video_bitrate', 'audio_channel_count', 'audio_bitrate', 'file_size'
            ]
            
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            # Ensure boolean columns are properly typed
            boolean_columns = [
                'licensed_content', 'has_custom_thumbnail', 'embeddable',
                'public_stats_viewable', 'made_for_kids'
            ]
            
            for col in boolean_columns:
                if col in df.columns:
                    df[col] = df[col].astype(bool)
            
            # Ensure datetime columns are properly typed
            datetime_columns = [
                'published_at', 'recording_date', 'creation_time',
                'actual_start_time', 'actual_end_time',
                'scheduled_start_time', 'scheduled_end_time'
            ]
            
            for col in datetime_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            
            # Log some basic statistics
            self.logger.info("\nVideo Data Statistics:")
            self.logger.info(f"Total videos processed: {len(df)}")
            self.logger.info(f"Date range: {df['published_at'].min()} to {df['published_at'].max()}")
            self.logger.info(f"Total views: {df['view_count'].sum():,}")
            self.logger.info(f"Average views per video: {df['view_count'].mean():,.0f}")
            self.logger.info(f"Average duration: {df['duration_seconds'].mean()/60:.1f} minutes")
            
            # Log category distribution
            if 'category_name' in df.columns:
                category_counts = df['category_name'].value_counts()
                self.logger.info("\nCategory Distribution:")
                for category, count in category_counts.items():
                    self.logger.info(f"{category}: {count} videos")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error processing comprehensive video data: {str(e)}")
            raise

    def _optimize_dataframe_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage by using appropriate data types."""
        for col in df.columns:
            # Optimize numeric columns
            if pd.api.types.is_integer_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], downcast='integer')
            elif pd.api.types.is_float_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], downcast='float')
            
            # Optimize string columns
            elif pd.api.types.is_object_dtype(df[col]):
                if df[col].nunique() / len(df) < 0.5:  # If less than 50% unique values
                    df[col] = df[col].astype('category')
        
        return df

    def _log_channel_statistics(self, df: pd.DataFrame) -> None:
        """Log channel statistics."""
        self.logger.info("\nCombined Channel Data Statistics:")
        self.logger.info(f"Total channels: {len(df)}")
        self.logger.info(f"Total subscribers: {df['subscriber_count'].sum():,}")
        self.logger.info(f"Total views: {df['view_count'].sum():,}")
        self.logger.info(f"Average subscribers per channel: {df['subscriber_count'].mean():,.0f}")
        self.logger.info(f"Average views per channel: {df['view_count'].mean():,.0f}")

    def _log_video_statistics(self, df: pd.DataFrame) -> None:
        """Log video statistics."""
        self.logger.info("\nCombined Video Data Statistics:")
        self.logger.info(f"Total videos: {len(df)}")
        self.logger.info(f"Date range: {df['published_at'].min()} to {df['published_at'].max()}")
        self.logger.info(f"Total views: {df['view_count'].sum():,}")
        self.logger.info(f"Average views per video: {df['view_count'].mean():,.0f}")
        self.logger.info(f"Average duration: {df['duration_seconds'].mean()/60:.1f} minutes")
        
        if 'category_name' in df.columns:
            category_counts = df['category_name'].value_counts()
            self.logger.info("\nCategory Distribution:")
            for category, count in category_counts.items():
                self.logger.info(f"{category}: {count} videos")

    def process_video_comments(self, video_ids: List[str], video_info: Dict[str, Dict]) -> None:
        """
        Process and save comments for a list of videos.
        
        Args:
            video_ids: List of video IDs to fetch comments for
            video_info: Dictionary mapping video IDs to their metadata (title, channel info, etc.)
        """
        try:
            self.logger.info(f"Starting comment processing for {len(video_ids)} videos")
            
            # Create a list to store all comment data
            all_comments_data = []
            
            # Process all videos at once
            for video_id in tqdm(video_ids, desc="Fetching comments"):
                if video_id not in video_info:
                    self.logger.warning(f"No metadata found for video {video_id}, skipping")
                    continue
                    
                # Get video metadata
                video_meta = video_info[video_id]
                
                # Fetch comments
                comments = self.youtube_api.get_video_comments(video_id)
                
                if comments:
                    # Combine all comment texts for this video
                    comment_texts = [comment['text'] for comment in comments]
                    combined_comments = ' | '.join(comment_texts)
                    
                    # Create a row with video info and combined comments
                    comment_row = {
                        'video_title': video_meta['title'],
                        'video_id': video_id,
                        'channel_title': video_meta['channel_title'],
                        'channel_id': video_meta['channel_id'],
                        'comments': combined_comments,
                        'comment_count': len(comments)
                    }
                    all_comments_data.append(comment_row)
            
            if all_comments_data:
                # Convert to DataFrame
                comments_df = pd.DataFrame(all_comments_data)
                
                # Save to CSV
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(
                    self.config['storage']['processed_data_dir'],
                    f"video_comments_{timestamp}.csv"
                )
                
                comments_df.to_csv(output_path, index=False, encoding='utf-8')
                self.logger.info(f"\nComments data saved to: {output_path}")
                self.logger.info(f"Total videos processed: {len(comments_df)}")
                self.logger.info(f"Total comments collected: {comments_df['comment_count'].sum():,}")
            else:
                self.logger.warning("No comments were collected for any videos")
                
        except Exception as e:
            self.logger.error(f"Error processing video comments: {str(e)}")
            raise

if __name__ == "__main__":
    # Example usage
    processor = YouTubeDataProcessor()
    # Process your data here 