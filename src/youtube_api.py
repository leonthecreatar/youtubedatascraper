from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import yaml
import os
from datetime import datetime, timedelta, timezone
import time
from typing import Dict, List, Optional, Union
import logging
from tqdm import tqdm

class YouTubeDataScraper:
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the YouTube Data Scraper with configuration."""
        self.config = self._load_config(config_path)
        self.youtube = self._build_youtube_client()
        self.logger = self._setup_logger()

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _build_youtube_client(self):
        """Build YouTube API client."""
        api_key = self.config['youtube_api']['api_key']
        if not api_key:
            raise ValueError("YouTube API key not found in config")
        return build('youtube', 'v3', developerKey=api_key)

    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger('YouTubeDataScraper')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def get_channel_info(self, channel_id: str) -> dict:
        """
        Get channel information using channel ID.
        
        Args:
            channel_id: YouTube channel ID
            
        Returns:
            dict: Channel information
        """
        try:
            request = self.youtube.channels().list(
                part="snippet,statistics,contentDetails",
                id=channel_id
            )
            response = request.execute()
            
            if not response.get('items'):
                raise ValueError(f"Channel not found: {channel_id}")
                
            return response['items'][0]
            
        except Exception as e:
            self.logger.error(f"Error getting channel info for {channel_id}: {str(e)}")
            raise

    def get_channel_videos(self, channel_id: str, max_results: int = 50) -> dict:
        """Get videos from a channel with their statistics."""
        try:
            # First get channel's uploads playlist ID
            channel_response = self.youtube.channels().list(
                part='contentDetails',
                id=channel_id
            ).execute()
            
            if not channel_response['items']:
                self.logger.warning(f"No channel found with ID: {channel_id}")
                return {}
            
            uploads_playlist_id = channel_response['items'][0]['contentDetails']['relatedPlaylists']['uploads']
            
            # Get videos from the uploads playlist
            videos = []
            next_page_token = None
            
            while len(videos) < max_results:
                playlist_response = self.youtube.playlistItems().list(
                    part='snippet,contentDetails',
                    playlistId=uploads_playlist_id,
                    maxResults=min(50, max_results - len(videos)),
                    pageToken=next_page_token
                ).execute()
                
                video_ids = [item['contentDetails']['videoId'] for item in playlist_response['items']]
                
                # Get detailed video statistics including share count
                video_response = self.youtube.videos().list(
                    part='snippet,statistics,contentDetails',
                    id=','.join(video_ids)
                ).execute()
                
                for video in video_response['items']:
                    video_data = {
                        'id': video['id'],
                        'snippet': video['snippet'],
                        'statistics': {
                            'viewCount': video['statistics'].get('viewCount', '0'),
                            'likeCount': video['statistics'].get('likeCount', '0'),
                            'commentCount': video['statistics'].get('commentCount', '0'),
                            'shareCount': video['statistics'].get('shareCount', '0'),  # Added share count
                        },
                        'duration': video['contentDetails'].get('duration', 'PT0S')
                    }
                    videos.append(video_data)
                
                next_page_token = playlist_response.get('nextPageToken')
                if not next_page_token:
                    break
            
            return {'videos': videos}
            
        except Exception as e:
            self.logger.error(f"Error fetching videos for channel {channel_id}: {str(e)}")
            return {}

    def get_video_details(self, video_id: str) -> Optional[Dict]:
        """Get detailed information about a specific video."""
        try:
            request = self.youtube.videos().list(
                part="snippet,statistics,contentDetails",
                id=video_id
            )
            response = request.execute()

            if not response['items']:
                return None

            video = response['items'][0]
            return {
                'video_id': video_id,
                'title': video['snippet']['title'],
                'description': video['snippet']['description'],
                'published_at': video['snippet']['publishedAt'],
                'view_count': int(video['statistics'].get('viewCount', 0)),
                'like_count': int(video['statistics'].get('likeCount', 0)),
                'comment_count': int(video['statistics'].get('commentCount', 0)),
                'duration': self._parse_duration(video['contentDetails']['duration']),
                'category_id': video['snippet']['categoryId']
            }
        except HttpError as e:
            self.logger.error(f"Error fetching video details: {e}")
            return None

    def _parse_duration(self, duration: str) -> int:
        """Convert ISO 8601 duration to seconds."""
        import re
        match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', duration)
        if not match:
            return 0
        
        hours, minutes, seconds = match.groups()
        hours = int(hours) if hours else 0
        minutes = int(minutes) if minutes else 0
        seconds = int(seconds) if seconds else 0
        
        return hours * 3600 + minutes * 60 + seconds

    def analyze_channel(self, channel_id: str) -> Dict:
        """Perform comprehensive channel analysis."""
        channel_info = self.get_channel_info(channel_id)
        if not channel_info:
            return {}

        # Get all videos
        videos_data = self.get_channel_videos(
            channel_id,
            max_results=100  # Adjust based on needs
        )
        
        if not videos_data or 'videos' not in videos_data:
            self.logger.warning(f"No videos found for channel {channel_id}")
            return {}

        videos = videos_data['videos']
        if not videos:
            self.logger.warning(f"Empty video list for channel {channel_id}")
            return {}

        # Calculate metrics for different time periods
        now = datetime.now(timezone.utc)  # Make timezone-aware
        six_months_ago = now - timedelta(days=180)

        # Sort videos by publish date (newest first)
        videos.sort(key=lambda x: x['snippet']['publishedAt'], reverse=True)

        # Split videos into time periods
        recent_videos = videos[:self.config['analysis']['recent_videos_count']]
        six_month_videos = [
            v for v in videos 
            if datetime.fromisoformat(v['snippet']['publishedAt'].replace('Z', '+00:00')) > six_months_ago
        ]
        all_time_videos = videos

        # Calculate metrics for each period
        analysis = {
            'channel_info': channel_info,
            'metrics': {
                'recent_7_videos': {
                    **self._calculate_metrics(recent_videos),
                    'videos': recent_videos  # Include actual video data
                },
                'last_6_months': {
                    **self._calculate_metrics(six_month_videos),
                    'videos': six_month_videos  # Include actual video data
                },
                'all_time': {
                    **self._calculate_metrics(all_time_videos),
                    'videos': all_time_videos  # Include actual video data
                }
            }
        }

        return analysis

    def _calculate_metrics(self, videos: List[Dict]) -> Dict:
        """Calculate various metrics for a set of videos."""
        if not videos:
            return {}

        total_views = sum(int(v['statistics']['viewCount']) for v in videos)
        total_likes = sum(int(v['statistics']['likeCount']) for v in videos)
        total_comments = sum(int(v['statistics']['commentCount']) for v in videos)
        total_duration = sum(self._parse_duration(v['duration']) for v in videos)
        
        video_count = len(videos)
        
        # Calculate upload frequency
        if len(videos) >= 2:
            first_video_date = datetime.fromisoformat(videos[-1]['snippet']['publishedAt'].replace('Z', '+00:00'))
            last_video_date = datetime.fromisoformat(videos[0]['snippet']['publishedAt'].replace('Z', '+00:00'))
            days_between = (last_video_date - first_video_date).days
            upload_frequency = days_between / (video_count - 1) if days_between > 0 else 0
        else:
            upload_frequency = 0

        return {
            'video_count': video_count,
            'average_views': total_views / video_count if video_count > 0 else 0,
            'average_likes': total_likes / video_count if video_count > 0 else 0,
            'average_comments': total_comments / video_count if video_count > 0 else 0,
            'average_duration': total_duration / video_count if video_count > 0 else 0,
            'engagement_rate': (total_likes + total_comments) / total_views if total_views > 0 else 0,
            'upload_frequency': upload_frequency
        }

    def get_channel_id_from_handle(self, handle: str) -> Optional[str]:
        """Get channel ID from a channel handle (without @ symbol)."""
        try:
            request = self.youtube.search().list(
                part="snippet",
                q=f"@{handle}",
                type="channel",
                maxResults=1
            )
            response = request.execute()
            
            if response['items']:
                return response['items'][0]['snippet']['channelId']
            else:
                self.logger.warning(f"No channel found with handle: {handle}")
                return None
                
        except HttpError as e:
            self.logger.error(f"Error fetching channel ID: {e}")
            return None

if __name__ == "__main__":
    # Example usage
    scraper = YouTubeDataScraper()
    # Test getting channel ID from handle
    handle = "HPWalkingTours"
    channel_id = scraper.get_channel_id_from_handle(handle)
    if channel_id:
        print(f"Channel ID for @{handle}: {channel_id}")
        # Get channel info
        channel_info = scraper.get_channel_info(channel_id)
        print("\nChannel Info:")
        print(channel_info)
    else:
        print(f"Could not find channel ID for @{handle}") 