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
        """Initialize the YouTube API client."""
        self.config = self._load_config(config_path)
        self.youtube = self._build_youtube_client()
        self.logger = logging.getLogger(__name__)
        self._video_categories_cache = None  # Cache for video categories
        self._last_api_call = 0  # Track last API call time for rate limiting
        self._min_api_interval = 0.1  # Minimum time between API calls (100ms)
        self._max_retries = 3  # Maximum number of retries for API calls
        self._retry_delay = 1  # Initial retry delay in seconds

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

    def _rate_limit(self):
        """Implement rate limiting for API calls."""
        current_time = time.time()
        time_since_last_call = current_time - self._last_api_call
        if time_since_last_call < self._min_api_interval:
            time.sleep(self._min_api_interval - time_since_last_call)
        self._last_api_call = time.time()

    def _make_api_request(self, request_func, *args, **kwargs):
        """Make an API request with retry logic and rate limiting."""
        for attempt in range(self._max_retries):
            try:
                self._rate_limit()
                return request_func(*args, **kwargs).execute()
            except HttpError as e:
                if e.resp.status in [429, 500, 502, 503, 504]:  # Rate limit or server errors
                    if attempt < self._max_retries - 1:
                        delay = self._retry_delay * (2 ** attempt)  # Exponential backoff
                        self.logger.warning(f"API request failed (attempt {attempt + 1}/{self._max_retries}). Retrying in {delay:.1f} seconds...")
                        time.sleep(delay)
                        continue
                self.logger.error(f"API request failed after {self._max_retries} attempts: {str(e)}")
                raise
            except Exception as e:
                self.logger.error(f"Unexpected error during API request: {str(e)}")
                raise

    def get_channel_info(self, channel_ids: Union[str, List[str]]) -> List[dict]:
        """
        Get comprehensive channel information using channel ID(s).
        
        Args:
            channel_ids: Single channel ID or list of channel IDs
            
        Returns:
            List[dict]: List of channel information including all specified fields
        """
        try:
            if isinstance(channel_ids, str):
                channel_ids = [channel_ids]
            batch_size = 50
            all_channels = []
            for i in range(0, len(channel_ids), batch_size):
                batch_ids = channel_ids[i:i + batch_size]
                next_page_token = None
                while True:
                    response = self._make_api_request(
                        self.youtube.channels().list,
                        part="snippet,contentDetails,statistics,topicDetails,brandingSettings,status,contentOwnerDetails,localizations",
                        id=','.join(batch_ids),
                        pageToken=next_page_token,
                        maxResults=50
                    )
                    if not response.get('items'):
                        if not all_channels:
                            self.logger.warning(f"No channels found for IDs: {batch_ids}")
                        break
                    all_channels.extend(response['items'])
                    page_info = response.get('pageInfo', {})
                    self.logger.info(f"Retrieved page with {len(response['items'])} items. Total results: {page_info.get('totalResults', 'unknown')}, Results per page: {page_info.get('resultsPerPage', 'unknown')}")
                    next_page_token = response.get('nextPageToken')
                    if not next_page_token:
                        break
            self.logger.info(f"Retrieved {len(all_channels)} channel(s) for {len(channel_ids)} channel ID(s)")
            return all_channels
        except Exception as e:
            self.logger.error(f"Error getting channel info: {str(e)}")
            raise

    def get_video_categories(self) -> Dict[str, str]:
        """
        Get video category IDs and their names with caching.
        
        Returns:
            Dict[str, str]: Dictionary mapping category IDs to their names
        """
        if self._video_categories_cache is not None:
            return self._video_categories_cache

        try:
            # Get categories for US region (most comprehensive)
            response = self._make_api_request(
                self.youtube.videoCategories().list,
                part='snippet',
                regionCode='US'
            )
            
            categories = {}
            for item in response.get('items', []):
                categories[item['id']] = item['snippet']['title']
            
            self._video_categories_cache = categories
            self.logger.info(f"Retrieved and cached {len(categories)} video categories")
            return categories
            
        except Exception as e:
            self.logger.error(f"Error fetching video categories: {str(e)}")
            return {}

    def get_channel_videos(self, channel_ids: Union[str, List[str]], max_results: int = None) -> Dict[str, dict]:
        """
        Get all videos from multiple channels' uploads playlists.
        
        Args:
            channel_ids: Single channel ID or list of channel IDs
            max_results: Optional maximum number of videos to retrieve per channel
            
        Returns:
            Dict[str, dict]: Dictionary mapping channel IDs to their video data
        """
        try:
            if isinstance(channel_ids, str):
                channel_ids = [channel_ids]
            categories = self.get_video_categories()
            channel_response = self._make_api_request(
                self.youtube.channels().list,
                part='contentDetails',
                id=','.join(channel_ids)
            )
            if not channel_response.get('items'):
                self.logger.error(f"No channels found for IDs: {channel_ids}")
                return {}
            channel_playlists = {
                item['id']: item['contentDetails']['relatedPlaylists']['uploads']
                for item in channel_response['items']
            }
            all_videos_data = {}
            batch_size = 50
            for channel_id, uploads_playlist_id in channel_playlists.items():
                self.logger.info(f"Processing videos for channel {channel_id}")
                videos = []
                next_page_token = None
                total_retrieved = 0
                while True:
                    if max_results:
                        remaining = max_results - total_retrieved
                        if remaining <= 0:
                            break
                        current_max = min(batch_size, remaining)
                    else:
                        current_max = batch_size
                    playlist_response = self._make_api_request(
                        self.youtube.playlistItems().list,
                        part='snippet,contentDetails',
                        playlistId=uploads_playlist_id,
                        maxResults=current_max,
                        pageToken=next_page_token
                    )
                    if not playlist_response.get('items'):
                        break
                    video_ids = [item['contentDetails']['videoId'] for item in playlist_response['items']]
                    for i in range(0, len(video_ids), batch_size):
                        batch_ids = video_ids[i:i + batch_size]
                        video_response = self._make_api_request(
                            self.youtube.videos().list,
                            part='snippet,contentDetails,statistics,status,topicDetails,localizations',
                            id=','.join(batch_ids)
                        )
                        for video in video_response.get('items', []):
                            category_id = video['snippet'].get('categoryId', '')
                            category_name = categories.get(category_id, 'Unknown')
                            video_data = {
                                'id': video['id'],
                                'etag': video['etag'],
                                'title': video['snippet']['title'],
                                'description': video['snippet']['description'],
                                'published_at': video['snippet']['publishedAt'],
                                'channel_id': video['snippet']['channelId'],
                                'channel_title': video['snippet']['channelTitle'],
                                'tags': ','.join(video['snippet'].get('tags', [])),
                                'category_id': category_id,
                                'category_name': category_name,
                                'live_broadcast_content': video['snippet'].get('liveBroadcastContent', 'none'),
                                'default_language': video['snippet'].get('defaultLanguage', ''),
                                'default_audio_language': video['snippet'].get('defaultAudioLanguage', ''),
                                'duration': video['contentDetails'].get('duration', 'PT0S'),
                                'duration_seconds': self._parse_duration(video['contentDetails'].get('duration', 'PT0S')),
                                'dimension': video['contentDetails'].get('dimension', ''),
                                'definition': video['contentDetails'].get('definition', ''),
                                'caption': video['contentDetails'].get('caption', ''),
                                'licensed_content': video['contentDetails'].get('licensedContent', False),
                                'projection': video['contentDetails'].get('projection', ''),
                                'has_custom_thumbnail': video['contentDetails'].get('hasCustomThumbnail', False),
                                'view_count': int(video['statistics'].get('viewCount', 0)),
                                'like_count': int(video['statistics'].get('likeCount', 0)),
                                'dislike_count': int(video['statistics'].get('dislikeCount', 0)),
                                'favorite_count': int(video['statistics'].get('favoriteCount', 0)),
                                'comment_count': int(video['statistics'].get('commentCount', 0)),
                                'privacy_status': video.get('status', {}).get('privacyStatus', ''),
                                'upload_status': video.get('status', {}).get('uploadStatus', ''),
                                'license': video.get('status', {}).get('license', ''),
                                'embeddable': video.get('status', {}).get('embeddable', False),
                                'public_stats_viewable': video.get('status', {}).get('publicStatsViewable', False),
                                'made_for_kids': video.get('status', {}).get('madeForKids', False),
                                'topic_ids': ','.join(video.get('topicDetails', {}).get('topicIds', [])),
                                'topic_categories': ','.join(video.get('topicDetails', {}).get('topicCategories', []))
                            }
                            videos.append(video_data)
                    total_retrieved += len(video_ids)
                    self.logger.info(f"Retrieved {len(video_ids)} videos for channel {channel_id}. Total so far: {total_retrieved}")
                    if max_results and total_retrieved >= max_results:
                        break
                    next_page_token = playlist_response.get('nextPageToken')
                    if not next_page_token:
                        break
                self.logger.info(f"Successfully retrieved {len(videos)} videos for channel {channel_id}")
                all_videos_data[channel_id] = {
                    'videos': videos,
                    'total_retrieved': len(videos),
                    'playlist_id': uploads_playlist_id,
                    'categories': categories
                }
            return all_videos_data
        except Exception as e:
            self.logger.error(f"Error fetching videos: {str(e)}")
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

        # Sort videos by publish date (newest first)
        videos.sort(key=lambda x: x['snippet']['publishedAt'], reverse=True)
        
        # Debug print: Show latest 10 videos with their view counts
        print("\nLatest 10 videos from channel:")
        print("Title | Views | Published Date")
        print("-" * 80)
        for video in videos[:10]:
            title = video['snippet']['title']
            views = int(video['statistics'].get('viewCount', 0))
            published = video['snippet']['publishedAt']
            print(f"{title[:60]:<60} | {views:,} | {published}")
        print("-" * 80)
        
        # Also print the video with the highest views
        if videos:
            max_views_video = max(videos, key=lambda x: int(x['statistics'].get('viewCount', 0)))
            print("\nVideo with highest views:")
            print(f"Title: {max_views_video['snippet']['title']}")
            print(f"Views: {int(max_views_video['statistics'].get('viewCount', 0)):,}")
            print(f"Published: {max_views_video['snippet']['publishedAt']}")
            print("-" * 80)

        # Calculate metrics for different time periods
        now = datetime.now(timezone.utc)  # Make timezone-aware
        six_months_ago = now - timedelta(days=180)

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

    def get_video_comments(self, video_id: str) -> List[Dict]:
        """
        Get all comments for a specific video.
        
        Args:
            video_id: The ID of the video to fetch comments for
            
        Returns:
            List[Dict]: List of comment data including text, author, and metadata
        """
        try:
            comments = []
            next_page_token = None
            
            while True:
                try:
                    response = self._make_api_request(
                        self.youtube.commentThreads().list,
                        part='snippet,replies',
                        videoId=video_id,
                        maxResults=100,  # Maximum allowed by API
                        pageToken=next_page_token,
                        textFormat='plainText'
                    )
                    
                    if not response.get('items'):
                        break
                        
                    for item in response['items']:
                        comment = item['snippet']['topLevelComment']['snippet']
                        comment_data = {
                            'comment_id': item['id'],
                            'video_id': video_id,
                            'author_display_name': comment['authorDisplayName'],
                            'author_channel_id': comment.get('authorChannelId', {}).get('value', ''),
                            'text': comment['textDisplay'],
                            'like_count': int(comment.get('likeCount', 0)),
                            'published_at': comment['publishedAt'],
                            'updated_at': comment['updatedAt']
                        }
                        comments.append(comment_data)
                        
                        # Get replies if any
                        if item.get('replies'):
                            for reply in item['replies']['comments']:
                                reply_snippet = reply['snippet']
                                reply_data = {
                                    'comment_id': reply['id'],
                                    'video_id': video_id,
                                    'parent_id': item['id'],
                                    'author_display_name': reply_snippet['authorDisplayName'],
                                    'author_channel_id': reply_snippet.get('authorChannelId', {}).get('value', ''),
                                    'text': reply_snippet['textDisplay'],
                                    'like_count': int(reply_snippet.get('likeCount', 0)),
                                    'published_at': reply_snippet['publishedAt'],
                                    'updated_at': reply_snippet['updatedAt']
                                }
                                comments.append(reply_data)
                    
                    next_page_token = response.get('nextPageToken')
                    if not next_page_token:
                        break
                        
                except HttpError as e:
                    if e.resp.status == 403:  # Comments disabled
                        self.logger.warning(f"Comments are disabled for video {video_id}")
                        break
                    raise
                    
            self.logger.info(f"Retrieved {len(comments)} comments for video {video_id}")
            return comments
            
        except Exception as e:
            self.logger.error(f"Error fetching comments for video {video_id}: {str(e)}")
            return []

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