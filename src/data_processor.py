import json
import os
from datetime import datetime, timezone
from typing import Dict, List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
import logging
import re

class YouTubeDataProcessor:
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the data processor with configuration."""
        self.config = self._load_config(config_path)
        self._setup_directories()
        # Set up logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

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

    def generate_analysis_report(self, df: pd.DataFrame, output_dir: str) -> str:
        """Generate a focused analysis report with specific metrics using all-time data."""
        if df.empty:
            return "No data available for analysis."

        os.makedirs(output_dir, exist_ok=True)
        
        # Debug: Print DataFrame structure
        self.logger.info("DataFrame columns: %s", df.columns.tolist())
        self.logger.info("DataFrame head:\n%s", df.head())
        
        # Extract and process data
        all_videos = []
        all_channels = []
        
        # Process each channel's data
        for idx, row in df.iterrows():
            try:
                # Debug: Print row data
                self.logger.info("Processing row %d:", idx)
                self.logger.info("Channel info keys: %s", row['channel_info'].keys() if isinstance(row['channel_info'], dict) else "Not a dict")
                self.logger.info("Metrics keys: %s", row['metrics'].keys() if isinstance(row['metrics'], dict) else "Not a dict")
                
                channel_info = row['channel_info']
                metrics = row['metrics']
                
                # Debug: Print channel info and statistics
                self.logger.info("Channel info: %s", channel_info)
                self.logger.info("Channel statistics: %s", channel_info.get('statistics', {}))
                
                # Get channel's total view count from API
                channel_total_views = int(channel_info.get('statistics', {}).get('viewCount', 0))
                self.logger.info(f"Channel total views from API: {channel_total_views:,}")
                
                # Get videos only from all_time period
                channel_videos = []
                if 'all_time' in metrics and 'videos' in metrics['all_time']:
                    period_videos = metrics['all_time']['videos']
                    if isinstance(period_videos, list):
                        for video in period_videos:
                            if isinstance(video, dict) and 'snippet' in video:
                                try:
                                    # Debug: Print video content details
                                    content_details = video.get('contentDetails', {})
                                    self.logger.debug(f"Video content details: {content_details}")
                                    
                                    # Safely get video data
                                    stats = video.get('statistics', {})
                                    duration = video.get('duration', 'PT0S')
                                    duration_minutes = self._parse_duration(duration)
                                    
                                    # Debug: Log duration parsing
                                    self.logger.debug(f"Video ID: {video.get('id')}, Raw duration: {duration}, Parsed duration (minutes): {duration_minutes}")
                                    
                                    video_data = {
                                        'channel_id': channel_info.get('id', ''),
                                        'channel_title': channel_info.get('snippet', {}).get('title', ''),
                                        'video_id': video.get('id', ''),
                                        'title': video['snippet'].get('title', ''),
                                        'published_at': pd.to_datetime(video['snippet'].get('publishedAt', '')),
                                        'views': int(stats.get('viewCount', 0)),
                                        'likes': int(stats.get('likeCount', 0)),
                                        'comments': int(stats.get('commentCount', 0)),
                                        'shares': int(stats.get('shareCount', 0)),
                                        'duration_minutes': duration_minutes
                                    }
                                    
                                    # Debug: Log the complete video data
                                    self.logger.debug(f"Processed video data: {video_data}")
                                    
                                    # Calculate engagement rates
                                    video_data['engagement_rate_per_view'] = (
                                        (video_data['likes'] + video_data['comments'] + video_data['shares']) / 
                                        video_data['views'] * 100 if video_data['views'] > 0 else 0
                                    )
                                    channel_videos.append(video_data)
                                    all_videos.append(video_data)
                                except Exception as e:
                                    self.logger.warning(f"Error processing video {video.get('id', 'unknown')}: {str(e)}")
                                    continue

                if channel_videos:
                    # Calculate channel metrics
                    videos_df = pd.DataFrame(channel_videos)
                    
                    # Debug: Print video duration statistics
                    self.logger.info(f"Channel {channel_info.get('snippet', {}).get('title', '')} duration stats:")
                    self.logger.info(f"Mean duration: {videos_df['duration_minutes'].mean():.2f} minutes")
                    self.logger.info(f"Duration range: {videos_df['duration_minutes'].min():.2f} - {videos_df['duration_minutes'].max():.2f} minutes")
                    
                    subscriber_count = int(channel_info.get('statistics', {}).get('subscriberCount', 0))
                    total_engagement = sum(v['likes'] + v['comments'] + v['shares'] for v in channel_videos)
                    
                    # Calculate upload frequency
                    if len(videos_df) >= 2:
                        first_video = videos_df['published_at'].min()
                        last_video = videos_df['published_at'].max()
                        days_between = (last_video - first_video).days
                        upload_frequency = days_between / (len(videos_df) - 1) if len(videos_df) > 1 else 0
                    else:
                        upload_frequency = 0

                    # Calculate average views per video using channel's total view count
                    avg_views_per_video = channel_total_views / int(channel_info.get('statistics', {}).get('videoCount', len(channel_videos)))
                    
                    channel_data = {
                        'channel_id': channel_info.get('id', ''),
                        'title': channel_info.get('snippet', {}).get('title', ''),
                        'subscriber_count': subscriber_count,
                        'video_count': int(channel_info.get('statistics', {}).get('videoCount', len(channel_videos))),  # Use API's video count
                        'total_views': channel_total_views,  # Use channel's total views from API
                        'avg_views_per_video': avg_views_per_video,  # Calculate using total views
                        'median_views_per_video': videos_df['views'].median(),  # Keep median from sampled videos
                        'highest_viewed_video': videos_df['views'].max(),  # Keep highest from sampled videos
                        'avg_engagement_rate_per_view': videos_df['engagement_rate_per_view'].mean(),
                        'avg_engagement_rate_per_subscriber': (total_engagement / subscriber_count * 100) if subscriber_count > 0 else 0,
                        'avg_video_duration': videos_df['duration_minutes'].mean(),
                        'upload_frequency': upload_frequency
                    }
                    
                    # Debug: Log channel data with views
                    self.logger.info(f"Channel data - Total views: {channel_data['total_views']:,}, Avg views per video: {channel_data['avg_views_per_video']:,.0f}")
                    
                    all_channels.append(channel_data)

            except Exception as e:
                self.logger.error(f"Error processing channel data: {str(e)}")
                continue

        # Create DataFrames
        channels_df = pd.DataFrame(all_channels)
        videos_df = pd.DataFrame(all_videos)

        # Debug: Print view statistics for all channels
        self.logger.info("\nOverall view statistics:")
        self.logger.info(f"Total views across all channels: {channels_df['total_views'].sum():,}")
        self.logger.info(f"Average views per channel: {channels_df['total_views'].mean():,.0f}")
        self.logger.info(f"Average views per video across all channels: {channels_df['avg_views_per_video'].mean():,.0f}")
        
        # Debug: Print duration statistics for all channels
        self.logger.info("\nOverall duration statistics:")
        self.logger.info(f"Mean duration across all channels: {channels_df['avg_video_duration'].mean():.2f} minutes")
        self.logger.info(f"Duration range across all channels: {channels_df['avg_video_duration'].min():.2f} - {channels_df['avg_video_duration'].max():.2f} minutes")
        
        # Debug: Print final DataFrames
        self.logger.info("\nChannels DataFrame columns: %s", channels_df.columns.tolist())
        self.logger.info("Channels DataFrame head:\n%s", channels_df.head())
        self.logger.info("\nVideos DataFrame columns: %s", videos_df.columns.tolist())
        self.logger.info("Videos DataFrame head:\n%s", videos_df.head())

        if channels_df.empty or videos_df.empty:
            return "No valid data available for analysis."

        # Generate the report
        report = []
        report.append("# YouTube Channel Analysis Report (All-Time Statistics)\n")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # 1. Overall Channel Metrics
        report.append("## 1. Overall Channel Metrics\n")
        
        # a. Channels ranked by subscribers
        report.append("### a. Channels Ranked by Subscribers")
        subscriber_rankings = channels_df.sort_values('subscriber_count', ascending=False)
        for _, channel in subscriber_rankings.iterrows():
            report.append(f"- **{channel['title']}**: {channel['subscriber_count']:,} subscribers")
        
        # b. Channels ranked by total views
        report.append("\n### b. Channels Ranked by Total Views")
        view_rankings = channels_df.sort_values('total_views', ascending=False)
        for _, channel in view_rankings.iterrows():
            report.append(f"- **{channel['title']}**: {channel['total_views']:,} views")
        
        # c-h. Overall averages
        report.append("\n### Overall Averages Across All Channels")
        overall_metrics = {
            'Average Views per Video': f"{channels_df['avg_views_per_video'].mean():,.0f}",
            'Median Views per Video': f"{channels_df['median_views_per_video'].median():,.0f}",
            'Average Engagement Rate per Video': f"{channels_df['avg_engagement_rate_per_view'].mean():.2f}%",
            'Average Engagement Rate per Subscriber': f"{channels_df['avg_engagement_rate_per_subscriber'].mean():.2f}%",
            'Average Video Duration': f"{channels_df['avg_video_duration'].mean():.1f} minutes",
            'Average Upload Frequency': f"{channels_df['upload_frequency'].mean():.1f} days between videos"
        }
        for metric, value in overall_metrics.items():
            report.append(f"- **{metric}**: {value}")

        # 2. Individual Channel Analysis
        report.append("\n## 2. Individual Channel Analysis\n")
        
        for _, channel in channels_df.sort_values('subscriber_count', ascending=False).iterrows():
            report.append(f"### {channel['title']}\n")
            
            # Basic Channel Information
            report.append("#### Channel Overview")
            report.append(f"- **Subscribers**: {channel['subscriber_count']:,}")
            report.append(f"- **Total Videos**: {channel['video_count']:,}")
            report.append(f"- **Total Views**: {channel['total_views']:,}")
            
            # Channel Performance
            report.append("\n#### Channel Performance")
            performance_metrics = {
                'Average Views per Video': f"{channel['avg_views_per_video']:,.0f}",
                'Median Views per Video': f"{channel['median_views_per_video']:,.0f}",
                'Highest Viewed Video': f"{channel['highest_viewed_video']:,.0f}",
                'Average Engagement Rate per View': f"{channel['avg_engagement_rate_per_view']:.2f}%",
                'Average Engagement Rate per Subscriber': f"{channel['avg_engagement_rate_per_subscriber']:.2f}%",
                'Average Video Duration': f"{channel['avg_video_duration']:.1f} minutes",
                'Upload Frequency': f"{channel['upload_frequency']:.1f} days between videos"
            }
            for metric, value in performance_metrics.items():
                report.append(f"- **{metric}**: {value}")
            
            report.append("\n---\n")  # Separator between channels

        # Save the report
        report_path = os.path.join(output_dir, 'channel_analysis_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))

        # Save raw data
        channels_df.to_csv(os.path.join(output_dir, 'channel_analysis.csv'), index=False)
        videos_df.to_csv(os.path.join(output_dir, 'video_analysis.csv'), index=False)

        self.logger.info(f"Analysis report saved to: {report_path}")
        self.logger.info(f"Raw data saved to: {output_dir}")

        return '\n'.join(report)

    def _plot_channel_overview(self, df: pd.DataFrame, output_dir: str):
        """Generate channel overview visualizations."""
        # Subscriber distribution
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df[df['period'] == 'all_time'], 
                   x='channel_name', 
                   y='subscriber_count')
        plt.xticks(rotation=45, ha='right')
        plt.title('Subscriber Count by Channel')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'subscriber_distribution.png'))
        plt.close()

        # Views per video
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, 
                   x='channel_name', 
                   y='average_views',
                   hue='period')
        plt.xticks(rotation=45, ha='right')
        plt.title('Average Views per Video by Period')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'views_per_video.png'))
        plt.close()

    def _plot_engagement_analysis(self, df: pd.DataFrame, output_dir: str):
        """Generate engagement analysis visualizations."""
        # Engagement rate
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, 
                   x='channel_name', 
                   y='engagement_rate',
                   hue='period')
        plt.xticks(rotation=45, ha='right')
        plt.title('Engagement Rate by Channel and Period')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'engagement_rate.png'))
        plt.close()

        # Likes and comments correlation
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df[df['period'] == 'all_time'],
                       x='average_likes',
                       y='average_comments',
                       hue='channel_name')
        plt.title('Likes vs Comments Correlation')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'likes_comments_correlation.png'))
        plt.close()

    def _plot_content_analysis(self, df: pd.DataFrame, output_dir: str):
        """Generate content analysis visualizations."""
        # Video duration distribution
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df,
                   x='channel_name',
                   y='average_duration',
                   hue='period')
        plt.xticks(rotation=45, ha='right')
        plt.title('Video Duration Distribution')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'video_duration.png'))
        plt.close()

        # Upload frequency
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df[df['period'] == 'all_time'],
                   x='channel_name',
                   y='upload_frequency_days')
        plt.xticks(rotation=45, ha='right')
        plt.title('Average Days Between Uploads')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'upload_frequency.png'))
        plt.close()

    def _generate_summary_stats(self, df: pd.DataFrame, output_dir: str):
        """Generate summary statistics report."""
        summary = df.groupby(['channel_name', 'period']).agg({
            'subscriber_count': 'first',
            'total_videos': 'first',
            'average_views': ['mean', 'std'],
            'average_likes': ['mean', 'std'],
            'average_comments': ['mean', 'std'],
            'engagement_rate': ['mean', 'std'],
            'average_duration': ['mean', 'std'],
            'upload_frequency_days': 'mean'
        }).round(2)

        # Save to CSV
        summary.to_csv(os.path.join(output_dir, 'summary_statistics.csv'))

        # Generate markdown report
        report = ["# YouTube Channel Analysis Report\n"]
        report.append("## Summary Statistics\n")
        report.append(summary.to_markdown())
        report.append("\n## Key Findings\n")
        
        # Add key findings based on the data
        for channel in df['channel_name'].unique():
            channel_data = df[df['channel_name'] == channel]
            report.append(f"\n### {channel}\n")
            
            # Add channel-specific insights
            recent_data = channel_data[channel_data['period'] == 'recent_7_videos'].iloc[0]
            all_time_data = channel_data[channel_data['period'] == 'all_time'].iloc[0]
            
            report.append(f"- Recent Performance (Last 7 videos):")
            report.append(f"  - Average Views: {recent_data['average_views']:,.0f}")
            report.append(f"  - Engagement Rate: {recent_data['engagement_rate']:.2%}")
            report.append(f"  - Average Video Length: {recent_data['average_duration']/60:.1f} minutes")
            
            report.append(f"\n- All-Time Performance:")
            report.append(f"  - Total Subscribers: {all_time_data['subscriber_count']:,.0f}")
            report.append(f"  - Total Videos: {all_time_data['total_videos']}")
            report.append(f"  - Average Upload Frequency: {all_time_data['upload_frequency_days']:.1f} days")

        # Save report
        with open(os.path.join(output_dir, 'analysis_report.md'), 'w') as f:
            f.write('\n'.join(report))

if __name__ == "__main__":
    # Example usage
    processor = YouTubeDataProcessor()
    # Process your data here 