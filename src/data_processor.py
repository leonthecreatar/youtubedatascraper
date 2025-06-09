import json
import os
from datetime import datetime
from typing import Dict, List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml

class YouTubeDataProcessor:
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the data processor with configuration."""
        self.config = self._load_config(config_path)
        self._setup_directories()

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
        """Parse ISO 8601 duration format (PT#H#M#S) to minutes."""
        import re
        if not duration or duration == 'PT0S':
            return 0.0
            
        # Extract hours, minutes, and seconds using regex
        hours = int(re.search(r'(\d+)H', duration).group(1)) if 'H' in duration else 0
        minutes = int(re.search(r'(\d+)M', duration).group(1)) if 'M' in duration else 0
        seconds = int(re.search(r'(\d+)S', duration).group(1)) if 'S' in duration else 0
        
        # Convert to total minutes
        return hours * 60 + minutes + seconds / 60

    def generate_analysis_report(self, df: pd.DataFrame, output_dir: str):
        """Generate a detailed analysis report in Markdown format."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract all video data from the nested structure
        all_videos = []
        channel_metadata = []
        
        for _, row in df.iterrows():
            channel_info = row['channel_info']
            metrics = row['metrics']
            
            # Store channel metadata
            channel_data = {
                'channel_id': channel_info['id'],
                'channel_name': channel_info['snippet']['title'],
                'subscriber_count': int(channel_info['statistics']['subscriberCount']),
                'total_video_count': int(channel_info['statistics']['videoCount']),
                'view_count': int(channel_info['statistics']['viewCount'])
            }
            
            # Extract video data from all time periods
            for period, videos in metrics.items():
                if isinstance(videos, dict) and 'videos' in videos:
                    for video in videos['videos']:
                        duration_minutes = self._parse_duration(video.get('duration', 'PT0S'))
                        video_data = {
                            'channel_id': channel_info['id'],
                            'channel_name': channel_info['snippet']['title'],
                            'video_id': video['id'],
                            'title': video['snippet']['title'],
                            'published_at': pd.to_datetime(video['snippet']['publishedAt']),
                            'views': int(video['statistics'].get('viewCount', 0)),
                            'likes': int(video['statistics'].get('likeCount', 0)),
                            'comments': int(video['statistics'].get('commentCount', 0)),
                            'duration_minutes': duration_minutes,
                            'period': period
                        }
                        all_videos.append(video_data)
            
            channel_metadata.append(channel_data)
        
        # Create DataFrames
        videos_df = pd.DataFrame(all_videos)
        channels_df = pd.DataFrame(channel_metadata)
        
        # Calculate upload frequency for each channel
        channel_frequencies = []
        for channel_id in videos_df['channel_id'].unique():
            channel_videos = videos_df[videos_df['channel_id'] == channel_id]
            first_video = channel_videos['published_at'].min()
            last_video = channel_videos['published_at'].max()
            days_between = (last_video - first_video).days
            total_videos = len(channel_videos)
            frequency = days_between / (total_videos - 1) if total_videos > 1 else 0
            
            channel_frequencies.append({
                'channel_id': channel_id,
                'channel_name': channel_videos['channel_name'].iloc[0],
                'first_video_date': first_video,
                'last_video_date': last_video,
                'days_between': days_between,
                'total_videos': total_videos,
                'upload_frequency': frequency
            })
        
        frequency_df = pd.DataFrame(channel_frequencies)
        
        # Start building the report
        report = []
        report.append("# YouTube Channel Analysis Report\n")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Channel Overview
        report.append("## Channel Overview")
        report.append(f"- Total Channels Analyzed: {len(channels_df)}")
        report.append(f"- Total Videos Analyzed: {len(videos_df):,.0f}")
        report.append(f"- Total Subscribers: {channels_df['subscriber_count'].sum():,.0f}")
        report.append(f"- Total Views: {channels_df['view_count'].sum():,.0f}")
        
        # Video Performance Statistics
        report.append("\n## Video Performance Statistics")
        
        # Video Duration Analysis
        report.append("\n### Video Duration Analysis")
        duration_stats = videos_df['duration_minutes'].agg(['mean', 'median', 'min', 'max'])
        report.append("| Metric | Average | Median | Min | Max |")
        report.append("|--------|---------|--------|-----|-----|")
        report.append(f"| Duration (minutes) | {duration_stats['mean']:.1f} | {duration_stats['median']:.1f} | {duration_stats['min']:.1f} | {duration_stats['max']:.1f} |")
        
        # Overall Video Metrics
        report.append("\n### Overall Video Metrics")
        stats_table = []
        stats_table.append("| Metric | Average | Median | Min | Max |")
        stats_table.append("|--------|---------|--------|-----|-----|")
        
        # Calculate statistics for video metrics
        video_metrics = {
            'views': 'Views per Video',
            'likes': 'Likes per Video',
            'comments': 'Comments per Video'
        }
        
        for col, label in video_metrics.items():
            stats = videos_df[col].agg(['mean', 'median', 'min', 'max'])
            stats_table.append(f"| {label} | {stats['mean']:,.0f} | {stats['median']:,.0f} | {stats['min']:,.0f} | {stats['max']:,.0f} |")
        
        # Calculate engagement rate (likes + comments) / subscribers * 100
        videos_df['engagement_rate'] = (videos_df['likes'] + videos_df['comments']) / videos_df['subscriber_count'].replace(0, 1) * 100
        engagement_stats = videos_df['engagement_rate'].agg(['mean', 'median', 'min', 'max'])
        stats_table.append(f"| Engagement Rate (%) | {engagement_stats['mean']:.2f} | {engagement_stats['median']:.2f} | {engagement_stats['min']:.2f} | {engagement_stats['max']:.2f} |")
        
        report.extend(stats_table)
        
        # Upload Frequency Analysis
        report.append("\n### Upload Frequency Analysis")
        freq_table = []
        freq_table.append("| Channel | First Video | Last Video | Days Between | Total Videos | Avg. Days Between Videos |")
        freq_table.append("|---------|-------------|------------|--------------|--------------|-------------------------|")
        
        for _, row in frequency_df.iterrows():
            freq_table.append(
                f"| {row['channel_name']} | {row['first_video_date'].strftime('%Y-%m-%d')} | "
                f"{row['last_video_date'].strftime('%Y-%m-%d')} | {row['days_between']:,.0f} | "
                f"{row['total_videos']:,.0f} | {row['upload_frequency']:.1f} |"
            )
        
        report.extend(freq_table)
        
        # Channel-specific Analysis
        report.append("\n## Channel-specific Analysis")
        for _, channel in channels_df.iterrows():
            channel_videos = videos_df[videos_df['channel_id'] == channel['channel_id']]
            
            report.append(f"\n### {channel['channel_name']}")
            report.append(f"- Channel ID: {channel['channel_id']}")
            report.append(f"- Subscribers: {channel['subscriber_count']:,.0f}")
            report.append(f"- Total Videos: {channel['total_video_count']:,.0f}")
            report.append(f"- Total Views: {channel['view_count']:,.0f}")
            
            # Channel-specific video metrics
            report.append("\n#### Video Performance")
            report.append(f"- Average Views: {channel_videos['views'].mean():,.0f}")
            report.append(f"- Average Likes: {channel_videos['likes'].mean():,.0f}")
            report.append(f"- Average Comments: {channel_videos['comments'].mean():,.0f}")
            
            # Calculate channel-specific engagement rate
            total_engagement = channel_videos['likes'].sum() + channel_videos['comments'].sum()
            engagement_rate = (total_engagement / channel['subscriber_count']) * 100 if channel['subscriber_count'] > 0 else 0
            report.append(f"- Engagement Rate: {engagement_rate:.2f}%")
            
            # Channel-specific duration analysis
            duration_stats = channel_videos['duration_minutes'].agg(['mean', 'median', 'min', 'max'])
            report.append("\n#### Video Duration")
            report.append(f"- Average Duration: {duration_stats['mean']:.1f} minutes")
            report.append(f"- Median Duration: {duration_stats['median']:.1f} minutes")
            report.append(f"- Shortest Video: {duration_stats['min']:.1f} minutes")
            report.append(f"- Longest Video: {duration_stats['max']:.1f} minutes")
            
            # Upload frequency
            channel_freq = frequency_df[frequency_df['channel_id'] == channel['channel_id']].iloc[0]
            report.append(f"\n#### Upload Frequency")
            report.append(f"- First Video: {channel_freq['first_video_date'].strftime('%Y-%m-%d')}")
            report.append(f"- Last Video: {channel_freq['last_video_date'].strftime('%Y-%m-%d')}")
            report.append(f"- Average Days Between Videos: {channel_freq['upload_frequency']:.1f}")
        
        # Save the report
        report_path = os.path.join(output_dir, 'channel_analysis_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        # Save raw data as CSV for further analysis
        videos_df.to_csv(os.path.join(output_dir, 'video_analysis.csv'), index=False)
        channels_df.to_csv(os.path.join(output_dir, 'channel_analysis.csv'), index=False)
        
        print(f"\nAnalysis report saved to: {report_path}")
        print(f"Raw data saved to: {os.path.join(output_dir, 'video_analysis.csv')} and {os.path.join(output_dir, 'channel_analysis.csv')}")

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