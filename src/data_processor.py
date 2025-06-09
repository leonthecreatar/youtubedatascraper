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

    def generate_analysis_report(self, df: pd.DataFrame, output_dir: str):
        """Generate a detailed analysis report in Markdown format."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Start building the report
        report = []
        report.append("# YouTube Channel Analysis Report\n")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Print available columns for debugging
        report.append("## Available Data Columns")
        report.append("```")
        report.append(", ".join(df.columns.tolist()))
        report.append("```\n")
        
        # Channel Overview
        report.append("## Channel Overview")
        report.append(f"- Total Videos: {df['total_videos'].sum():,.0f}")
        report.append(f"- Total Subscribers: {df['subscriber_count'].sum():,.0f}")
        report.append(f"- Total Views: {df['total_views'].sum():,.0f}")
        
        # Basic Statistics
        report.append("\n## Basic Statistics")
        stats_table = []
        stats_table.append("| Metric | Average | Median | Min | Max |")
        stats_table.append("|--------|---------|--------|-----|-----|")
        
        # Define metrics to analyze
        metrics = {
            'average_views': 'Average Views',
            'average_likes': 'Average Likes',
            'average_comments': 'Average Comments',
            'engagement_rate': 'Engagement Rate',
            'upload_frequency_days': 'Upload Frequency (days)'
        }
        
        for col, label in metrics.items():
            if col in df.columns:
                stats = df[col].agg(['mean', 'median', 'min', 'max'])
                if col == 'engagement_rate':
                    stats_table.append(f"| {label} | {stats['mean']:.2%} | {stats['median']:.2%} | {stats['min']:.2%} | {stats['max']:.2%} |")
                elif col == 'upload_frequency_days':
                    stats_table.append(f"| {label} | {stats['mean']:.1f} | {stats['median']:.1f} | {stats['min']:.1f} | {stats['max']:.1f} |")
                else:
                    stats_table.append(f"| {label} | {stats['mean']:,.0f} | {stats['median']:,.0f} | {stats['min']:,.0f} | {stats['max']:,.0f} |")
        
        report.extend(stats_table)
        
        # Channel-specific Analysis
        report.append("\n## Channel-specific Analysis")
        for _, channel in df.iterrows():
            report.append(f"\n### {channel['channel_name']}")
            report.append(f"- Channel ID: {channel['channel_id']}")
            report.append(f"- Subscribers: {channel['subscriber_count']:,.0f}")
            report.append(f"- Total Videos: {channel['total_videos']:,.0f}")
            report.append(f"- Total Views: {channel['total_views']:,.0f}")
            report.append(f"- Average Views: {channel['average_views']:,.0f}")
            report.append(f"- Average Likes: {channel['average_likes']:,.0f}")
            report.append(f"- Average Comments: {channel['average_comments']:,.0f}")
            report.append(f"- Engagement Rate: {channel['engagement_rate']:.2%}")
            report.append(f"- Upload Frequency: {channel['upload_frequency_days']:.1f} days")
        
        # Save the report
        report_path = os.path.join(output_dir, 'channel_analysis_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        # Also save raw data as CSV for further analysis
        df.to_csv(os.path.join(output_dir, 'channel_analysis.csv'), index=False)
        
        print(f"\nAnalysis report saved to: {report_path}")
        print(f"Raw data saved to: {os.path.join(output_dir, 'channel_analysis.csv')}")

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