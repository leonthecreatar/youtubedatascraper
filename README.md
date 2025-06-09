# YouTube Channel Analytics

A Python-based tool for analyzing YouTube channel performance metrics. This tool fetches data from the YouTube Data API v3 and generates comprehensive analytics reports for specified channels.

## Features

- Fetch channel statistics and video data using YouTube Data API v3
- Calculate key performance metrics:
  - Average views, likes, and comments
  - Engagement rates
  - Upload frequency
  - Channel growth metrics
- Generate detailed analysis reports in Markdown format
- Support for analyzing multiple channels
- Data export to CSV for further analysis

## Prerequisites

- Python 3.11 or later
- YouTube Data API v3 key
- Required Python packages (see `requirements.txt`)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/youtube-channel-analytics.git
cd youtube-channel-analytics
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Configure your YouTube API key:
   - Get an API key from the [Google Cloud Console](https://console.cloud.google.com/)
   - Enable the YouTube Data API v3
   - Copy your API key
   - Update `config/config.yaml` with your API key

## Configuration

Edit `config/config.yaml` to specify:
- Your YouTube API key
- Channel IDs to analyze
- Analysis parameters

Example configuration:
```yaml
youtube_api:
  api_key: "YOUR_API_KEY_HERE"
  max_results: 50  # Maximum number of videos to analyze per channel

channels:
  - name: "Channel Name"
    channel_id: "CHANNEL_ID_HERE"
```

## Usage

Run the analysis script:
```bash
python src/main.py
```

The script will:
1. Fetch data for each configured channel
2. Process and analyze the data
3. Generate a detailed report in `output/channel_analysis_report.md`
4. Save raw data to `output/channel_analysis.csv`

## Output

The analysis generates:
- A comprehensive Markdown report with:
  - Channel overview statistics
  - Basic metrics (views, likes, comments, engagement)
  - Channel-specific analysis
- Raw data in CSV format for further analysis

## Security Note

Never commit your API key to the repository. The `config.yaml` file is included in `.gitignore` to prevent accidental exposure of sensitive information.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- YouTube Data API v3 for providing the data
- Python community for the excellent libraries used in this project 