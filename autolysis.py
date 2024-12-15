# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pandas",
#     "seaborn",
#     "matplotlib",
#     "httpx",
#     "tenacity",
#     "scikit-learn",
#     "numpy"
# ]
# ///

import os
import sys
import json
import base64
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
from tenacity import retry, stop_after_attempt, wait_random
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Constants
AIPROXY_TOKEN = os.environ.get("AIPROXY_TOKEN")
API_URL = "https://api.openai.com/v1/chat/completions"
VISION_URL = "https://api.openai.com/v1/chat/completions"

def load_and_analyze_data(filepath):
    """Load CSV and perform initial analysis"""
    df = pd.read_csv(filepath)
    
    # Basic statistics
    stats = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing": df.isnull().sum().to_dict(),
        "numeric_summary": df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {},
    }
    
    return df, stats

@retry(stop=stop_after_attempt(3), wait=wait_random(min=1, max=3))
async def query_llm(messages, functions=None):
    """Query the LLM with retry logic"""
    import httpx
    
    headers = {
        "Authorization": f"Bearer {AIPROXY_TOKEN}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "gpt-4",
        "messages": messages,
        "temperature": 0.7
    }
    
    if functions:
        payload["functions"] = functions
        payload["function_call"] = "auto"
    
    async with httpx.AsyncClient() as client:
        response = await client.post(API_URL, json=payload, headers=headers)
        return response.json()

def create_visualizations(df):
    """Create data visualizations"""
    visualizations = []
    
    # Correlation heatmap for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        plt.figure(figsize=(10, 8))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        plt.savefig('correlation.png')
        plt.close()
        visualizations.append('correlation.png')

    # Time series if date column exists
    date_cols = df.select_dtypes(include=['datetime64']).columns
    if len(date_cols) > 0:
        plt.figure(figsize=(12, 6))
        df.set_index(date_cols[0]).plot()
        plt.title('Time Series Analysis')
        plt.tight_layout()
        plt.savefig('timeseries.png')
        plt.close()
        visualizations.append('timeseries.png')

    return visualizations

async def main():
    if len(sys.argv) != 2:
        print("Usage: uv run autolysis.py <csv_file>")
        sys.exit(1)
        
    csv_file = sys.argv[1]
    
    # Load and analyze data
    df, stats = load_and_analyze_data(csv_file)
    
    # Create visualizations
    viz_files = create_visualizations(df)
    
    # Generate story with LLM
    initial_prompt = f"""
    Analyze this dataset and create a story:
    
    Filename: {csv_file}
    Statistics: {json.dumps(stats, indent=2)}
    
    Focus on key insights and implications.
    Format the response in Markdown.
    """
    
    response = await query_llm([{"role": "user", "content": initial_prompt}])
    
    # Write README.md
    with open('README.md', 'w') as f:
        story = response['choices'][0]['message']['content']
        f.write(story)
        
        # Add visualization references
        if viz_files:
            f.write("\n\n## Visualizations\n\n")
            for viz in viz_files:
                f.write(f"![{viz}]({viz})\n\n")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
