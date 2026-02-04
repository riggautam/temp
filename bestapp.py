"""Enhanced Multi-Sensor LLM Testing + Data Visualization Interface"""
import streamlit as st
import subprocess
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import textwrap
import re
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUTPUT_DIR = Path("./outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'models' not in st.session_state:
    st.session_state.models = []
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'current_data' not in st.session_state:
    st.session_state.current_data = None
if 'window_data' not in st.session_state:
    st.session_state.window_data = None
if 'segment_summary' not in st.session_state:
    st.session_state.segment_summary = None

# ============================================================================
# CONFIGURATION
# ============================================================================

SENSOR_CONFIGS = {
    'Motion': {'type': 'binary', 'unit': '', 'color': '#FF6B6B'},
    'Temperature': {'type': 'continuous', 'unit': '°C', 'color': '#4ECDC4'},
    'Humidity': {'type': 'continuous', 'unit': '%', 'color': '#45B7D1'},
    'Power': {'type': 'continuous', 'unit': 'W', 'color': '#FFA07A'},
    'Light': {'type': 'binary', 'unit': '', 'color': '#FFD93D'}
}

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data_from_file(filepath):
    """Load data from CSV file with all sensor types."""
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['local_timestamp'], format='mixed', dayfirst=True)
    
    for col in ['sensor_name', 'room_name', 'sensor_type']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    
    df['sensor_value'] = pd.to_numeric(df['sensor_value'], errors='coerce')
    
    # Drop erroneous columns if present
    COLUMNS_TO_DROP = ['mobility', 'hygiene', 'dressing', 'meal']
    df = df.drop(columns=[col for col in COLUMNS_TO_DROP if col in df.columns], errors='ignore')
    
    EXCLUDED_SENSORS = ['Accelerometer', 'Tamper']
    df = df[~df['sensor_type'].isin(EXCLUDED_SENSORS)]
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    return df

def extract_window(df, hours, start_time):
    """Extract time window from dataframe."""
    end_time = start_time + timedelta(hours=hours)
    return df[(df['timestamp'] >= start_time) & (df['timestamp'] < end_time)].copy()

def filter_by_rooms(df, selected_rooms):
    """Filter dataframe by selected rooms."""
    if not selected_rooms or 'All Rooms' in selected_rooms:
        return df
    return df[df['room_name'].isin(selected_rooms)].copy()

def filter_by_sensors(df, selected_sensors):
    """Filter dataframe by sensor types."""
    if not selected_sensors or 'All Sensors' in selected_sensors:
        return df
    return df[df['sensor_type'].isin(selected_sensors)].copy()

# ============================================================================
# STATISTICS
# ============================================================================

def compute_sensor_stats(df, sensor_type):
    """Compute statistics for a specific sensor type."""
    sensor_data = df[df['sensor_type'] == sensor_type]
    if len(sensor_data) == 0:
        return None
    
    config = SENSOR_CONFIGS.get(sensor_type, {'type': 'continuous'})
    
    if config['type'] == 'binary':
        activations = sensor_data[sensor_data['sensor_value'] == 1]
        return {
            'total_readings': len(sensor_data),
            'activations': len(activations),
            'activation_rate': len(activations) / len(sensor_data) if len(sensor_data) > 0 else 0,
            'rooms': sensor_data['room_name'].nunique()
        }
    else:
        return {
            'total_readings': len(sensor_data),
            'mean': sensor_data['sensor_value'].mean(),
            'std': sensor_data['sensor_value'].std(),
            'min': sensor_data['sensor_value'].min(),
            'max': sensor_data['sensor_value'].max(),
            'median': sensor_data['sensor_value'].median(),
            'rooms': sensor_data['room_name'].nunique()
        }

def generate_segment_summary(df, start_dt, end_dt, num_segments):
    """Generate formatted summary of sensor data by time segments."""
    total_duration = (end_dt - start_dt).total_seconds()
    segment_duration = total_duration / num_segments
    
    summary_lines = []
    
    # Get all sensor types
    sensor_types = sorted(df['sensor_type'].unique())
    
    for sensor_type in sensor_types:
        sensor_data = df[df['sensor_type'] == sensor_type]
        config = SENSOR_CONFIGS.get(sensor_type, {'type': 'continuous', 'unit': ''})
        
        summary_lines.append(f"{sensor_type}:")
        
        for i in range(num_segments):
            seg_start = start_dt + timedelta(seconds=i * segment_duration)
            seg_end = start_dt + timedelta(seconds=(i + 1) * segment_duration)
            seg_data = sensor_data[(sensor_data['timestamp'] >= seg_start) & 
                                   (sensor_data['timestamp'] < seg_end)]
            
            start_str = seg_start.strftime('%H:%M')
            end_str = seg_end.strftime('%H:%M')
            
            if config['type'] == 'binary':
                # Count activations (sensor_value == 1)
                activations = len(seg_data[seg_data['sensor_value'] == 1])
                summary_lines.append(f"{start_str} - {end_str}: {activations} Activations")
            else:
                # Average for continuous sensors
                if len(seg_data) > 0:
                    avg_value = seg_data['sensor_value'].mean()
                    summary_lines.append(f"{start_str} - {end_str}: {avg_value:.1f} {config['unit']}")
                else:
                    summary_lines.append(f"{start_str} - {end_str}: No data")
        
        summary_lines.append("")  # Blank line between sensor types
    
    return "\n".join(summary_lines)

def generate_multi_sensor_summary(df):
    """Generate summary statistics across all sensor types."""
    summary = {}
    for sensor_type in df['sensor_type'].unique():
        stats = compute_sensor_stats(df, sensor_type)
        if stats:
            summary[sensor_type] = stats
    return summary

# ============================================================================
# VISUALIZATIONS - BINARY SENSORS
# ============================================================================

def plot_motion_scatter(df):
    """Scatter plot of motion activations by room."""
    motion = df[(df['sensor_type'] == 'Motion') & (df['sensor_value'] == 1)].copy()
    if len(motion) == 0:
        return None
    
    room_counts = motion.groupby('room_name').size().to_dict()
    motion['room_label'] = motion['room_name'].apply(lambda x: f"{x} ({room_counts[x]})")
    
    fig = px.scatter(
        motion, x='timestamp', y='room_name', color='room_label',
        title='Motion Activations by Room',
        labels={'timestamp': 'Time', 'room_name': 'Room'},
        height=500
    )
    fig.update_traces(marker=dict(size=8, opacity=0.7))
    fig.update_xaxes(tickformat='%H:%M')
    return fig

def plot_binary_timeline(df, sensor_type='Motion'):
    """Timeline of binary sensor activations."""
    sensor_data = df[(df['sensor_type'] == sensor_type) & (df['sensor_value'] == 1)].copy()
    if len(sensor_data) == 0:
        return None
    
    fig = px.scatter(
        sensor_data, x='timestamp', y='room_name', color='room_name',
        title=f'{sensor_type} Activations Timeline',
        labels={'timestamp': 'Time', 'room_name': 'Room'},
        height=500
    )
    fig.update_traces(marker=dict(size=8, opacity=0.7))
    return fig

def plot_activity_heatmap(df, sensor_type='Motion'):
    """Heatmap of hourly activity by room."""
    sensor_data = df[(df['sensor_type'] == sensor_type) & (df['sensor_value'] == 1)].copy()
    if len(sensor_data) == 0:
        return None
    
    sensor_data['hour'] = sensor_data['timestamp'].dt.hour
    heatmap_data = sensor_data.groupby(['room_name', 'hour']).size().reset_index(name='count')
    pivot = heatmap_data.pivot(index='room_name', columns='hour', values='count').fillna(0)
    
    fig = px.imshow(
        pivot, labels=dict(x="Hour", y="Room", color="Activations"),
        title=f"{sensor_type} Activity Heatmap (Hourly)",
        aspect="auto", height=500, color_continuous_scale='YlOrRd'
    )
    return fig

# ============================================================================
# VISUALIZATIONS - CONTINUOUS SENSORS
# ============================================================================

def plot_continuous_timeseries(df, sensor_type='Temperature'):
    """Line plot of continuous sensor readings over time."""
    sensor_data = df[df['sensor_type'] == sensor_type].copy()
    if len(sensor_data) == 0:
        return None
    
    config = SENSOR_CONFIGS.get(sensor_type, {'unit': '', 'color': '#4ECDC4'})
    
    fig = px.line(
        sensor_data, x='timestamp', y='sensor_value', color='room_name',
        title=f'{sensor_type} Over Time',
        labels={'timestamp': 'Time', 'sensor_value': f'{sensor_type} ({config["unit"]})'},
        height=500
    )
    fig.update_traces(mode='lines+markers', marker=dict(size=4))
    return fig

def plot_continuous_distribution(df, sensor_type='Temperature'):
    """Box/violin plot of continuous sensor distribution by room."""
    sensor_data = df[df['sensor_type'] == sensor_type].copy()
    if len(sensor_data) == 0:
        return None
    
    config = SENSOR_CONFIGS.get(sensor_type, {'unit': '', 'color': '#4ECDC4'})
    
    fig = px.box(
        sensor_data, x='room_name', y='sensor_value', color='room_name',
        title=f'{sensor_type} Distribution by Room',
        labels={'room_name': 'Room', 'sensor_value': f'{sensor_type} ({config["unit"]})'},
        height=500
    )
    return fig

def plot_continuous_heatmap(df, sensor_type='Temperature'):
    """Heatmap of continuous sensor average by room and hour."""
    sensor_data = df[df['sensor_type'] == sensor_type].copy()
    if len(sensor_data) == 0:
        return None
    
    config = SENSOR_CONFIGS.get(sensor_type, {'unit': '', 'color': '#4ECDC4'})
    sensor_data['hour'] = sensor_data['timestamp'].dt.hour
    
    heatmap_data = sensor_data.groupby(['room_name', 'hour'])['sensor_value'].mean().reset_index()
    pivot = heatmap_data.pivot(index='room_name', columns='hour', values='sensor_value')
    
    fig = px.imshow(
        pivot, labels=dict(x="Hour", y="Room", color=f"{sensor_type} ({config['unit']})"),
        title=f"Average {sensor_type} by Room and Hour",
        aspect="auto", height=500, color_continuous_scale='RdYlBu_r'
    )
    return fig

# ============================================================================
# VISUALIZATIONS - MULTI-SENSOR
# ============================================================================

def plot_multi_sensor_overlay(df, sensor_types=['Motion', 'Temperature']):
    """Overlay multiple sensor types on dual y-axes."""
    if len(sensor_types) < 2:
        return None
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # First sensor (left y-axis)
    sensor1 = df[df['sensor_type'] == sensor_types[0]]
    if SENSOR_CONFIGS.get(sensor_types[0], {}).get('type') == 'binary':
        sensor1 = sensor1[sensor1['sensor_value'] == 1]
        sensor1_agg = sensor1.groupby(sensor1['timestamp'].dt.floor('H')).size().reset_index(name='count')
        fig.add_trace(
            go.Scatter(x=sensor1_agg['timestamp'], y=sensor1_agg['count'], 
                      name=f'{sensor_types[0]} Count', mode='lines+markers'),
            secondary_y=False
        )
    else:
        sensor1_agg = sensor1.groupby('timestamp')['sensor_value'].mean().reset_index()
        fig.add_trace(
            go.Scatter(x=sensor1_agg['timestamp'], y=sensor1_agg['sensor_value'],
                      name=sensor_types[0], mode='lines'),
            secondary_y=False
        )
    
    # Second sensor (right y-axis)
    sensor2 = df[df['sensor_type'] == sensor_types[1]]
    if SENSOR_CONFIGS.get(sensor_types[1], {}).get('type') == 'binary':
        sensor2 = sensor2[sensor2['sensor_value'] == 1]
        sensor2_agg = sensor2.groupby(sensor2['timestamp'].dt.floor('H')).size().reset_index(name='count')
        fig.add_trace(
            go.Scatter(x=sensor2_agg['timestamp'], y=sensor2_agg['count'],
                      name=f'{sensor_types[1]} Count', mode='lines+markers'),
            secondary_y=True
        )
    else:
        sensor2_agg = sensor2.groupby('timestamp')['sensor_value'].mean().reset_index()
        fig.add_trace(
            go.Scatter(x=sensor2_agg['timestamp'], y=sensor2_agg['sensor_value'],
                      name=sensor_types[1], mode='lines'),
            secondary_y=True
        )
    
    config1 = SENSOR_CONFIGS.get(sensor_types[0], {'unit': ''})
    config2 = SENSOR_CONFIGS.get(sensor_types[1], {'unit': ''})
    
    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text=f"{sensor_types[0]} ({config1['unit']})", secondary_y=False)
    fig.update_yaxes(title_text=f"{sensor_types[1]} ({config2['unit']})", secondary_y=True)
    fig.update_layout(title=f"{sensor_types[0]} vs {sensor_types[1]}", height=500)
    
    return fig

def plot_correlation_matrix(df):
    """Correlation matrix of continuous sensors."""
    continuous_sensors = [s for s, c in SENSOR_CONFIGS.items() if c['type'] == 'continuous']
    available_sensors = [s for s in continuous_sensors if s in df['sensor_type'].unique()]
    
    if len(available_sensors) < 2:
        return None
    
    # Pivot to wide format
    corr_data = []
    for sensor in available_sensors:
        sensor_vals = df[df['sensor_type'] == sensor].groupby('timestamp')['sensor_value'].mean()
        corr_data.append(sensor_vals)
    
    corr_df = pd.concat(corr_data, axis=1, keys=available_sensors)
    corr_matrix = corr_df.corr()
    
    fig = px.imshow(
        corr_matrix, text_auto='.2f',
        labels=dict(color="Correlation"),
        title="Sensor Correlation Matrix",
        aspect="auto", height=500, color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1
    )
    return fig

def plot_room_environmental_profile(df):
    """Scatter plot of temperature vs humidity by room."""
    temp_data = df[df['sensor_type'] == 'Temperature'][['timestamp', 'room_name', 'sensor_value']].rename(columns={'sensor_value': 'temperature'})
    humidity_data = df[df['sensor_type'] == 'Humidity'][['timestamp', 'room_name', 'sensor_value']].rename(columns={'sensor_value': 'humidity'})
    
    merged = pd.merge(temp_data, humidity_data, on=['timestamp', 'room_name'], how='inner')
    
    if len(merged) == 0:
        return None
    
    fig = px.scatter(
        merged, x='temperature', y='humidity', color='room_name',
        title='Room Environmental Profile (Temperature vs Humidity)',
        labels={'temperature': 'Temperature (°C)', 'humidity': 'Humidity (%)'},
        height=500, opacity=0.6
    )
    return fig

# ============================================================================
# ANOMALY DETECTION
# ============================================================================

def detect_anomalies_continuous(df, sensor_type, sigma=2):
    """Detect anomalies in continuous sensor data using z-score."""
    sensor_data = df[df['sensor_type'] == sensor_type].copy()
    if len(sensor_data) == 0:
        return None
    
    mean = sensor_data['sensor_value'].mean()
    std = sensor_data['sensor_value'].std()
    
    sensor_data['z_score'] = (sensor_data['sensor_value'] - mean) / std if std > 0 else 0
    sensor_data['anomaly'] = sensor_data['z_score'].abs() > sigma
    
    anomalies = sensor_data[sensor_data['anomaly']]
    
    return {
        'total_readings': len(sensor_data),
        'anomalies': len(anomalies),
        'anomaly_rate': len(anomalies) / len(sensor_data) if len(sensor_data) > 0 else 0,
        'mean': mean,
        'std': std,
        'threshold_high': mean + sigma * std,
        'threshold_low': mean - sigma * std,
        'anomaly_data': anomalies
    }

def plot_anomalies(df, sensor_type, sigma=2):
    """Plot continuous sensor with anomalies highlighted."""
    result = detect_anomalies_continuous(df, sensor_type, sigma)
    if not result:
        return None
    
    sensor_data = df[df['sensor_type'] == sensor_type].copy()
    config = SENSOR_CONFIGS.get(sensor_type, {'unit': '', 'color': '#4ECDC4'})
    
    fig = go.Figure()
    
    # Normal data
    normal_data = sensor_data[~sensor_data.index.isin(result['anomaly_data'].index)]
    fig.add_trace(go.Scatter(
        x=normal_data['timestamp'], y=normal_data['sensor_value'],
        mode='markers', name='Normal',
        marker=dict(color='blue', size=4)
    ))
    
    # Anomalies
    if len(result['anomaly_data']) > 0:
        fig.add_trace(go.Scatter(
            x=result['anomaly_data']['timestamp'], y=result['anomaly_data']['sensor_value'],
            mode='markers', name='Anomaly',
            marker=dict(color='red', size=8, symbol='x')
        ))
    
    # Threshold lines
    fig.add_hline(y=result['threshold_high'], line_dash="dash", line_color="red", 
                  annotation_text=f"Upper ({result['threshold_high']:.1f})")
    fig.add_hline(y=result['threshold_low'], line_dash="dash", line_color="red",
                  annotation_text=f"Lower ({result['threshold_low']:.1f})")
    fig.add_hline(y=result['mean'], line_dash="dot", line_color="gray",
                  annotation_text=f"Mean ({result['mean']:.1f})")
    
    fig.update_layout(
        title=f"{sensor_type} Anomaly Detection (±{sigma}σ)",
        xaxis_title="Time",
        yaxis_title=f"{sensor_type} ({config['unit']})",
        height=500
    )
    
    return fig

# ============================================================================
# LLM INTEGRATION
# ============================================================================

def get_available_models():
    """Get list of available Ollama models."""
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            return []
        
        lines = result.stdout.strip().split('\n')
        if len(lines) <= 1:
            return []
        
        models = []
        for line in lines[1:]:
            parts = line.split()
            if parts:
                models.append(parts[0])
        
        return models
    except:
        return []

def sanitize_model_name(model_name):
    """Sanitize model name for use in filenames."""
    return re.sub(r'[^\w\-.]', '_', model_name)

def call_ollama_with_metrics(model, system_prompt, user_prompt, temperature=None, max_tokens=None):
    """Call Ollama and return metrics with optional parameters."""
    try:
        import json
        import urllib.request
        
        start_time = time.time()
        first_token_time = None
        output_text = ""
        
        # If parameters are specified, use API; otherwise use CLI
        if temperature is not None or max_tokens is not None:
            # Use Ollama API
            url = "http://localhost:11434/api/generate"
            
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            data = {
                "model": model,
                "prompt": full_prompt,
                "stream": False
            }
            
            # Add optional parameters
            if temperature is not None:
                data["temperature"] = temperature
            if max_tokens is not None:
                data["num_predict"] = max_tokens
            
            req = urllib.request.Request(
                url,
                data=json.dumps(data).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )
            
            with urllib.request.urlopen(req) as response:
                first_token_time = time.time()
                result = json.loads(response.read().decode('utf-8'))
                output_text = result.get('response', '')
            
            end_time = time.time()
            
        else:
            # Use original CLI method
            process = subprocess.Popen(
                ['ollama', 'run', model],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            process.stdin.write(full_prompt)
            process.stdin.close()
            
            for line in process.stdout:
                if first_token_time is None:
                    first_token_time = time.time()
                output_text += line
            
            process.wait()
            end_time = time.time()
        
        latency = end_time - start_time
        ttft = (first_token_time - start_time) if first_token_time else latency
        
        # Estimate tokens (rough approximation)
        output_tokens = len(output_text.split())
        prompt_tokens = len(system_prompt.split()) + len(user_prompt.split())
        total_tokens = prompt_tokens + output_tokens
        
        throughput = output_tokens / latency if latency > 0 else 0
        compression_ratio = len(output_text) / (len(system_prompt) + len(user_prompt)) if (len(system_prompt) + len(user_prompt)) > 0 else 0
        
        return {
            'output': output_text.strip(),
            'latency_s': latency,
            'ttft_s': ttft,
            'throughput_tok_s': throughput,
            'prompt_tokens': prompt_tokens,
            'output_tokens': output_tokens,
            'total_tokens': total_tokens,
            'compression_ratio': compression_ratio,
            'word_count': len(output_text.split()),
            'avg_chars_per_token': len(output_text) / output_tokens if output_tokens > 0 else 0,
            'temperature': temperature,
            'max_tokens': max_tokens
        }, None
        
    except Exception as e:
        return None, str(e)

        return None, str(e)

def save_output_to_file(model, timestamp, system_prompt, user_prompt, metrics, output):
    """Save output to formatted text file."""
    safe_model = sanitize_model_name(model)
    timestamp_str = datetime.fromisoformat(timestamp).strftime('%Y%m%d_%H%M%S')
    filename = f"{timestamp_str}_{safe_model}.txt"
    filepath = OUTPUT_DIR / filename
    
    wrapper = textwrap.TextWrapper(width=80, break_long_words=False, replace_whitespace=False)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"MODEL: {model}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"TIMESTAMP: {timestamp}\n\n")
        f.write("=" * 80 + "\n")
        f.write("SYSTEM PROMPT:\n")
        f.write("=" * 80 + "\n")
        for line in system_prompt.split('\n'):
            f.write('\n'.join(wrapper.wrap(line) if line.strip() else [line]) + '\n')
        f.write("\n" + "=" * 80 + "\n")
        f.write("USER PROMPT:\n")
        f.write("=" * 80 + "\n")
        for line in user_prompt.split('\n'):
            f.write('\n'.join(wrapper.wrap(line) if line.strip() else [line]) + '\n')
        f.write("\n" + "=" * 80 + "\n")
        f.write("METRICS:\n")
        f.write("=" * 80 + "\n")
        f.write(f"Latency (s):              {metrics['latency_s']:.2f}\n")
        f.write(f"Throughput (tok/s):       {metrics['throughput_tok_s']:.2f}\n")
        f.write(f"TTFT (s):                 {metrics['ttft_s']:.4f}\n")
        f.write(f"Prompt Tokens:            {metrics['prompt_tokens']}\n")
        f.write(f"Output Tokens:            {metrics['output_tokens']}\n")
        f.write(f"Total Tokens:             {metrics['total_tokens']}\n")
        f.write(f"Compression Ratio:        {metrics['compression_ratio']:.2f}\n")
        f.write(f"Word Count:               {metrics['word_count']}\n")
        f.write(f"Avg Chars per Token:      {metrics['avg_chars_per_token']:.2f}\n")
        f.write("\n" + "=" * 80 + "\n")
        f.write("OUTPUT:\n")
        f.write("=" * 80 + "\n")
        for line in output.split('\n'):
            f.write('\n'.join(wrapper.wrap(line) if line.strip() else [line]) + '\n')
        f.write("\n" + "=" * 80 + "\n")
    
    return filepath

def save_output_to_file(model, timestamp, system_prompt, user_prompt, metrics, output):
    """Save output to formatted text file."""
    safe_model = sanitize_model_name(model)
    timestamp_str = datetime.fromisoformat(timestamp).strftime('%Y%m%d_%H%M%S')
    filename = f"{timestamp_str}_{safe_model}.txt"
    filepath = OUTPUT_DIR / filename
    
    wrapper = textwrap.TextWrapper(width=80, break_long_words=False, replace_whitespace=False)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"MODEL: {model}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"TIMESTAMP: {timestamp}\n\n")
        f.write("=" * 80 + "\n")
        f.write("SYSTEM PROMPT:\n")
        f.write("=" * 80 + "\n")
        for line in system_prompt.split('\n'):
            f.write('\n'.join(wrapper.wrap(line) if line.strip() else [line]) + '\n')
        f.write("\n" + "=" * 80 + "\n")
        f.write("USER PROMPT:\n")
        f.write("=" * 80 + "\n")
        for line in user_prompt.split('\n'):
            f.write('\n'.join(wrapper.wrap(line) if line.strip() else [line]) + '\n')
        f.write("\n" + "=" * 80 + "\n")
        f.write("METRICS:\n")
        f.write("=" * 80 + "\n")
        f.write(f"Latency (s):              {metrics['latency_s']:.2f}\n")
        f.write(f"Throughput (tok/s):       {metrics['throughput_tok_s']:.2f}\n")
        f.write(f"TTFT (s):                 {metrics['ttft_s']:.4f}\n")
        f.write(f"Prompt Tokens:            {metrics['prompt_tokens']}\n")
        f.write(f"Output Tokens:            {metrics['output_tokens']}\n")
        f.write(f"Total Tokens:             {metrics['total_tokens']}\n")
        f.write(f"Compression Ratio:        {metrics['compression_ratio']:.2f}\n")
        f.write(f"Word Count:               {metrics['word_count']}\n")
        f.write(f"Avg Chars per Token:      {metrics['avg_chars_per_token']:.2f}\n")
        if metrics.get('temperature') is not None:
            f.write(f"Temperature:              {metrics['temperature']}\n")
        if metrics.get('max_tokens') is not None:
            f.write(f"Max Tokens:               {metrics['max_tokens']}\n")
        f.write("\n" + "=" * 80 + "\n")
        f.write("OUTPUT:\n")
        f.write("=" * 80 + "\n")
        for line in output.split('\n'):
            f.write('\n'.join(wrapper.wrap(line) if line.strip() else [line]) + '\n')
        f.write("\n" + "=" * 80 + "\n")
    
    return filepath

def run_batch_test(models, system_prompts, user_prompts, temperatures, max_tokens_list, progress_callback=None):
    """Execute batch testing across all parameter combinations."""
    batch_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    batch_dir = Path('./batch_outputs') / f"{batch_timestamp}_batch_test"
    batch_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    total_tests = len(models) * len(system_prompts) * len(user_prompts) * len(temperatures) * len(max_tokens_list)
    
    test_num = 0
    for model in models:
        for sys_prompt in system_prompts:
            for usr_prompt in user_prompts:
                for temp in temperatures:
                    for max_tok in max_tokens_list:
                        test_num += 1
                        
                        # Update progress
                        if progress_callback:
                            progress_callback(test_num, total_tests)
                        
                        # Generate test
                        timestamp = datetime.now().isoformat()
                        metrics, error = call_ollama_with_metrics(
                            model, sys_prompt, usr_prompt, temp, max_tok
                        )
                        
                        if error:
                            results.append({
                                'test_num': test_num,
                                'model': model,
                                'temperature': temp,
                                'max_tokens': max_tok,
                                'status': 'error',
                                'error': error
                            })
                        else:
                            # Save individual output
                            safe_model = sanitize_model_name(model)
                            filename = f"test_{test_num:03d}_{safe_model}_temp{temp}_max{max_tok}.txt"
                            filepath = batch_dir / filename
                            
                            wrapper = textwrap.TextWrapper(width=80, break_long_words=False, replace_whitespace=False)
                            
                            with open(filepath, 'w', encoding='utf-8') as f:
                                f.write("=" * 80 + "\n")
                                f.write(f"BATCH TEST #{test_num} / {total_tests}\n")
                                f.write("=" * 80 + "\n")
                                f.write(f"MODEL: {model}\n")
                                f.write(f"TEMPERATURE: {temp}\n")
                                f.write(f"MAX TOKENS: {max_tok}\n")
                                f.write(f"TIMESTAMP: {timestamp}\n\n")
                                f.write("=" * 80 + "\n")
                                f.write("SYSTEM PROMPT:\n")
                                f.write("=" * 80 + "\n")
                                for line in sys_prompt.split('\n'):
                                    f.write('\n'.join(wrapper.wrap(line) if line.strip() else [line]) + '\n')
                                f.write("\n" + "=" * 80 + "\n")
                                f.write("USER PROMPT:\n")
                                f.write("=" * 80 + "\n")
                                for line in usr_prompt.split('\n'):
                                    f.write('\n'.join(wrapper.wrap(line) if line.strip() else [line]) + '\n')
                                f.write("\n" + "=" * 80 + "\n")
                                f.write("METRICS:\n")
                                f.write("=" * 80 + "\n")
                                f.write(f"Latency (s):              {metrics['latency_s']:.2f}\n")
                                f.write(f"Throughput (tok/s):       {metrics['throughput_tok_s']:.2f}\n")
                                f.write(f"TTFT (s):                 {metrics['ttft_s']:.4f}\n")
                                f.write(f"Prompt Tokens:            {metrics['prompt_tokens']}\n")
                                f.write(f"Output Tokens:            {metrics['output_tokens']}\n")
                                f.write(f"Total Tokens:             {metrics['total_tokens']}\n")
                                f.write(f"Compression Ratio:        {metrics['compression_ratio']:.2f}\n")
                                f.write(f"Word Count:               {metrics['word_count']}\n")
                                f.write(f"Avg Chars per Token:      {metrics['avg_chars_per_token']:.2f}\n")
                                f.write("\n" + "=" * 80 + "\n")
                                f.write("OUTPUT:\n")
                                f.write("=" * 80 + "\n")
                                for line in metrics['output'].split('\n'):
                                    f.write('\n'.join(wrapper.wrap(line) if line.strip() else [line]) + '\n')
                                f.write("\n" + "=" * 80 + "\n")
                            
                            results.append({
                                'test_num': test_num,
                                'filename': filename,
                                'model': model,
                                'temperature': temp,
                                'max_tokens': max_tok,
                                'latency_s': metrics['latency_s'],
                                'throughput_tok_s': metrics['throughput_tok_s'],
                                'ttft_s': metrics['ttft_s'],
                                'prompt_tokens': metrics['prompt_tokens'],
                                'output_tokens': metrics['output_tokens'],
                                'total_tokens': metrics['total_tokens'],
                                'word_count': metrics['word_count'],
                                'status': 'success'
                            })
    
    # Generate summary.txt
    summary_path = batch_dir / 'summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("BATCH TEST SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Timestamp: {batch_timestamp}\n")
        f.write(f"Total Tests: {total_tests}\n")
        f.write(f"Successful: {sum(1 for r in results if r['status'] == 'success')}\n")
        f.write(f"Failed: {sum(1 for r in results if r['status'] == 'error')}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("TEST VARIABLES\n")
        f.write("=" * 80 + "\n")
        f.write(f"Models: {', '.join(models)}\n")
        f.write(f"Temperatures: {', '.join(map(str, temperatures))}\n")
        f.write(f"Max Tokens: {', '.join(map(str, max_tokens_list))}\n")
        f.write(f"System Prompts: {len(system_prompts)}\n")
        f.write(f"User Prompts: {len(user_prompts)}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("RESULTS TABLE\n")
        f.write("=" * 80 + "\n\n")
        
        # Header
        f.write(f"{'Test':<6} {'Model':<25} {'Temp':<6} {'MaxTok':<8} {'Lat(s)':<8} {'Thru':<8} {'Tokens':<8} {'Status':<10}\n")
        f.write("-" * 80 + "\n")
        
        # Results
        for r in results:
            if r['status'] == 'success':
                f.write(f"{r['test_num']:<6} {r['model'][:24]:<25} {r['temperature']:<6.2f} {r['max_tokens']:<8} "
                       f"{r['latency_s']:<8.2f} {r['throughput_tok_s']:<8.1f} {r['total_tokens']:<8} {r['status']:<10}\n")
            else:
                f.write(f"{r['test_num']:<6} {r['model'][:24]:<25} {r['temperature']:<6.2f} {r['max_tokens']:<8} "
                       f"{'N/A':<8} {'N/A':<8} {'N/A':<8} {r['status']:<10}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("STATISTICS\n")
        f.write("=" * 80 + "\n\n")
        
        successful = [r for r in results if r['status'] == 'success']
        if successful:
            avg_latency = sum(r['latency_s'] for r in successful) / len(successful)
            avg_throughput = sum(r['throughput_tok_s'] for r in successful) / len(successful)
            avg_tokens = sum(r['total_tokens'] for r in successful) / len(successful)
            
            f.write(f"Average Latency: {avg_latency:.2f}s\n")
            f.write(f"Average Throughput: {avg_throughput:.2f} tok/s\n")
            f.write(f"Average Total Tokens: {avg_tokens:.0f}\n")
    
    return batch_dir, results

def load_batch_test_results():
    """Load all batch test summaries."""
    batch_dir = Path('./batch_outputs')
    if not batch_dir.exists():
        return []
    
    batch_tests = []
    for test_dir in sorted(batch_dir.glob('*_batch_test'), reverse=True):
        summary_file = test_dir / 'summary.txt'
        if summary_file.exists():
            with open(summary_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract key info
            timestamp_match = re.search(r'Timestamp: (\d+)', content)
            total_match = re.search(r'Total Tests: (\d+)', content)
            success_match = re.search(r'Successful: (\d+)', content)
            
            batch_tests.append({
                'directory': test_dir.name,
                'path': str(test_dir),
                'timestamp': timestamp_match.group(1) if timestamp_match else 'Unknown',
                'total_tests': int(total_match.group(1)) if total_match else 0,
                'successful': int(success_match.group(1)) if success_match else 0,
                'summary_content': content
            })
    
    return batch_tests

def load_output_history():
    """Load all saved outputs and extract metadata."""
    history = []
    
    for filepath in sorted(OUTPUT_DIR.glob("*.txt"), reverse=True):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract metadata
            model_match = re.search(r'MODEL: (.+)', content)
            timestamp_match = re.search(r'TIMESTAMP: (.+)', content)
            latency_match = re.search(r'Latency \(s\):\s+(\d+\.?\d*)', content)
            throughput_match = re.search(r'Throughput \(tok/s\):\s+(\d+\.?\d*)', content)
            prompt_tokens_match = re.search(r'Prompt Tokens:\s+(\d+)', content)
            output_tokens_match = re.search(r'Output Tokens:\s+(\d+)', content)
            total_tokens_match = re.search(r'Total Tokens:\s+(\d+)', content)
            word_count_match = re.search(r'Word Count:\s+(\d+)', content)
            
            user_prompt_match = re.search(r'USER PROMPT:\n={80}\n(.*?)\n={80}', content, re.DOTALL)
            output_match = re.search(r'OUTPUT:\n={80}\n(.*?)\n={80}', content, re.DOTALL)
            
            if all([model_match, timestamp_match, latency_match]):
                history.append({
                    'filename': filepath.name,
                    'model': model_match.group(1),
                    'timestamp': timestamp_match.group(1),
                    'latency_s': float(latency_match.group(1)),
                    'throughput_tok_s': float(throughput_match.group(1)) if throughput_match else 0,
                    'prompt_tokens': int(prompt_tokens_match.group(1)) if prompt_tokens_match else 0,
                    'output_tokens': int(output_tokens_match.group(1)) if output_tokens_match else 0,
                    'total_tokens': int(total_tokens_match.group(1)) if total_tokens_match else 0,
                    'word_count': int(word_count_match.group(1)) if word_count_match else 0,
                    'user_prompt': user_prompt_match.group(1).strip() if user_prompt_match else '',
                    'output': output_match.group(1).strip() if output_match else ''
                })
        except Exception:
            continue
    
    return history

# ============================================================================
# STREAMLIT APP
# ============================================================================

st.set_page_config(page_title="Smart Home Sensor Analysis", layout="wide")

# Auto-load data_full.csv on startup
if not st.session_state.data_loaded and Path('data_full.csv').exists():
    try:
        df = load_data_from_file('data_full.csv')
        st.session_state.current_data = df
        st.session_state.data_loaded = True
    except Exception as e:
        st.session_state.load_error = str(e)

st.title("Smart Home Multi-Sensor Analysis & LLM Testing")

tab1, tab2, tab3, tab4 = st.tabs(["Data Explorer", "Multi-Sensor Analysis", "LLM Testing", "History & Compare"])

# ============================================================================
# TAB 1: DATA EXPLORER
# ============================================================================
with tab1:
    st.subheader("Data Explorer")
    
    # Show data status
    if st.session_state.data_loaded and st.session_state.current_data is not None:
        df = st.session_state.current_data
        st.success(f"Loaded {len(df):,} records from {df['timestamp'].min()} to {df['timestamp'].max()}")
    elif hasattr(st.session_state, 'load_error'):
        st.error(f"Error loading data_full.csv: {st.session_state.load_error}")
        st.stop()
    else:
        st.error("data_full.csv not found in current directory")
        st.stop()
    
    if st.session_state.data_loaded and st.session_state.current_data is not None:
        df = st.session_state.current_data
        
        st.divider()
        st.subheader("Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Records", f"{len(df):,}")
        col2.metric("Sensor Types", df['sensor_type'].nunique())
        col3.metric("Rooms", df['room_name'].nunique())
        col4.metric("Date Range", f"{(df['timestamp'].max() - df['timestamp'].min()).days}d")
        
        with st.expander("Sensor Type Breakdown"):
            sensor_counts = df.groupby('sensor_type').size().reset_index(name='count')
            sensor_counts = sensor_counts.sort_values('count', ascending=False)
            st.dataframe(sensor_counts, use_container_width=True)
        
        st.divider()
        st.subheader("Time Window Selection")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            min_date = df['timestamp'].min().date()
            max_date = df['timestamp'].max().date()
            start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
        
        with col2:
            start_time = st.time_input("Start Time", value=datetime.min.time())
        
        with col3:
            window_hours = st.selectbox("Duration", [24, 48, 168, 720], 
                                       format_func=lambda x: f"{x}h ({x//24}d)")
        
        start_dt = datetime.combine(start_date, start_time)
        end_dt = start_dt + timedelta(hours=window_hours)
        
        st.divider()
        st.subheader("Filters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            all_rooms = sorted(df['room_name'].unique().tolist())
            room_filter_type = st.radio("Room Filter", ["All Rooms", "Specific Rooms"], horizontal=True)
            
            if room_filter_type == "Specific Rooms":
                selected_rooms = st.multiselect("Select Rooms", all_rooms, default=all_rooms[:3] if len(all_rooms) >= 3 else all_rooms)
            else:
                selected_rooms = ['All Rooms']
        
        with col2:
            all_sensors = sorted(df['sensor_type'].unique().tolist())
            sensor_filter_type = st.radio("Sensor Filter", ["All Sensors", "Specific Sensors"], horizontal=True)
            
            if sensor_filter_type == "Specific Sensors":
                selected_sensors = st.multiselect("Select Sensors", all_sensors, default=all_sensors)
            else:
                selected_sensors = ['All Sensors']
        
        st.divider()
        num_segments = st.slider("Number of Time Segments", min_value=1, max_value=24, value=24, 
                                help="Divide the time window into this many segments for analysis")
        
        if st.button("Extract & Filter Window", type="primary"):
            with st.spinner("Processing..."):
                window = extract_window(df, window_hours, start_dt)
                filtered = filter_by_rooms(window, selected_rooms)
                filtered = filter_by_sensors(filtered, selected_sensors)
                st.session_state.window_data = filtered
                
                # Generate segment summary
                segment_summary = generate_segment_summary(filtered, start_dt, end_dt, num_segments)
                st.session_state.segment_summary = segment_summary
                
                st.success(f"Extracted {len(filtered):,} records")
        
        if st.session_state.window_data is not None:
            window_df = st.session_state.window_data
            
            st.divider()
            st.subheader("Window Summary")
            
            summary = generate_multi_sensor_summary(window_df)
            
            if summary:
                for sensor_type, stats in summary.items():
                    with st.expander(f"{sensor_type} Statistics"):
                        config = SENSOR_CONFIGS.get(sensor_type, {'type': 'continuous', 'unit': ''})
                        
                        if config['type'] == 'binary':
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Total Readings", f"{stats['total_readings']:,}")
                            col2.metric("Activations", f"{stats['activations']:,}")
                            col3.metric("Activation Rate", f"{stats['activation_rate']:.1%}")
                            col4.metric("Rooms", stats['rooms'])
                        else:
                            col1, col2, col3, col4, col5, col6 = st.columns(6)
                            col1.metric("Readings", f"{stats['total_readings']:,}")
                            col2.metric("Mean", f"{stats['mean']:.2f}")
                            col3.metric("Std Dev", f"{stats['std']:.2f}")
                            col4.metric("Min", f"{stats['min']:.2f}")
                            col5.metric("Max", f"{stats['max']:.2f}")
                            col6.metric("Rooms", stats['rooms'])
            
            # Display segment summary
            if hasattr(st.session_state, 'segment_summary') and st.session_state.segment_summary:
                st.divider()
                st.subheader("Segmented Analysis")
                st.text_area("Sensor Data by Time Segment", 
                           value=st.session_state.segment_summary, 
                           height=400,
                           help="Activations for binary sensors, averages for continuous sensors")

# ============================================================================
# TAB 2: MULTI-SENSOR ANALYSIS
# ============================================================================
with tab2:
    st.subheader("Advanced Visualization & Analysis")
    
    if st.session_state.window_data is None:
        st.warning("Please extract a time window in the Data Explorer tab first.")
    else:
        window_df = st.session_state.window_data
        available_sensors = window_df['sensor_type'].unique().tolist()
        
        viz_category = st.selectbox("Analysis Category", 
                                    ["Binary Sensors (Motion/Light)", 
                                     "Continuous Sensors (Temp/Humidity/Power)",
                                     "Multi-Sensor Correlation",
                                     "Anomaly Detection"])
        
        if viz_category == "Binary Sensors (Motion/Light)":
            binary_sensors = [s for s in available_sensors if SENSOR_CONFIGS.get(s, {}).get('type') == 'binary']
            
            if not binary_sensors:
                st.warning("No binary sensors in filtered data.")
            else:
                selected_binary = st.selectbox("Select Binary Sensor", binary_sensors)
                
                viz_type = st.radio("Visualization Type", 
                                   ["Scatter Timeline", "Heatmap", "Distribution Bar"], 
                                   horizontal=True)
                
                if st.button("Generate Binary Visualization"):
                    with st.spinner("Creating plot..."):
                        if viz_type == "Scatter Timeline":
                            fig = plot_binary_timeline(window_df, selected_binary)
                        elif viz_type == "Heatmap":
                            fig = plot_activity_heatmap(window_df, selected_binary)
                        else:
                            # Bar chart
                            sensor_data = window_df[(window_df['sensor_type'] == selected_binary) & (window_df['sensor_value'] == 1)]
                            room_counts = sensor_data.groupby('room_name').size().reset_index(name='count').sort_values('count', ascending=True)
                            fig = px.bar(room_counts, x='count', y='room_name', orientation='h',
                                        title=f'{selected_binary} Events by Room',
                                        labels={'count': 'Events', 'room_name': 'Room'},
                                        height=500)
                        
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("No data available for this visualization.")
        
        elif viz_category == "Continuous Sensors (Temp/Humidity/Power)":
            continuous_sensors = [s for s in available_sensors if SENSOR_CONFIGS.get(s, {}).get('type') == 'continuous']
            
            if not continuous_sensors:
                st.warning("No continuous sensors in filtered data.")
            else:
                selected_continuous = st.selectbox("Select Continuous Sensor", continuous_sensors)
                
                viz_type = st.radio("Visualization Type",
                                   ["Time Series", "Distribution (Box)", "Heatmap (Hourly Average)"],
                                   horizontal=True)
                
                if st.button("Generate Continuous Visualization"):
                    with st.spinner("Creating plot..."):
                        if viz_type == "Time Series":
                            fig = plot_continuous_timeseries(window_df, selected_continuous)
                        elif viz_type == "Distribution (Box)":
                            fig = plot_continuous_distribution(window_df, selected_continuous)
                        else:
                            fig = plot_continuous_heatmap(window_df, selected_continuous)
                        
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("No data available for this visualization.")
        
        elif viz_category == "Multi-Sensor Correlation":
            st.markdown("**Overlay & Correlation Analysis**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### Dual-Axis Overlay")
                if len(available_sensors) >= 2:
                    overlay_sensor1 = st.selectbox("Primary Sensor (Left Y-axis)", available_sensors, key='overlay1')
                    overlay_sensor2 = st.selectbox("Secondary Sensor (Right Y-axis)", 
                                                   [s for s in available_sensors if s != overlay_sensor1], 
                                                   key='overlay2')
                    
                    if st.button("Generate Overlay"):
                        with st.spinner("Creating overlay..."):
                            fig = plot_multi_sensor_overlay(window_df, [overlay_sensor1, overlay_sensor2])
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Need at least 2 sensor types.")
            
            with col2:
                st.markdown("##### Correlation Matrix")
                if st.button("Generate Correlation Matrix"):
                    with st.spinner("Computing correlations..."):
                        fig = plot_correlation_matrix(window_df)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("Need at least 2 continuous sensors.")
            
            st.divider()
            st.markdown("##### Environmental Profile")
            if 'Temperature' in available_sensors and 'Humidity' in available_sensors:
                if st.button("Generate Room Climate Profile"):
                    with st.spinner("Creating profile..."):
                        fig = plot_room_environmental_profile(window_df)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Temperature and Humidity sensors required for this plot.")
        
        else:  # Anomaly Detection
            continuous_sensors = [s for s in available_sensors if SENSOR_CONFIGS.get(s, {}).get('type') == 'continuous']
            
            if not continuous_sensors:
                st.warning("Anomaly detection requires continuous sensors.")
            else:
                selected_sensor = st.selectbox("Select Sensor for Anomaly Detection", continuous_sensors)
                sigma_threshold = st.slider("Sigma Threshold", 1.0, 3.0, 2.0, 0.5)
                
                if st.button("Detect Anomalies"):
                    with st.spinner("Analyzing anomalies..."):
                        result = detect_anomalies_continuous(window_df, selected_sensor, sigma_threshold)
                        
                        if result:
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Total Readings", f"{result['total_readings']:,}")
                            col2.metric("Anomalies", f"{result['anomalies']:,}")
                            col3.metric("Anomaly Rate", f"{result['anomaly_rate']:.2%}")
                            col4.metric("Mean ± Std", f"{result['mean']:.1f} ± {result['std']:.1f}")
                            
                            fig = plot_anomalies(window_df, selected_sensor, sigma_threshold)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                            
                            if len(result['anomaly_data']) > 0:
                                with st.expander("View Anomalous Readings"):
                                    st.dataframe(result['anomaly_data'][['timestamp', 'room_name', 'sensor_value', 'z_score']], 
                                               use_container_width=True)
                        else:
                            st.warning("No data available.")

# ============================================================================
# TAB 3: LLM TESTING
# ============================================================================
with tab3:
    st.subheader("LLM Prompt Testing")
    
    # Mode selection
    test_mode = st.radio("Test Mode", ["Single Test", "Batch Test"], horizontal=True)
    
    st.divider()
    
    if test_mode == "Single Test":
        # Original single test interface
        col1, col2 = st.columns([1, 3])
        
        with col1:
            if st.button("Refresh Models"):
                st.session_state.models = get_available_models()
            
            if not st.session_state.models:
                st.session_state.models = get_available_models()
            
            if st.session_state.models:
                model = st.selectbox("Select Model", st.session_state.models)
                st.success(f"{len(st.session_state.models)} models available")
            else:
                st.error("No Ollama models found")
                model = None
        
        with col2:
            system_prompt = st.text_area("System Prompt", 
                                         value="You are a helpful assistant analyzing smart home sensor data.",
                                         height=100)
        
        if st.session_state.window_data is not None:
            col_a, col_b = st.columns(2)
            
            with col_a:
                if st.button("Insert Window Summary"):
                    summary = generate_multi_sensor_summary(st.session_state.window_data)
                    summary_text = "SENSOR DATA SUMMARY:\n\n"
                    for sensor_type, stats in summary.items():
                        config = SENSOR_CONFIGS.get(sensor_type, {'type': 'continuous', 'unit': ''})
                        summary_text += f"{sensor_type}:\n"
                        if config['type'] == 'binary':
                            summary_text += f"  - Activations: {stats['activations']} / {stats['total_readings']} ({stats['activation_rate']:.1%})\n"
                        else:
                            summary_text += f"  - Mean: {stats['mean']:.2f} {config['unit']}\n"
                            summary_text += f"  - Range: [{stats['min']:.2f}, {stats['max']:.2f}] {config['unit']}\n"
                            summary_text += f"  - Std Dev: {stats['std']:.2f}\n"
                        summary_text += f"  - Rooms: {stats['rooms']}\n\n"
                    
                    st.session_state.quick_insert = summary_text
            
            with col_b:
                if st.button("Insert Segment Analysis"):
                    if hasattr(st.session_state, 'segment_summary') and st.session_state.segment_summary:
                        st.session_state.quick_insert = st.session_state.segment_summary
                    else:
                        st.warning("No segment analysis available. Extract window data first.")
        
        user_prompt = st.text_area("User Prompt", height=300,
                                  value=st.session_state.get('quick_insert', ''))
        
        if st.button("Generate", type="primary"):
            if user_prompt and model:
                with st.spinner(f"Generating with {model}..."):
                    metrics, error = call_ollama_with_metrics(model, system_prompt, user_prompt)
                    
                    if error:
                        st.error(error)
                    else:
                        timestamp = datetime.now().isoformat()
                        filepath = save_output_to_file(
                            model, timestamp, system_prompt, user_prompt,
                            metrics, metrics['output']
                        )
                        
                        st.success(f"Saved: {filepath.name}")
                        
                        st.subheader("Response")
                        st.markdown(metrics['output'])
                        
                        st.divider()
                        st.subheader("Performance Metrics")
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Latency", f"{metrics['latency_s']:.2f}s")
                        col2.metric("Throughput", f"{metrics['throughput_tok_s']:.1f} tok/s")
                        col3.metric("TTFT", f"{metrics['ttft_s']:.3f}s")
                        col4.metric("Total Tokens", metrics['total_tokens'])
                        
                        col5, col6, col7, col8 = st.columns(4)
                        col5.metric("Input Tokens", metrics['prompt_tokens'])
                        col6.metric("Output Tokens", metrics['output_tokens'])
                        col7.metric("Word Count", metrics['word_count'])
                        col8.metric("Compression", f"{metrics['compression_ratio']:.2f}")
            else:
                st.warning("Please provide user prompt and select a model.")
    
    else:  # Batch Test Mode
        st.subheader("Batch Test Configuration")
        
        # Model selection
        if not st.session_state.models:
            st.session_state.models = get_available_models()
        
        if st.session_state.models:
            selected_models = st.multiselect("Select Models", st.session_state.models, 
                                            default=[st.session_state.models[0]] if st.session_state.models else [])
        else:
            st.error("No Ollama models found")
            selected_models = []
        
        st.divider()
        
        # System prompts
        st.subheader("System Prompts")
        if 'batch_system_prompts' not in st.session_state:
            st.session_state.batch_system_prompts = ["You are a helpful assistant analyzing smart home sensor data."]
        
        for i, prompt in enumerate(st.session_state.batch_system_prompts):
            col1, col2 = st.columns([9, 1])
            with col1:
                st.session_state.batch_system_prompts[i] = st.text_area(
                    f"System Prompt {i+1}", value=prompt, height=80, key=f"sys_{i}"
                )
            with col2:
                if len(st.session_state.batch_system_prompts) > 1:
                    if st.button("✕", key=f"del_sys_{i}"):
                        st.session_state.batch_system_prompts.pop(i)
                        st.rerun()
        
        if st.button("Add System Prompt"):
            st.session_state.batch_system_prompts.append("")
            st.rerun()
        
        st.divider()
        
        # User prompts
        st.subheader("User Prompts")
        if 'batch_user_prompts' not in st.session_state:
            st.session_state.batch_user_prompts = [""]
        
        for i, prompt in enumerate(st.session_state.batch_user_prompts):
            col1, col2 = st.columns([9, 1])
            with col1:
                st.session_state.batch_user_prompts[i] = st.text_area(
                    f"User Prompt {i+1}", value=prompt, height=120, key=f"usr_{i}"
                )
            with col2:
                if len(st.session_state.batch_user_prompts) > 1:
                    if st.button("✕", key=f"del_usr_{i}"):
                        st.session_state.batch_user_prompts.pop(i)
                        st.rerun()
        
        if st.button("Add User Prompt"):
            st.session_state.batch_user_prompts.append("")
            st.rerun()
        
        # Quick insert buttons for batch prompts
        if st.session_state.window_data is not None:
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Insert Window Summary to Last Prompt"):
                    if st.session_state.batch_user_prompts:
                        summary = generate_multi_sensor_summary(st.session_state.window_data)
                        summary_text = "SENSOR DATA SUMMARY:\n\n"
                        for sensor_type, stats in summary.items():
                            config = SENSOR_CONFIGS.get(sensor_type, {'type': 'continuous', 'unit': ''})
                            summary_text += f"{sensor_type}:\n"
                            if config['type'] == 'binary':
                                summary_text += f"  - Activations: {stats['activations']} / {stats['total_readings']} ({stats['activation_rate']:.1%})\n"
                            else:
                                summary_text += f"  - Mean: {stats['mean']:.2f} {config['unit']}\n"
                                summary_text += f"  - Range: [{stats['min']:.2f}, {stats['max']:.2f}] {config['unit']}\n"
                                summary_text += f"  - Std Dev: {stats['std']:.2f}\n"
                            summary_text += f"  - Rooms: {stats['rooms']}\n\n"
                        st.session_state.batch_user_prompts[-1] = summary_text
                        st.rerun()
            
            with col_b:
                if st.button("Insert Segment Analysis to Last Prompt"):
                    if hasattr(st.session_state, 'segment_summary') and st.session_state.segment_summary:
                        if st.session_state.batch_user_prompts:
                            st.session_state.batch_user_prompts[-1] = st.session_state.segment_summary
                            st.rerun()
                    else:
                        st.warning("No segment analysis available.")
        
        st.divider()
        
        # Hyperparameters
        st.subheader("Hyperparameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Temperature Values**")
            temp_input_method = st.radio("Temperature Input", ["Slider", "Manual"], horizontal=True, key="temp_method")
            
            if temp_input_method == "Slider":
                temp_min = st.slider("Min Temperature", 0.0, 2.0, 0.7, 0.1, key="temp_min")
                temp_max = st.slider("Max Temperature", 0.0, 2.0, 1.0, 0.1, key="temp_max")
                temp_step = st.number_input("Step", 0.1, 1.0, 0.1, 0.1, key="temp_step")
                temperatures = [round(t, 2) for t in np.arange(temp_min, temp_max + temp_step/2, temp_step)]
            else:
                temp_values = st.text_input("Temperatures (comma-separated)", "0.7, 1.0", key="temp_manual")
                temperatures = [float(x.strip()) for x in temp_values.split(',') if x.strip()]
            
            st.info(f"Testing {len(temperatures)} temperature(s): {temperatures}")
        
        with col2:
            st.markdown("**Max Tokens Values**")
            use_max_tokens = st.checkbox("Limit max tokens", value=False)
            
            if use_max_tokens:
                max_tok_values = st.text_input("Max Tokens (comma-separated)", "256, 512", key="max_tok_manual")
                max_tokens_list = [int(x.strip()) for x in max_tok_values.split(',') if x.strip()]
            else:
                max_tokens_list = [None]
            
            st.info(f"Testing {len(max_tokens_list)} max_token setting(s)")
        
        st.divider()
        
        # Calculate total tests
        total_tests = (len(selected_models) * len(st.session_state.batch_system_prompts) * 
                      len(st.session_state.batch_user_prompts) * len(temperatures) * len(max_tokens_list))
        
        st.info(f"**Total tests to run: {total_tests}**")
        
        if st.button("Run Batch Test", type="primary"):
            if not selected_models:
                st.error("Please select at least one model")
            elif not any(st.session_state.batch_system_prompts):
                st.error("Please provide at least one system prompt")
            elif not any(st.session_state.batch_user_prompts):
                st.error("Please provide at least one user prompt")
            else:
                with st.spinner(f"Running {total_tests} tests..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Progress callback
                    def update_progress(current, total):
                        progress_bar.progress(current / total)
                        status_text.text(f"Running test {current} of {total}...")
                    
                    # Run batch test
                    batch_dir, results = run_batch_test(
                        selected_models,
                        st.session_state.batch_system_prompts,
                        st.session_state.batch_user_prompts,
                        temperatures,
                        max_tokens_list,
                        progress_callback=update_progress
                    )
                    
                    progress_bar.progress(1.0)
                    status_text.text("Batch test complete!")
                    
                    st.success(f"Batch test complete! Results saved to: {batch_dir}")
                    
                    # Show summary
                    successful = sum(1 for r in results if r['status'] == 'success')
                    failed = sum(1 for r in results if r['status'] == 'error')
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Tests", total_tests)
                    col2.metric("Successful", successful)
                    col3.metric("Failed", failed)

# ============================================================================
# TAB 4: HISTORY & COMPARE
# ============================================================================
with tab4:
    st.subheader("Generation History & Comparison")
    
    # History type selection
    history_type = st.radio("View", ["Single Tests", "Batch Tests"], horizontal=True)
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("Refresh History"):
            st.rerun()
    
    if history_type == "Batch Tests":
        # Batch test results
        batch_tests = load_batch_test_results()
        
        if not batch_tests:
            st.info("No batch test results found in ./batch_outputs/")
        else:
            st.divider()
            st.subheader(f"Batch Tests ({len(batch_tests)})")
            
            for batch in batch_tests:
                with st.expander(f"Batch Test: {batch['directory']}"):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Tests", batch['total_tests'])
                    col2.metric("Successful", batch['successful'])
                    col3.metric("Failed", batch['total_tests'] - batch['successful'])
                    
                    st.divider()
                    st.subheader("Summary Content")
                    st.text(batch['summary_content'])
                    
                    st.divider()
                    st.info(f"Full results available at: {batch['path']}")
    
    else:
        # Single test history (original)
        history = load_output_history()
        
        if not history:
            st.info("No generation history found in ./outputs/")
        else:
            df_hist = pd.DataFrame(history)
        
        st.divider()
        st.subheader("Filters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            all_models = sorted(df_hist['model'].unique().tolist())
            model_filter = st.multiselect("Filter by Model", all_models, default=all_models)
        
        with col2:
            min_latency = st.number_input("Min Latency (s)", min_value=0.0, value=0.0, step=0.1)
            max_latency = st.number_input("Max Latency (s)", min_value=0.0, value=1000.0, step=1.0)
        
        with col3:
            search_term = st.text_input("Search in Prompt/Output", "")
        
        # Apply filters
        filtered_hist = df_hist[df_hist['model'].isin(model_filter)]
        filtered_hist = filtered_hist[(filtered_hist['latency_s'] >= min_latency) & 
                                     (filtered_hist['latency_s'] <= max_latency)]
        
        if search_term:
            filtered_hist = filtered_hist[
                filtered_hist['user_prompt'].str.contains(search_term, case=False, na=False) |
                filtered_hist['output'].str.contains(search_term, case=False, na=False)
            ]
        
        st.divider()
        st.subheader(f"Summary ({len(filtered_hist)} runs)")
        
        if len(filtered_hist) > 0:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Runs", len(filtered_hist))
            col2.metric("Avg Latency", f"{filtered_hist['latency_s'].mean():.2f}s")
            col3.metric("Avg Throughput", f"{filtered_hist['throughput_tok_s'].mean():.1f} tok/s")
            col4.metric("Avg Tokens", f"{filtered_hist['total_tokens'].mean():.0f}")
            
            # Performance visualizations
            st.divider()
            st.subheader("Performance Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_latency = px.box(filtered_hist, x='model', y='latency_s',
                                    title='Latency Distribution by Model',
                                    labels={'latency_s': 'Latency (s)', 'model': 'Model'},
                                    height=400)
                st.plotly_chart(fig_latency, use_container_width=True)
            
            with col2:
                fig_throughput = px.box(filtered_hist, x='model', y='throughput_tok_s',
                                       title='Throughput Distribution by Model',
                                       labels={'throughput_tok_s': 'Throughput (tok/s)', 'model': 'Model'},
                                       height=400)
                st.plotly_chart(fig_throughput, use_container_width=True)
            
            # Detailed table
            st.divider()
            st.subheader("Run Details")
            
            display_cols = ['filename', 'model', 'latency_s', 'throughput_tok_s',
                          'prompt_tokens', 'output_tokens', 'total_tokens', 'word_count']
            
            # Make table sortable
            st.dataframe(
                filtered_hist[display_cols].sort_values('latency_s'),
                use_container_width=True,
                height=400
            )
            
            # Side-by-side comparison
            st.divider()
            st.subheader("Side-by-Side Comparison")
            
            selected_files = st.multiselect(
                "Select runs to compare (up to 3)",
                filtered_hist['filename'].tolist(),
                max_selections=3
            )
            
            if selected_files:
                compare_data = filtered_hist[filtered_hist['filename'].isin(selected_files)]
                
                cols = st.columns(len(selected_files))
                
                for i, (_, row) in enumerate(compare_data.iterrows()):
                    with cols[i]:
                        st.markdown(f"**{row['model']}**")
                        st.caption(f"File: {row['filename']}")
                        
                        st.metric("Latency", f"{row['latency_s']:.2f}s")
                        st.metric("Throughput", f"{row['throughput_tok_s']:.1f} tok/s")
                        st.metric("Tokens", row['total_tokens'])
                        
                        with st.expander("User Prompt"):
                            st.text_area("", row['user_prompt'], key=f"cmp_prompt_{i}", height=150, disabled=True)
                        
                        with st.expander("Output"):
                            st.text_area("", row['output'], key=f"cmp_output_{i}", height=200, disabled=True)
            
            # Full logs expandable
            st.divider()
            with st.expander("Full Generation Logs"):
                for i, record in filtered_hist.iterrows():
                    st.markdown(f"### Run: {record['filename']}")
                    st.markdown(f"**Model:** {record['model']} | **Time:** {record['timestamp']}")
                    
                    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                    mcol1.metric("Latency", f"{record['latency_s']:.2f}s")
                    mcol2.metric("Throughput", f"{record['throughput_tok_s']:.1f} tok/s")
                    mcol3.metric("Total Tokens", record['total_tokens'])
                    mcol4.metric("Words", record['word_count'])
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.text_area("User Prompt", record['user_prompt'], key=f"log_p_{i}", height=150, disabled=True)
                    with col2:
                        st.text_area("Output", record['output'], key=f"log_o_{i}", height=150, disabled=True)
                    
                    st.divider()
        else:
            st.warning("No runs match the current filters.")