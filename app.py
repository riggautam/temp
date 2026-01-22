"""Simple Streamlit + Ollama Interface - Basic model selection and prompt testing"""
import streamlit as st
import subprocess
import warnings
import os
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta, time
warnings.filterwarnings('ignore')

# Import data formatting functions
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_formatter import extract_window, format_combined

def load_data_from_file(filepath):
    """Load data from CSV file path."""
    df = pd.read_csv(filepath)
    # Parse timestamp
    df['timestamp'] = pd.to_datetime(df['local_timestamp'], format='mixed', dayfirst=True)
    # Clean string columns
    for col in ['sensor_name', 'room_name', 'sensor_type']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    # Ensure numeric values
    df['sensor_value'] = pd.to_numeric(df['sensor_value'], errors='coerce')
    # Filter excluded sensors
    EXCLUDED_SENSORS = ['Accelerometer', 'Tamper']
    df = df[~df['sensor_type'].isin(EXCLUDED_SENSORS)]
    # Sort chronologically
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df

def get_data_date_range(filepath):
    """Lightweight function to get only min/max dates from CSV."""
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['local_timestamp'], format='mixed', dayfirst=True)
    return df['timestamp'].min(), df['timestamp'].max()

def plot_motion_activations(df):
    """Create scatter plot of motion activations by room over time."""
    # Filter to motion activations only (value == 1)
    motion = df[(df['sensor_type'] == 'Motion') & (df['sensor_value'] == 1)].copy()
    
    if len(motion) == 0:
        return None
    
    # Count activations per room for legend
    room_counts = motion.groupby('room_name').size().to_dict()
    
    # Add room labels with counts
    motion['room_label'] = motion['room_name'].apply(lambda x: f"{x} ({room_counts[x]})")
    
    # Create scatter plot
    fig = px.scatter(
        motion,
        x='timestamp',
        y='room_name',
        color='room_label',
        title='Motion Activations by Room',
        labels={'timestamp': 'Time', 'room_name': 'Room'},
        height=500
    )
    
    # Update layout
    fig.update_traces(marker=dict(size=8, opacity=0.7))
    fig.update_layout(
        xaxis_title='Time',
        yaxis_title='',
        showlegend=True,
        legend_title_text='Room',
        hovermode='closest'
    )
    
    # Format x-axis based on time range (correct method is update_xaxes)
    fig.update_xaxes(tickformat='%H:%M' if len(motion) > 0 else '%Y-%m-%d %H:%M')
    
    return fig

def segment_and_analyze(df, start_dt, end_dt):
    """
    Split time range into 7 segments and calculate statistics with anomaly detection.
    Returns formatted narrative summary.
    """
    # Filter to motion activations only
    motion = df[(df['sensor_type'] == 'Motion') & (df['sensor_value'] == 1)].copy()
    
    if len(motion) == 0:
        return "No motion data available for analysis."
    
    # Calculate segment duration
    total_duration = (end_dt - start_dt).total_seconds()
    segment_duration = total_duration / 7
    
    segments = []
    for i in range(7):
        seg_start = start_dt + timedelta(seconds=i * segment_duration)
        seg_end = start_dt + timedelta(seconds=(i + 1) * segment_duration)
        
        # Count activations in this segment
        seg_data = motion[(motion['timestamp'] >= seg_start) & (motion['timestamp'] < seg_end)]
        count = len(seg_data)
        
        segments.append({
            'segment': i + 1,
            'start': seg_start,
            'end': seg_end,
            'count': count
        })
    
    # Calculate statistics
    counts = [s['count'] for s in segments]
    mean_count = sum(counts) / len(counts)
    
    # Calculate standard deviation manually
    variance = sum((c - mean_count) ** 2 for c in counts) / len(counts)
    std_count = variance ** 0.5
    
    # Identify anomalies (2 std from mean)
    threshold_high = mean_count + (2 * std_count)
    threshold_low = mean_count - (2 * std_count)
    
    # Build narrative
    lines = ["TEMPORAL ANALYSIS SUMMARY:"]
    lines.append(f"Period: {start_dt.strftime('%Y-%m-%d %H:%M UTC')} to {end_dt.strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append(f"Analysis divided into 7 equal segments for trend identification.\n")
    
    lines.append("ACTIVITY METRICS:")
    lines.append(f"Average activity per segment: {mean_count:.1f} motion events (±{std_count:.1f})")
    lines.append(f"Total motion events: {sum(counts)}")
    lines.append(f"Range: {min(counts)} to {max(counts)} events per segment\n")
    
    # Segment breakdown with anomalies
    lines.append("SEGMENT BREAKDOWN:")
    for seg in segments:
        anomaly_flag = ""
        if seg['count'] > threshold_high:
            anomaly_flag = " [ANOMALY: Elevated Activity]"
        elif seg['count'] < threshold_low and seg['count'] > 0:
            anomaly_flag = " [ANOMALY: Reduced Activity]"
        elif seg['count'] == 0:
            anomaly_flag = " [ANOMALY: No Activity Detected]"
        
        lines.append(f"Segment {seg['segment']}: {seg['start'].strftime('%H:%M')} - {seg['end'].strftime('%H:%M')} | "
                    f"{seg['count']} events{anomaly_flag}")
    
    # Anomaly summary
    anomalies = [s for s in segments if s['count'] > threshold_high or s['count'] < threshold_low]
    if anomalies:
        lines.append(f"\nANOMALY DETECTION: {len(anomalies)} segment(s) outside normal range (±2 SD)")
        for seg in anomalies:
            if seg['count'] > threshold_high:
                lines.append(f"  - Segment {seg['segment']}: Elevated ({seg['count']} events, "
                           f"{((seg['count'] - mean_count) / std_count):.1f} SD above mean)")
            elif seg['count'] < threshold_low:
                lines.append(f"  - Segment {seg['segment']}: Reduced ({seg['count']} events, "
                           f"{((mean_count - seg['count']) / std_count):.1f} SD below mean)")
    else:
        lines.append("\nNo significant anomalies detected. Activity levels remain within normal variance.")
    
    return '\n'.join(lines)

def get_ollama_models():
    """Get list of available Ollama models."""
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return []
        lines = result.stdout.strip().split('\n')[1:]
        models = []
        for line in lines:
            if line.strip():
                model_name = line.split()[0]
                models.append(model_name)
        return models
    except Exception as e:
        st.error(f"Error getting models: {e}")
        return []

def call_ollama(model, prompt):
    """Call Ollama with model and prompt."""
    try:
        print(f"\n{'='*70}\nMODEL: {model}\nPROMPT: {prompt}\n{'='*70}")
        result = subprocess.run(['ollama', 'run', model], input=prompt, capture_output=True, text=True, timeout=120)
        response = result.stdout.strip()
        print(f"RESPONSE:\n{response}\n{'='*70}\n")
        return response
    except subprocess.TimeoutExpired:
        return "[TIMEOUT - Model took too long]"
    except Exception as e:
        return f"[ERROR: {str(e)}]"

def load_prompt_template(mode, length):
    """Load prompt template from file."""
    import os
    
    # Get directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    mode_dir = f"{mode.lower()}_prompts"
    filename = f"{length.lower()}_prompt.txt"
    filepath = os.path.join(script_dir, mode_dir, filename)
    
    try:
        with open(filepath, 'r') as f:
            return f.read()
    except FileNotFoundError:
        return f"[ERROR: Template not found at {filepath}]\nCurrent directory: {os.getcwd()}\nScript directory: {script_dir}"
    except Exception as e:
        return f"[ERROR loading template: {str(e)}]"

# Streamlit App
st.set_page_config(page_title="Sensor Data Narrative Generator", layout="wide")
st.title("Smart Home Sensor Narrative Generator")
st.markdown("---")

# Get data file path relative to script location
script_dir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(script_dir, "data_full.csv")

# Load date range for calendar validation (lightweight)
if not os.path.exists(data_file):
    st.error(f"Data file not found: {data_file}")
    st.stop()

try:
    with st.spinner("Loading data range..."):
        min_date, max_date = get_data_date_range(data_file)
    st.success(f"Data available: {min_date.date()} to {max_date.date()}")
except Exception as e:
    st.error(f"Error reading data file: {e}")
    st.stop()

st.markdown("---")

# Step 1: Model selection
models = get_ollama_models()
if not models:
    st.error("No Ollama models found. Run `ollama pull <model>` first.")
    st.stop()

selected_model = st.selectbox("Select Model", models, index=0)
st.markdown("---")

# Step 2: Time window selection
st.subheader("Select Time Window")
time_window = st.radio("Duration", ["1 Day (24h)", "1 Week (7d)", "30 Days"], horizontal=True)

# Map selection to hours
window_hours = {"1 Day (24h)": 24, "1 Week (7d)": 168, "30 Days": 720}[time_window]

# Sensor type selection
sensor_type = st.selectbox("Sensor Type", ["Motion"])

st.markdown("---")

# Step 3: Date and time picker with validation
st.subheader("Select Start Date and Time")
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input(
        "Start Date",
        value=min_date.date(),
        min_value=min_date.date(),
        max_value=max_date.date()
    )
with col2:
    start_time_input = st.time_input("Start Time", value=time(0, 0))

start_datetime = datetime.combine(start_date, start_time_input)
end_datetime = start_datetime + timedelta(hours=window_hours)

# Validate that end doesn't exceed data range
if end_datetime > max_date:
    st.error(f"Selected window extends beyond available data. Window end: {end_datetime}, Data end: {max_date}")
    st.warning("Please select an earlier start date or shorter time window.")
    st.stop()

st.info(f"Selected window: {start_datetime} to {end_datetime} ({window_hours}h)")
st.markdown("---")

# Load and visualize data immediately after date selection
st.subheader("Data Preview")
with st.spinner("Loading data and generating visualization..."):
    try:
        # Load full data
        df = load_data_from_file(data_file)
        
        # Extract time window
        window_df = extract_window(df, hours=window_hours, start_time=start_datetime)
        
        if len(window_df) == 0:
            st.error(f"No data found in selected time window")
            st.stop()
        
        # Filter to motion data only
        motion_df = window_df[window_df['sensor_type'] == 'Motion'].copy()
        
        if len(motion_df) == 0:
            st.error("No motion data found in selected time window")
            st.stop()
        
        st.success(f"Loaded {len(motion_df):,} motion sensor readings")
        
        # Display motion visualization
        st.markdown("#### Motion Activations by Room")
        fig = plot_motion_activations(motion_df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No motion activations to visualize")
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

st.markdown("---")

# Step 4: Prompt type and length
st.subheader("Prompt Configuration")
prompt_mode = st.radio("Prompt Type", ["Scientific", "Normal"], horizontal=True)
prompt_length = st.radio("Length", ["Short", "Medium", "Long"], horizontal=True)

# Load prompt template
prompt_template = load_prompt_template(prompt_mode, prompt_length)

# Show loaded prompt (preview)
with st.expander("View Prompt Template"):
    st.text(prompt_template)

st.markdown("---")

# Step 5: Two-step generation process
st.markdown("---")

# Initialize session state for prompt editing
if 'generated_prompt' not in st.session_state:
    st.session_state.generated_prompt = None
if 'edited_prompt' not in st.session_state:
    st.session_state.edited_prompt = None

# Step 5a: Generate Prompt button
if st.button("Generate Prompt", type="primary"):
    with st.spinner("Formatting data and building prompt..."):
        try:
            # Generate segment analysis summary
            segment_summary = segment_and_analyze(motion_df, start_datetime, end_datetime)
            
            # Format motion data (already loaded as motion_df)
            formatted_data = format_combined(motion_df)
            
            # Combine segment analysis with detailed data
            full_data = segment_summary + "\n\n" + "="*70 + "\n\nDETAILED SENSOR DATA:\n" + "="*70 + "\n\n" + formatted_data
            
            # Build final prompt by inserting data into template
            if "{data}" in prompt_template:
                final_prompt = prompt_template.replace("{data}", full_data)
            else:
                final_prompt = prompt_template + "\n\n" + full_data
            
            # Store in session state
            st.session_state.generated_prompt = final_prompt
            st.session_state.edited_prompt = final_prompt
            
            st.success("Prompt generated! Review and edit below, then click 'Generate Report'.")
            
        except Exception as e:
            st.error(f"Error formatting data: {e}")
            st.stop()

# Step 5b: Show editable prompt if generated
if st.session_state.generated_prompt is not None:
    st.markdown("### Edit Prompt (Optional)")
    st.info("You can modify the prompt below before generating the report.")
    
    edited_prompt = st.text_area(
        "Final Prompt",
        value=st.session_state.edited_prompt,
        height=500,
        key="prompt_editor"
    )
    
    # Update session state with edited content
    st.session_state.edited_prompt = edited_prompt
    
    st.markdown("---")
    
    # Step 5c: Generate Report button (only visible after prompt is generated)
    if st.button("Generate Report", type="secondary"):
        with st.spinner(f"Generating narrative with {selected_model}..."):
            response = call_ollama(selected_model, st.session_state.edited_prompt)
        
        # Display response
        st.markdown("### Generated Narrative:")
        st.text_area("Response", value=response, height=400, disabled=True)
        
        # Option to regenerate with same prompt
        st.markdown("---")
        st.info("To generate a new prompt, modify your selections above and click 'Generate Prompt' again.")