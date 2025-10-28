"""
Streamlit Emotion Dashboard
File: streamlit_emotion_dashboard.py
Purpose: Interactive in-class visualization for the "Motivation & Emotion" weekly diary (Oct 20-26).
Provides editable 7-day diary, emotion frequency chart, timeline, mood color map, "Inside Out" dashboard, and CSV export.

Run: 
1. python -m pip install streamlit pandas matplotlib
2. streamlit run streamlit_emotion_dashboard.py

Author: Generated for Reda HEDDAD
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="Emotion Dashboard — Oct 20-26", layout="wide")

# ---------- Helper functions ----------
EMOTION_LIST = ["Joy", "Sadness", "Anger", "Fear", "Disgust", "Frustration", "Fatigue", "Anxiety", "Satisfaction"]
DEFAULT_COLORS = {
    "Joy": "#FFD54F",
    "Sadness": "#64B5F6",
    "Anger": "#E57373",
    "Fear": "#BA68C8",
    "Disgust": "#81C784",
    "Frustration": "#FFB74D",
    "Fatigue": "#90A4AE",
    "Anxiety": "#FF8A65",
    "Satisfaction": "#A5D6A7",
}

def load_csv_data(uploaded_file):
    """Load and process CSV file with proper column mapping"""
    df = pd.read_csv(uploaded_file)
    
    # Map CSV columns to expected format
    column_mapping = {
        'Date': 'date',
        'Situation / Context': 'situation',
        'Dominant Emotion': 'dominant_emotion',
        'Intensity (1–5)': 'intensity',
        'Intensity (1â€"5)': 'intensity',  # Handle encoding issue
        'Body Sensations': 'body_sensations',
        'Thoughts / Triggers': 'thoughts_triggers',
    }
    
    # Rename columns
    df = df.rename(columns=column_mapping)
    
    # Remove the 'Psychological Link' column if present
    if 'Psychological Link' in df.columns:
        df = df.drop(columns=['Psychological Link'])
    
    # Convert date column to datetime.date
    df['date'] = pd.to_datetime(df['date']).dt.date
    
    # Ensure intensity is numeric
    df['intensity'] = pd.to_numeric(df['intensity'], errors='coerce').fillna(3).astype(int)
    
    # Clean up emotion names (remove extra spaces)
    df['dominant_emotion'] = df['dominant_emotion'].str.strip()
    
    return df

def sample_week(start_date="2025-10-20"):
    """Generate sample data if no CSV is uploaded"""
    start = datetime.fromisoformat(start_date)
    dates = [(start + timedelta(days=i)).date() for i in range(7)]
    data = [
        {"date": dates[0], "situation": "Project work + midterm prep", "dominant_emotion": "Frustration", "intensity": 4, "body_sensations": "tight chest, tired eyes", "thoughts_triggers": "overlap of deadlines"},
        {"date": dates[1], "situation": "Late-night hackathon session", "dominant_emotion": "Fatigue", "intensity": 5, "body_sensations": "heavy limbs, low energy", "thoughts_triggers": "sleep deprivation"},
        {"date": dates[2], "situation": "Submitting deliverable", "dominant_emotion": "Satisfaction", "intensity": 5, "body_sensations": "rush of relief", "thoughts_triggers": "task completed"},
        {"date": dates[3], "situation": "Group coordination problems", "dominant_emotion": "Frustration", "intensity": 5, "body_sensations": "irritation, clenched jaw", "thoughts_triggers": "miscommunication"},
        {"date": dates[4], "situation": "All-nighter preparing for midterm", "dominant_emotion": "Fatigue", "intensity": 5, "body_sensations": "sluggish, headaches", "thoughts_triggers": "time pressure"},
        {"date": dates[5], "situation": "Hackathon presentation", "dominant_emotion": "Anxiety", "intensity": 4, "body_sensations": "fast heartbeat", "thoughts_triggers": "public demo"},
        {"date": dates[6], "situation": "Reflection and recovery", "dominant_emotion": "Frustration", "intensity": 4, "body_sensations": "mixed tiredness & relief", "thoughts_triggers": "evaluating progress"},
    ]
    return pd.DataFrame(data)

def emotion_frequency(df):
    """Calculate emotion frequency"""
    freq = df['dominant_emotion'].value_counts().rename_axis('emotion').reset_index(name='count')
    return freq

def plot_bar(freq_df):
    """Create bar chart of emotion frequency"""
    fig, ax = plt.subplots(figsize=(6,4))
    colors = [DEFAULT_COLORS.get(e, '#CCCCCC') for e in freq_df['emotion']]
    ax.bar(freq_df['emotion'], freq_df['count'], color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Days')
    ax.set_title('Emotion Frequency (days dominated)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

def plot_timeline(df):
    """Create timeline visualization"""
    fig, ax = plt.subplots(figsize=(10,2.5))
    y = [1] * len(df)
    colors = [DEFAULT_COLORS.get(e, '#CCCCCC') for e in df['dominant_emotion']]
    ax.scatter(df['date'], y, s=[300]*len(df), c=colors, alpha=0.7, edgecolor='black', linewidth=2)
    for i, txt in enumerate(df['dominant_emotion']):
        ax.text(df['date'].iloc[i], 1.15, txt, ha='center', va='bottom', fontsize=9, rotation=25)
    ax.get_yaxis().set_visible(False)
    ax.set_title('7-Day Emotion Timeline')
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout()
    return fig

def mood_color_map(df):
    """Create mood color strip"""
    fig, ax = plt.subplots(figsize=(8,1.5))
    
    # Create a simple array of ones with the same length as the dataframe
    # This will serve as the numerical data for imshow
    color_data = np.ones((1, len(df)))
    
    # Get colors for each day
    colors = [DEFAULT_COLORS.get(e, '#CCCCCC') for e in df['dominant_emotion']]
    
    # Create the color map
    ax.imshow(color_data, aspect='auto', cmap='viridis')  # Using a fallback colormap
    
    # Overlay with our custom colors
    for i, color in enumerate(colors):
        ax.add_patch(plt.Rectangle((i-0.5, -0.5), 1, 1, color=color, alpha=0.7))
    
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([d.strftime('%a %d') for d in df['date']], rotation=45)
    ax.set_yticks([])
    ax.set_title('Mood Color Map')
    ax.set_xlim(-0.5, len(df)-0.5)
    ax.set_ylim(-0.5, 0.5)
    plt.tight_layout()
    return fig

def to_csv_bytes(df):
    """Convert dataframe to CSV bytes"""
    return df.to_csv(index=False).encode('utf-8')

# ---------- UI ----------
st.title('Inside Out — Weekly Emotion Dashboard')
st.markdown('**Week:** Oct 20 — Oct 26, 2025')

# File upload option
uploaded_file = st.file_uploader("Upload your emotion diary CSV (optional)", type=['csv'])

# Load or create session data
if 'df' not in st.session_state or uploaded_file is not None:
    if uploaded_file is not None:
        try:
            st.session_state.df = load_csv_data(uploaded_file)
            st.success('✅ CSV loaded successfully!')
        except Exception as e:
            st.error(f'Error loading CSV: {str(e)}')
            st.session_state.df = sample_week()
    else:
        st.session_state.df = sample_week()

col1, col2 = st.columns([2,1])
with col1:
    st.header('Editable 7-Day Diary')
    st.write('Edit any cell. When done, press the **Apply changes** button to update visualizations.')
    edited = st.data_editor(
        st.session_state.df, 
        num_rows="fixed", 
        key='editor', 
        use_container_width=True,
        column_config={
            "date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
            "intensity": st.column_config.NumberColumn("Intensity", min_value=1, max_value=5),
        }
    )
    if st.button('Apply changes'):
        try:
            edited['date'] = pd.to_datetime(edited['date']).dt.date
            edited['intensity'] = pd.to_numeric(edited['intensity'], errors='coerce').fillna(3).astype(int)
            st.session_state.df = edited
            st.success('✅ Changes applied!')
        except Exception as e:
            st.error(f'Error applying changes: {str(e)}')

with col2:
    st.header('Summary')
    freq = emotion_frequency(st.session_state.df)
    most_freq = freq.iloc[0]['emotion'] if not freq.empty else '—'
    most_freq_count = int(freq.iloc[0]['count']) if not freq.empty else 0
    st.metric('Most frequent emotion', most_freq, f"{most_freq_count} days")
    
    avg_intensity = st.session_state.df['intensity'].mean()
    st.metric('Average intensity', f"{avg_intensity:.1f}/5")
    
    st.markdown('**Typical triggers:** Overlapping deadlines, late nights, group coordination, hackathon pressure')

# Visuals
st.divider()
st.header('Visual Summary')
vis_col1, vis_col2 = st.columns(2)

with vis_col1:
    freq = emotion_frequency(st.session_state.df)
    fig_bar = plot_bar(freq)
    st.pyplot(fig_bar)
    st.caption('Bar chart showing how many days each emotion dominated.')

with vis_col2:
    fig_tl = plot_timeline(st.session_state.df)
    st.pyplot(fig_tl)
    st.caption('Timeline with labeled dominant emotion for each day.')

st.divider()
map_col1, map_col2 = st.columns([3,1])
with map_col1:
    fig_map = mood_color_map(st.session_state.df)
    st.pyplot(fig_map)
    st.caption('A compact color strip representing daily moods.')

with map_col2:
    st.header('Psychological Notes')
    st.write('- **Emotion differentiation:** labeling helps clarity and regulation.')
    st.write('- **Reappraisal observed:** stress reframed as achievement (eustress).')
    st.write('- **Neuroscience link:** concurrent amygdala (stress) and prefrontal activation (goal pursuit).')

# Inside Out Dashboard
st.divider()
st.header('Inside Out Dashboard')
st.write('Compact overview for class presentation.')
io_col1, io_col2 = st.columns(2)
with io_col1:
    st.subheader('Dominant Emotion')
    st.markdown(f'**{most_freq}**')
    st.write('Days dominated:', most_freq_count if most_freq != '—' else 0)

with io_col2:
    st.subheader('Top Triggers')
    # Extract most common triggers from data
    triggers = st.session_state.df['thoughts_triggers'].value_counts().head(3)
    for i, (trigger, count) in enumerate(triggers.items(), 1):
        st.write(f'{i}. {trigger}')

# Download
st.divider()
st.header('Export & Present')
csv_bytes = to_csv_bytes(st.session_state.df)
st.download_button('Download diary CSV', csv_bytes, file_name='emotion_diary_oct20-26.csv', mime='text/csv')

# Small printable summary
st.markdown('### Printable Summary')
summary = (
    f"Week: Oct 20–26, 2025\nMost frequent emotion: {most_freq} ({most_freq_count} days)\n"
    f"Average intensity: {avg_intensity:.1f}/5\n"
    f"Typical triggers: Overlapping deadlines, late nights, hackathon pressure.\n"
    "Psychological insight: The blend of fatigue and satisfaction reflects eustress and goal-directed control."
)
st.text_area('One-paragraph summary (copy for slides)', value=summary, height=140)

st.caption('Prepared for in-class presentation — editable to match your diary entries.')