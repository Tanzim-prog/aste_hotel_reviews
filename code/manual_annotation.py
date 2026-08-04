import streamlit as st
import pandas as pd
import os

# Page config
st.set_page_config(page_title="ASTE Annotation Tool", layout="wide")
st.title("🏨 Hotel Review Annotation Tool")

# Define paths
DATA_PATH = r"F:\Research\data"
TRAIN_PATH = r"F:\Research\data\train.csv"
ANNOTATIONS_SAVE_PATH = r"F:\Research\data\manual_triplets.csv"

# Create folders if they don't exist
os.makedirs(DATA_PATH, exist_ok=True)

# Aspect ontology
ASPECTS = {
    'Room': 'Bedroom, bed, bathroom, spacious, decor',
    'Staff': 'Receptionist, service, helpful, friendly',
    'Food': 'Breakfast, restaurant, meal, taste',
    'Location': 'Area, neighborhood, distance, transport',
    'Cleanliness': 'Clean, dirty, hygiene, maintenance',
    'Value': 'Price, expensive, cheap, worth',
    'WiFi': 'Internet, connection, speed, signal',
    'Amenities': 'Pool, gym, parking, elevator, AC'
}

SENTIMENTS = ['POSITIVE', 'NEGATIVE', 'NEUTRAL']
COLORS = {
    'POSITIVE': '🟢',
    'NEGATIVE': '🔴',
    'NEUTRAL': '🟡'
}

# Initialize session state - FIXED
@st.cache_resource
def load_data():
    """Load training data once"""
    try:
        train_data = pd.read_csv(TRAIN_PATH).head(300)
        st.sidebar.success(f"✓ Loaded {len(train_data)} reviews")
        return train_data
    except FileNotFoundError:
        st.error(f"❌ Error: File not found at {TRAIN_PATH}")
        st.stop()

reviews_data = load_data()

# Initialize session state
if 'current_review' not in st.session_state:
    st.session_state.current_review = 0
    st.session_state.triplets = []
    st.session_state.skipped = set()
    st.session_state.completed = set()

# Sidebar - Progress
with st.sidebar:
    st.header("📊 Progress")
    total = len(reviews_data)
    completed = len(st.session_state.completed)
    skipped = len(st.session_state.skipped)
    remaining = total - completed - skipped
    
    st.metric("Total Reviews", total)
    st.metric("Annotated", completed)
    st.metric("Skipped", skipped)
    st.metric("Remaining", remaining)
    
    progress = completed / total if total > 0 else 0
    st.progress(progress)
    
    st.markdown("---")
    st.markdown(f"**📁 Save Path:** {DATA_PATH}")
    
    if st.button("💾 Save Progress", key="save_btn"):
        if st.session_state.triplets:
            df = pd.DataFrame(st.session_state.triplets)
            df.to_csv(ANNOTATIONS_SAVE_PATH, index=False)
            st.success(f"✓ Saved {len(st.session_state.triplets)} triplets!")
            st.info(f"Location: {ANNOTATIONS_SAVE_PATH}")
        else:
            st.warning("No triplets to save yet")

# Main annotation area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Review Content")
    
    current_idx = st.session_state.current_review
    review_row = reviews_data.iloc[current_idx]
    
    st.markdown(f"**Hotel:** {review_row['hotel_name']}")
    st.markdown(f"**Reviewer:** {review_row['reviewer_name']}")
    
    st.markdown("---")
    st.markdown("### 📝 Review Text:")
    st.info(review_row['review_text'], icon="💬")
    st.markdown("---")

with col2:
    st.header("Navigation")
    
    col_nav1, col_nav2, col_nav3 = st.columns(3)
    
    with col_nav1:
        if st.button("⬅️ Previous", key="prev"):
            if st.session_state.current_review > 0:
                st.session_state.current_review -= 1
                st.rerun()
    
    with col_nav2:
        st.write(f"{current_idx + 1}/{len(reviews_data)}")
    
    with col_nav3:
        if st.button("Next ➡️", key="next"):
            if st.session_state.current_review < len(reviews_data) - 1:
                st.session_state.current_review += 1
                st.rerun()
    
    if st.button("⏭️ Skip", key="skip"):
        st.session_state.skipped.add(current_idx)
        if st.session_state.current_review < len(reviews_data) - 1:
            st.session_state.current_review += 1
        st.rerun()

# Annotation section
st.header("🏷️ Add Triplets")

col1, col2, col3 = st.columns(3)

with col1:
    aspect = st.selectbox("Select Aspect:", list(ASPECTS.keys()), key="aspect_select")
    st.caption(f"Examples: {ASPECTS[aspect]}")

with col2:
    opinion = st.text_input("Opinion:", placeholder="spacious, friendly, slow", key="opinion_input")

with col3:
    sentiment = st.selectbox("Sentiment:", SENTIMENTS, key="sentiment_select")

if st.button("➕ Add Triplet", key="add_triplet"):
    if not opinion.strip():
        st.error("Please enter an opinion!")
    else:
        triplet = {
            'review_id': current_idx,
            'hotel': review_row['hotel_name'],
            'review_text': review_row['review_text'],
            'aspect': aspect,
            'opinion': opinion,
            'sentiment': sentiment
        }
        st.session_state.triplets.append(triplet)
        st.session_state.completed.add(current_idx)
        st.success(f"✓ Added: ({aspect}, {opinion}, {sentiment})")

# Show triplets for current review
st.header("✅ Triplets for This Review")

current_triplets = [t for t in st.session_state.triplets if t['review_id'] == current_idx]

if current_triplets:
    for idx, triplet in enumerate(current_triplets):
        col1, col2 = st.columns([5, 1])
        with col1:
            emoji = COLORS[triplet['sentiment']]
            st.markdown(f"{emoji} **({triplet['aspect']}, {triplet['opinion']}, {triplet['sentiment']})**")
        with col2:
            if st.button("❌", key=f"delete_{current_idx}_{idx}"):
                st.session_state.triplets.pop(idx)
                st.rerun()
else:
    st.info("No triplets added yet.")

# Statistics
st.markdown("---")
st.header("📈 Statistics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Triplets", len(st.session_state.triplets))

with col2:
    aspect_counts = {}
    for t in st.session_state.triplets:
        aspect_counts[t['aspect']] = aspect_counts.get(t['aspect'], 0) + 1
    top_aspect = max(aspect_counts, key=aspect_counts.get) if aspect_counts else "N/A"
    st.metric("Top Aspect", top_aspect)

with col3:
    sentiment_counts = {}
    for t in st.session_state.triplets:
        sentiment_counts[t['sentiment']] = sentiment_counts.get(t['sentiment'], 0) + 1
    pos_count = sentiment_counts.get('POSITIVE', 0)
    st.metric("Positive", pos_count)

with col4:
    neg_count = sentiment_counts.get('NEGATIVE', 0)
    st.metric("Negative", neg_count)

# Aspect distribution chart
if st.session_state.triplets:
    aspect_data = pd.DataFrame(st.session_state.triplets)
    aspect_dist = aspect_data['aspect'].value_counts()
    st.bar_chart(aspect_dist)

# Auto-save every 5 triplets
if len(st.session_state.triplets) > 0 and len(st.session_state.triplets) % 5 == 0:
    df = pd.DataFrame(st.session_state.triplets)
    df.to_csv(ANNOTATIONS_SAVE_PATH, index=False)