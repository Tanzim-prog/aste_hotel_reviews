import pandas as pd
import requests
import json
import os
import time
import logging
from datetime import datetime, timedelta
import sys

# ============================================================================
# CONFIGURATION
# ============================================================================

TRAIN_PATH = r"F:\Research\data\train.csv"
MANUAL_TRIPLETS_PATH = r"F:\Research\data\manual_triplets.csv"
WEAK_LABELS_PATH = r"F:\Research\data\weak_triplets.csv"
COMBINED_PATH = r"F:\Research\data\all_triplets.csv"
LOG_FILE = r"F:\Research\data\weak_labeling.log"
CHECKPOINT_FILE = r"F:\Research\data\weak_labeling_checkpoint.json"  # New checkpoint file

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "mistral"

VALID_ASPECTS = {
    "Room",
    "Staff",
    "Food",
    "Location",
    "Cleanliness",
    "Value",
    "WiFi",
    "Amenities"
}

ASPECT_MAPPING = {
    "suite": "Room", "bedroom": "Room", "bed": "Room", "bathroom": "Room",
    "view": "Room", "decor": "Room", "space": "Room", "window": "Room",
    "management": "Staff", "check-in": "Staff", "reception": "Staff",
    "receptionist": "Staff", "service": "Staff", "friendly": "Staff",
    "restaurant": "Food", "breakfast": "Food", "meal": "Food", "coffee": "Food",
    "convenience": "Location", "area": "Location", "distance": "Location",
    "accessibility": "Location", "hygiene": "Cleanliness", "clean": "Cleanliness",
    "dirty": "Cleanliness", "price": "Value", "cost": "Value", "expensive": "Value",
    "elevator": "Amenities", "roof top": "Amenities", "rooftop": "Amenities",
    "pool": "Amenities", "gym": "Amenities", "parking": "Amenities",
    "internet": "WiFi", "wifi": "WiFi", "connection": "WiFi", "signal": "WiFi"
}

RETRY_ATTEMPTS = 3
RETRY_BASE_WAIT = 1  # seconds
OLLAMA_TIMEOUT = 180  # seconds
BATCH_SAVE_INTERVAL = 10  # Save batch every 10 reviews
MIN_OPINION_LENGTH = 2  # Minimum characters for opinion
CHECKPOINT_INTERVAL = 5  # Save checkpoint every 5 reviews

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# CHECKPOINT FUNCTIONS
# ============================================================================

def save_checkpoint(review_id, processed_count, total_triplets, last_timestamp=None):
    """Save checkpoint to resume later."""
    checkpoint_data = {
        "last_review_id": str(review_id),
        "processed_count": processed_count,
        "total_triplets": total_triplets,
        "timestamp": last_timestamp or datetime.now().isoformat(),
        "model": MODEL
    }
    try:
        with open(CHECKPOINT_FILE, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        logger.debug(f"Checkpoint saved at review {review_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")
        return False

def load_checkpoint():
    """Load checkpoint to resume from where we left off."""
    if not os.path.exists(CHECKPOINT_FILE):
        return None
    
    try:
        with open(CHECKPOINT_FILE, 'r') as f:
            checkpoint_data = json.load(f)
        
        last_review_id = checkpoint_data.get("last_review_id")
        if last_review_id:
            logger.info(f"Found checkpoint: last processed review ID = {last_review_id}")
            logger.info(f"  Processed: {checkpoint_data.get('processed_count', 0)} reviews")
            logger.info(f"  Total triplets: {checkpoint_data.get('total_triplets', 0)}")
            logger.info(f"  Last checkpoint at: {checkpoint_data.get('timestamp', 'unknown')}")
            return checkpoint_data
        else:
            return None
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        return None

def clear_checkpoint():
    """Clear checkpoint after successful completion."""
    try:
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
            logger.info("Checkpoint cleared")
            return True
    except Exception as e:
        logger.error(f"Failed to clear checkpoint: {e}")
    return False

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def normalize_text(text):
    """Normalize text: lowercase, strip, collapse whitespace."""
    if not isinstance(text, str):
        return ""
    return " ".join(text.strip().lower().split())


def normalize_aspect(aspect_text):
    """Map aspect text to valid category or return None."""
    if not aspect_text or not isinstance(aspect_text, str):
        return None
    
    normalized = normalize_text(aspect_text)
    
    # Direct match
    for valid in VALID_ASPECTS:
        if normalized == valid.lower():
            return valid
    
    # Substring match in mapping
    for key, value in ASPECT_MAPPING.items():
        if key in normalized:
            return value
    
    return None


def validate_opinion(opinion_text):
    """Validate opinion: non-empty, minimum length, not just punctuation."""
    if not opinion_text or not isinstance(opinion_text, str):
        return None
    
    opinion = opinion_text.strip()
    
    if len(opinion) < MIN_OPINION_LENGTH:
        return None
    
    # Check if opinion is just punctuation/spaces
    if not any(c.isalnum() for c in opinion):
        return None
    
    return opinion


def validate_sentiment(sentiment_text):
    """Validate and normalize sentiment."""
    if not sentiment_text or not isinstance(sentiment_text, str):
        return "NEUTRAL"
    
    sentiment = sentiment_text.strip().upper()
    if sentiment in ["POSITIVE", "NEGATIVE", "NEUTRAL"]:
        return sentiment
    
    return "NEUTRAL"


def normalize_triplet_for_dedup(triplet):
    """Create normalized tuple for duplicate detection (case-insensitive, whitespace-normalized)."""
    return (
        triplet["aspect"].lower(),
        normalize_text(triplet["opinion"]),
        triplet["sentiment"].upper()
    )


def remove_duplicate_triplets(triplets):
    """Remove exact duplicate triplets (case-insensitive, whitespace-normalized)."""
    seen = set()
    unique = []
    
    for t in triplets:
        key = normalize_triplet_for_dedup(t)
        if key not in seen:
            seen.add(key)
            unique.append(t)
    
    return unique


def extract_triplets_from_ollama(review_text, review_id, attempt=1):
    """
    Extract triplets from Ollama with exponential backoff retry.
    Returns list of valid triplets or empty list on failure.
    Logs raw response on JSON parsing failure.
    """
    if attempt > RETRY_ATTEMPTS:
        logger.error(f"Review {review_id}: Failed after {RETRY_ATTEMPTS} attempts")
        return []
    
    review_text_trunc = str(review_text)[:1500]
    
    prompt = f"""IMPORTANT: Extract aspect-opinion-sentiment triplets from this hotel review.

STRICT RULES:
1. Use ONLY these aspects (NO INVENTED ASPECTS):
   Room, Staff, Food, Location, Cleanliness, Value, WiFi, Amenities
2. Only map opinions that EXPLICITLY appear in the review.
3. Do NOT invent or infer opinions not mentioned.
4. Do NOT create new aspect categories.
5. Map similar concepts to the closest valid aspect only.
6. Sentiment must be: POSITIVE, NEGATIVE, or NEUTRAL only.
7. Opinion must be a real adjective/descriptor from the text.

Return ONLY a valid JSON array. No markdown, no comments, no extra text.

Examples:
Input: "The receptionist was unfriendly."
Output: [{{"aspect":"Staff","opinion":"unfriendly","sentiment":"NEGATIVE"}}]

Input: "The room was spacious and clean."
Output: [{{"aspect":"Room","opinion":"spacious","sentiment":"POSITIVE"}},{{"aspect":"Cleanliness","opinion":"clean","sentiment":"POSITIVE"}}]

Input: "Good hotel"
Output: []

Review:
{review_text_trunc}

JSON array (empty array [] if no explicit aspects found):"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.1,
                "keep_alive": "30m"
            },
            timeout=OLLAMA_TIMEOUT
        )
        
        if response.status_code != 200:
            logger.warning(f"Review {review_id}: HTTP {response.status_code}, retrying...")
            time.sleep(RETRY_BASE_WAIT ** attempt)
            return extract_triplets_from_ollama(review_text, review_id, attempt + 1)
        
        result = response.json()
        text = result.get("response", "")
        
        # Extract JSON array
        start = text.find("[")
        end = text.rfind("]") + 1
        
        if start < 0 or end <= start:
            logger.warning(f"Review {review_id}: No JSON array found in response")
            logger.debug(f"Raw response: {text[:500]}")  # Log first 500 chars
            return []
        
        json_str = text[start:end]
        data = json.loads(json_str)
        
        if not isinstance(data, list):
            logger.warning(f"Review {review_id}: JSON is not array")
            logger.debug(f"Raw response: {text[:500]}")
            return []
        
        # Validate and normalize triplets
        valid_triplets = []
        for item in data:
            if not isinstance(item, dict):
                continue
            
            aspect = normalize_aspect(item.get("aspect", ""))
            opinion = validate_opinion(item.get("opinion", ""))
            sentiment = validate_sentiment(item.get("sentiment", "NEUTRAL"))
            
            # Skip if aspect is not valid or opinion is empty
            if not aspect or not opinion:
                continue
            
            valid_triplets.append({
                "aspect": aspect,
                "opinion": opinion,
                "sentiment": sentiment
            })
        
        # Remove duplicates (case-insensitive, whitespace-normalized)
        valid_triplets = remove_duplicate_triplets(valid_triplets)
        
        return valid_triplets
    
    except json.JSONDecodeError as e:
        logger.warning(f"Review {review_id}: JSON decode error: {e}")
        logger.debug(f"Raw response that failed to parse: {text[:1000]}")
        time.sleep(RETRY_BASE_WAIT ** attempt)
        return extract_triplets_from_ollama(review_text, review_id, attempt + 1)
    
    except requests.Timeout:
        logger.warning(f"Review {review_id}: Ollama timeout, retrying...")
        time.sleep(RETRY_BASE_WAIT ** attempt)
        return extract_triplets_from_ollama(review_text, review_id, attempt + 1)
    
    except Exception as e:
        logger.warning(f"Review {review_id}: {type(e).__name__}: {e}")
        time.sleep(RETRY_BASE_WAIT ** attempt)
        return extract_triplets_from_ollama(review_text, review_id, attempt + 1)


def warmup_ollama():
    """Warm up Ollama model before processing."""
    logger.info("Warming up Ollama model...")
    try:
        requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": "Hello",
                "stream": False,
                "keep_alive": "30m"
            },
            timeout=OLLAMA_TIMEOUT
        )
        logger.info("✓ Ollama ready")
        return True
    except Exception as e:
        logger.error(f"Ollama warmup failed: {e}")
        return False


def load_processed_ids():
    """Load already processed review_ids from weak_triplets.csv if it exists."""
    if not os.path.exists(WEAK_LABELS_PATH):
        return set()
    
    try:
        existing_df = pd.read_csv(WEAK_LABELS_PATH)
        processed = set(existing_df["review_id"].unique())
        logger.info(f"Found {len(processed)} already processed reviews in weak_triplets.csv")
        return processed
    except Exception as e:
        logger.error(f"Error loading processed reviews: {e}")
        return set()


def append_batch_to_csv(batch_df):
    """Append batch of triplets to CSV file."""
    try:
        if not batch_df.empty:
            if os.path.exists(WEAK_LABELS_PATH):
                batch_df.to_csv(WEAK_LABELS_PATH, mode='a', header=False, index=False)
            else:
                batch_df.to_csv(WEAK_LABELS_PATH, mode='w', header=True, index=False)
    except Exception as e:
        logger.error(f"Error saving batch to CSV: {e}")


def estimate_eta(processed, total, start_time):
    """Calculate estimated time to completion."""
    if processed == 0:
        return "N/A"
    
    elapsed = time.time() - start_time
    per_review = elapsed / processed
    remaining = total - processed
    eta_seconds = per_review * remaining
    
    return str(timedelta(seconds=int(eta_seconds)))


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    print("\n" + "="*70)
    print("STAGE 2B: PRODUCTION-READY OLLAMA WEAK LABELING FOR ASTE")
    print("Enhanced with checkpoint/resume functionality")
    print("="*70)
    
    logger.info("Starting weak labeling pipeline...")
    
    # Load training data
    logger.info("[1/6] Loading training data...")
    try:
        train_df = pd.read_csv(TRAIN_PATH)
        logger.info(f"✓ Loaded {len(train_df)} reviews")
    except Exception as e:
        logger.error(f"Failed to load training data: {e}")
        return
    
    # Ensure review_id column exists
    if "review_id" not in train_df.columns:
        train_df["review_id"] = train_df.index
        logger.warning("No review_id column; using DataFrame index as review_id")
    
    # Load manual triplets
    logger.info("[2/6] Loading manual triplets...")
    try:
        manual_df = pd.read_csv(MANUAL_TRIPLETS_PATH)
        logger.info(f"✓ Loaded {len(manual_df)} manual triplets")
    except Exception as e:
        logger.error(f"Failed to load manual triplets: {e}")
        return
    
    # Find unannotated reviews
    logger.info("[3/6] Identifying unannotated reviews...")
    manual_review_ids = set(manual_df["review_id"].unique())
    
    unannotated = train_df[~train_df["review_id"].isin(manual_review_ids)].copy()
    
    logger.info(f"✓ {len(manual_review_ids)} reviews already annotated")
    logger.info(f"✓ {len(unannotated)} reviews require labeling")
    
    # Load already processed reviews from CSV
    logger.info("[4/6] Checking for resumable progress...")
    processed_ids = load_processed_ids()
    
    if len(processed_ids) > 0:
        unannotated = unannotated[~unannotated["review_id"].isin(processed_ids)].copy()
        logger.info(f"✓ Resuming: {len(unannotated)} reviews remaining")
    
    # Check for checkpoint (more granular than CSV-based)
    checkpoint = load_checkpoint()
    if checkpoint:
        last_review_id = checkpoint["last_review_id"]
        # Find the position of the last processed review
        try:
            # Convert to appropriate type for comparison
            last_id = str(last_review_id)
            # Filter out reviews up to and including the checkpoint
            # We need to find the index position
            review_ids = unannotated["review_id"].astype(str)
            if last_id in review_ids.values:
                idx = review_ids[review_ids == last_id].index[0]
                # Get the position in the DataFrame
                pos = unannotated.index.get_loc(idx)
                # Skip all reviews up to and including the checkpoint
                unannotated = unannotated.iloc[pos + 1:] if pos + 1 < len(unannotated) else pd.DataFrame()
                logger.info(f"✓ Resuming from after review ID {last_id}")
                logger.info(f"✓ {len(unannotated)} reviews remaining after checkpoint")
            else:
                logger.warning(f"Checkpoint review ID {last_id} not found in remaining reviews")
                # Clear invalid checkpoint
                clear_checkpoint()
        except Exception as e:
            logger.warning(f"Error processing checkpoint: {e}")
            # Clear invalid checkpoint
            clear_checkpoint()
    
    if len(unannotated) == 0:
        logger.info("All reviews already labeled. Combining datasets...")
        combine_datasets(manual_df)
        clear_checkpoint()
        return
    
    # Warmup Ollama
    logger.info("[5/6] Warming up Ollama...")
    if not warmup_ollama():
        logger.error("Cannot proceed without Ollama")
        return
    
    logger.info("[6/6] Starting processing...")
    
    # Main processing loop
    logger.info("\n" + "="*70)
    logger.info("PROCESSING REVIEWS")
    logger.info("="*70)
    
    start_time = time.time()
    total_reviews = len(unannotated)
    processed_count = 0
    skipped_count = 0
    total_triplets = 0
    batch_records = []
    
    # Restore checkpoint counts if available
    if checkpoint:
        processed_count = checkpoint.get("processed_count", 0)
        total_triplets = checkpoint.get("total_triplets", 0)
        logger.info(f"Restored counts: {processed_count} processed, {total_triplets} triplets")
    
    try:
        for idx, (_, row) in enumerate(unannotated.iterrows(), 1):
            
            review_id = row["review_id"]
            
            try:
                # Extract triplets
                triplets = extract_triplets_from_ollama(row["review_text"], review_id)
                
                if triplets:
                    processed_count += 1
                    total_triplets += len(triplets)
                    
                    # Create records for batch write (with provenance columns)
                    for triplet in triplets:
                        record = {
                            "review_id": review_id,
                            "hotel": row.get("hotel_name", ""),
                            "review_text": row["review_text"],
                            "aspect": triplet["aspect"],
                            "opinion": triplet["opinion"],
                            "sentiment": triplet["sentiment"],
                            "model": MODEL,
                            "timestamp": datetime.now().isoformat()
                        }
                        batch_records.append(record)
                else:
                    skipped_count += 1
                
                # Save checkpoint periodically
                if idx % CHECKPOINT_INTERVAL == 0:
                    save_checkpoint(review_id, processed_count, total_triplets)
                
                # Batch save every N reviews
                if idx % BATCH_SAVE_INTERVAL == 0 or idx == total_reviews:
                    if batch_records:
                        batch_df = pd.DataFrame(batch_records)
                        append_batch_to_csv(batch_df)
                        batch_records = []
                    
                    eta = estimate_eta(processed_count, total_reviews, start_time)
                    avg_triplets = total_triplets / processed_count if processed_count > 0 else 0
                    
                    print(f"\nProgress: {idx}/{total_reviews} ({idx/total_reviews*100:.1f}%)")
                    print(f"  Processed: {processed_count} | Skipped: {skipped_count}")
                    print(f"  Total triplets: {total_triplets} | Avg per review: {avg_triplets:.2f}")
                    print(f"  ETA: {eta}")
                    
                    logger.info(
                        f"Batch checkpoint: {idx}/{total_reviews}, "
                        f"Triplets: {total_triplets}, ETA: {eta}"
                    )
            
            except Exception as e:
                logger.error(f"Error processing review {review_id}: {e}")
                skipped_count += 1
                # Save checkpoint on error to allow resuming
                save_checkpoint(review_id, processed_count, total_triplets)
                continue
    
    except KeyboardInterrupt:
        logger.warning("\n✓ Processing interrupted by user (Ctrl+C)")
        # Save checkpoint and remaining batch
        if batch_records:
            batch_df = pd.DataFrame(batch_records)
            append_batch_to_csv(batch_df)
        save_checkpoint(review_id, processed_count, total_triplets) if 'review_id' in locals() else None
        print("\n✓ Progress saved. Resume later.")
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        # Save checkpoint on unexpected error
        if 'review_id' in locals():
            save_checkpoint(review_id, processed_count, total_triplets)
    
    finally:
        # Save any remaining batch
        if batch_records:
            batch_df = pd.DataFrame(batch_records)
            append_batch_to_csv(batch_df)
        
        # Clear checkpoint if all reviews processed
        if processed_count >= total_reviews:
            clear_checkpoint()
        
        # Final stats and combination
        logger.info("\n" + "="*70)
        logger.info("FINALIZING")
        logger.info("="*70)
        
        logger.info(f"Processed: {processed_count} reviews")
        logger.info(f"Skipped: {skipped_count} reviews")
        logger.info(f"Total weak triplets: {total_triplets}")
        
        avg_triplets_final = total_triplets / processed_count if processed_count > 0 else 0
        logger.info(f"Average triplets per review: {avg_triplets_final:.2f}")
        
        combine_datasets(manual_df)


def combine_datasets(manual_df):
    """Combine manual and weak triplets into single file."""
    try:
        if not os.path.exists(WEAK_LABELS_PATH):
            logger.warning("No weak triplets found")
            return
        
        weak_df = pd.read_csv(WEAK_LABELS_PATH)
        combined = pd.concat([manual_df, weak_df], ignore_index=True)
        combined.to_csv(COMBINED_PATH, index=False)
        
        logger.info("\n" + "="*70)
        logger.info("STAGE 2B COMPLETE")
        logger.info("="*70)
        logger.info(f"Manual triplets: {len(manual_df)}")
        logger.info(f"Weak triplets: {len(weak_df)}")
        logger.info(f"Total combined: {len(combined)}")
        logger.info(f"✓ Combined file: {COMBINED_PATH}")
        logger.info("\nNext: Implement GAS Stage 3 with combined triplets")
        
        print(f"\n✓ All complete!")
        print(f"  Manual: {len(manual_df)}")
        print(f"  Weak: {len(weak_df)}")
        print(f"  Total: {len(combined)}")
        print(f"  Saved: {COMBINED_PATH}")
    
    except Exception as e:
        logger.error(f"Error combining datasets: {e}")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()