# ASTE: Aspect-Opinion-Sentiment Triplet Extraction from Hotel Reviews

Aspect-Opinion-Sentiment Triplet Extraction from South Asian hotel reviews using weak supervision.

## What is This?

I built a system to automatically extract structured opinions from hotel reviews. Instead of just knowing "the review is positive," I can extract specific insights like "the room is spacious and guests like it" or "WiFi is slow and frustrating."

The main idea: combine human annotation (careful but slow) with AI labeling (fast but noisy) to create training data. Then fine-tune a transformer model to extract these opinion triplets.

**Why it matters:**
- First ASTE dataset for South Asian hotels
- Shows that hybrid labeling (manual + LLM) works for NLP tasks
- Gives hotel owners actionable feedback on what guests like/dislike

**The honest part:** Opinion detection is only 47% accurate right now. I document this upfront because it's important to know what works and what doesn't.

---

## The Dataset

I collected 3,727 reviews from 8 hotels in Dhaka, Bangladesh:
- Hotel Sarina Dhaka (1,584 reviews)
- Lakeshore Banani (833 reviews)
- THE WAY Dhaka (753 reviews)
- Dhaka Regency (580 reviews)
- And 4 others...

### What I'm Extracting

I focus on 8 aspects guests actually talk about:
1. **Room** - Is it spacious? Clean? Good view?
2. **Staff** - Are they friendly? Professional?
3. **Food** - Good breakfast? Restaurant quality?
4. **Location** - Near airport? Good neighborhood?
5. **Cleanliness** - Is everything clean?
6. **Value** - Is it worth the price?
7. **WiFi** - How's the internet?
8. **Amenities** - Pool? Gym? Parking?

Each triplet looks like: `(aspect, opinion, sentiment)`

Example: `(room, spacious, POSITIVE)`

---

## How I Built This

### Stage 1: Clean the Data
Started with messy Excel files. Removed duplicates, empty reviews, fixed formatting. Ended up with clean train/val/test splits (70/15/15).

### Stage 2A: Manual Annotation
Built a simple Streamlit web interface and manually labeled ~300 reviews. For each review, I selected the aspect, the opinion word, and whether it's positive/negative/neutral. Got 1,216 unique triplets this way.

It's slow (takes hours), but the labels are accurate.

### Stage 2B: Weak Labeling with Ollama
For the remaining 2,300 reviews, I used a local LLM (Ollama + Mistral). Instead of paying OpenAI, I ran the model locally on my machine.

Gave it strict instructions: "Only extract aspects that are Room, Staff, Food, etc. Don't make up opinions. Only use words from the review."

Got another 2,000-3,000 weak labels. They're noisier than manual labels, but better than nothing.

Combined manual + weak labels = 5,000+ triplets for training.

### Stage 3: Train the Model
Used DeBERTa (a BERT variant) to do token classification. The idea is simple: for each word in the review, predict if it's the start/middle of an aspect, opinion, or neither.

Used BIO tagging (Begin-Inside-Outside scheme).

Training took 3 days on GPU with FP16 precision. Early stopping kicked in around epoch 3.

### Stage 4 & 5: (In Progress)
Will extract triplets from all 3,727 reviews, analyze what guests like/dislike, and build a dashboard.

---

## Results

**Good news:**
- Weighted F1: 94.03% - The model is pretty good at finding aspects
- Precision: 93.68% - When it predicts something, it's usually right
- Recall: 94.55% - Catches most aspects

**Bad news:**
- Opinion detection only works 47% of the time
- Multi-word opinions are often missed
- Class imbalance is brutal (92.7% of tokens are "outside any entity")

### What Went Wrong

Looking at the confusion matrix, the model struggles with opinions. It predicts "not an opinion" way too often.

Why? Probably:
1. The training data is imbalanced (mostly non-entity tokens)
2. Weak labels from Ollama probably missed some opinions
3. BIO tagging might not be the best approach for this task

I documented this because it's real. The model works for aspect extraction but is mediocre at opinion extraction.

---

## Installation

### Requirements
- Python 3.8+
- 32GB RAM (for training)
- 8GB VRAM if using GPU (CUDA 11.0+)

### Setup

```bash
# Clone and setup
git clone https://github.com/Tanzim-prog/aste_hotel_reviews
cd aste-hotel-reviews

# Virtual environment
python -m venv aste_env
source aste_env/bin/activate  # Linux/Mac
# or: aste_env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Optional: Setup Ollama for weak labeling
# Download from https://ollama.ai
ollama pull mistral
ollama serve  # Keep running in background
```

---

## Usage

### Data Preparation
```bash
python stage1_data_preparation.py
```

### Manual Annotation (Web GUI)
```bash
streamlit run stage2a_annotation_gui.py
# Opens http://localhost:8501
```

### Weak Labeling (Requires Ollama)
```bash
python stage2b_ollama_weak_labeling.py
# Runs for ~1 hour
# Auto-resumes if interrupted
```

### Train the Model
```bash
python stage3_deberta_aste.py
# Takes ~3 days on GPU
# Auto-saves checkpoints every 200 steps
```

### Extract Triplets
```bash
python stage4_triplet_extraction.py
# Output: extracted_triplets.json
```

### Analyze Results
```bash
python stage5_analysis.py
streamlit run stage5_dashboard.py
```

---

## Project Structure
aste-hotel-reviews/
├── code/
│ ├── stage1_data_preparation.py
│ ├── stage2a_annotation_gui.py
│ ├── stage2b_ollama_weak_labeling.py
│ ├── stage3_deberta_aste.py
│ ├── stage4_triplet_extraction.py
│ └── stage5_analysis.py
│
├── data/
│ ├── raw/
│ │ └── Hotel_Review_Data_Table.xlsx
│ ├── processed/
│ │ ├── train.csv
│ │ ├── val.csv
│ │ ├── test.csv
│ │ ├── manual_triplets.csv
│ │ ├── weak_triplets.csv
│ │ └── all_triplets.csv
│ └── statistics/
│ └── dataset_statistics.json
│
├── model/
│ └── aste_deberta/
│ ├── pytorch_model.bin
│ ├── config.json
│ ├── confusion_matrix.png
│ ├── classification_report.json
│ ├── training_history.csv
│ └── experiment_summary.txt
│
├── results/
│ ├── extracted_triplets.json
│ ├── aspect_sentiment_scorecard.csv
│ └── hotel_rankings.csv
│
├── requirements.txt
├── README.md
└── notebooks/
├── 01_exploratory_analysis.ipynb
└── 02_results_visualization.ipynb
## How I Evaluate Performance

### The Numbers
- **Weighted F1: 0.9403** - Good overall
- **Macro F1: 0.6653** - Bad on minority classes
- **Precision: 0.9368** - Few false positives
- **Recall: 0.9455** - Catches most things

### Per-Tag Breakdown
| Tag | Precision | Recall | Notes |
|-----|-----------|--------|-------|
| Outside (O) | 0.98 | 0.98 | Works great |
| Begin Aspect | 0.96 | 0.97 | Works great |
| Begin Opinion | 0.61 | 0.47 | Problem area |
| Inside Opinion | 0.21 | 0.19 | Major problem |

### What the Confusion Matrix Tells Me
The model is conservative. It often says "this isn't an entity" when it actually is. That's why recall is low for opinions.

Interestingly, it almost never confuses aspects with opinions (only 3 errors). So when it finds an aspect, that's probably right.

---

## Known Problems

**Opinion Detection Sucks**
- Only catches 47% of opinion phrases
- Multi-word opinions are especially missed
- Class imbalance is the main culprit

**Not Optimized for This Task**
- BIO tagging treats aspects/opinions as independent
- Doesn't explicitly model aspect-opinion pairing
- Other ASTE methods (like Span-ASTE) might work better

**Limited Dataset**
- Only 8 hotels in one city
- Weak labels add noise (~10-15% estimated error)
- No comparison with other methods

**Data Issues**
- Long reviews get truncated at 384 tokens
- Some reviews have NaN values in the data
- Sentiment labels sometimes inferred, not explicit

---

## What I'd Do Differently (Next Time)

1. **Retrain with weighted loss** - Give more weight to entity classes. Should improve opinion recall significantly.

2. **Try Span-ASTE instead** - BIO tagging might be the bottleneck. Span-based methods might work better.

3. **Get more training data** - 5K triplets is okay but 10K+ would be better.

4. **Validate manually** - Have multiple people label some reviews to check label quality.

5. **Compare baselines** - Should've compared against other ASTE methods from the start.

---

## For Research/Publication

If you're using this for a thesis or paper, here's how to cite it:

```bibtex
@article{yourname2024aste,
  title={Aspect-Opinion-Sentiment Triplet Extraction from South Asian Hotel Reviews Using Weak Supervision},
  author={Your Name},
  journal={Expert Systems with Applications},
  year={2024},
  institution={AIUB}
}
```

Or just mention:
"We used the ASTE hotel reviews dataset from [link to repo]"

---

## FAQ

**Q: Why didn't you just use GPT-4?**
A: Cost. Running Ollama locally was free. Also, wanted to explore weak supervision with open-source tools.

**Q: Can I use this on other hotels?**
A: Sure, but it's trained on Dhaka hotels. Results might vary on different regions or languages.

**Q: Why is the opinion detection so bad?**
A: Mostly class imbalance. 92.7% of tokens are "not an opinion," so the model is biased to predict that.

**Q: How long does training take?**
A: 3 days on NVIDIA GPU with FP16. On CPU? Probably a week. Not recommended.

**Q: Can I resume if it crashes?**
A: Yes. Checkpoints save every 200 steps. Just run the script again.

**Q: How do I add more reviews?**
A: Put new review files in `data/raw/`, run stage 1 to clean them, then stage 2 to label them.

**Q: Should I use this in production?**
A: Not yet. Opinion detection is too weak. Good for research, not for business decisions.

---

## Things I Learned

Building this taught me a lot:
- Weak supervision is powerful but noisy
- Class imbalance breaks models in subtle ways
- Token alignment is tricky (subword tokens vs. original tokens)
- Local LLMs (Ollama) are surprisingly good for constrained tasks
- BIO tagging has limitations for extracting relationships

---
