import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import re

# Create output folder
os.makedirs('data', exist_ok=True)

print("="*60)
print("STAGE 1: DATA PREPARATION")
print("="*60)

# STEP 1: Load Excel file
print("\n[1/5] Loading Excel file...")
file_path = "F:\\Research\\data\\Hotel Review Data Table.xlsx"  # Update this path if needed

try:
    xl_file = pd.ExcelFile(file_path)
    print(f"Found sheets: {xl_file.sheet_names}")
    
    # Load all sheets and combine
    dfs = []
    for sheet in xl_file.sheet_names:
        df = pd.read_excel(file_path, sheet_name=sheet)
        df['hotel_name'] = sheet
        dfs.append(df)
    
    data = pd.concat(dfs, ignore_index=True)
    print(f"✓ Loaded {len(data)} reviews from {len(dfs)} hotels")
except FileNotFoundError:
    print(f"✗ File not found: {file_path}")
    print("Make sure Hotel_Review_Data_Table.xlsx is in the same folder as this script")
    exit(1)

# STEP 2: Clean text
print("\n[2/5] Cleaning text...")

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r'\s+', ' ', text)  # Multiple spaces → single space
    text = text.strip()
    return text

data['review_text'] = data['Review'].apply(clean_text)
data['reviewer_name'] = data.get('Reviewer Name', 'Anonymous').fillna('Anonymous')

# Remove empty reviews
data = data[data['review_text'].str.len() > 0]
print(f"✓ After cleaning: {len(data)} reviews (removed {len(dfs) - len(data)} empty ones)")

# STEP 3: Check for duplicates
print("\n[3/5] Checking for duplicates...")
duplicates = data.duplicated(subset=['review_text']).sum()
print(f"Found {duplicates} duplicate reviews")
if duplicates > 0:
    data = data.drop_duplicates(subset=['review_text'])
    print(f"✓ Removed duplicates. Now: {len(data)} reviews")

# STEP 4: Split dataset
print("\n[4/5] Splitting into train/val/test...")
train, temp = train_test_split(data, test_size=0.30, random_state=42)
val, test = train_test_split(temp, test_size=0.50, random_state=42)

print(f"Train: {len(train)} ({len(train)/len(data)*100:.1f}%)")
print(f"Val:   {len(val)} ({len(val)/len(data)*100:.1f}%)")
print(f"Test:  {len(test)} ({len(test)/len(data)*100:.1f}%)")

# STEP 5: Save to CSV
print("\n[5/5] Saving to CSV...")

train.to_csv("F:\\Research\\data\\train.csv", index=False)
val.to_csv("F:\\Research\\data\\val.csv", index=False)
test.to_csv("F:\\Research\\data\\test.csv", index=False)

print(f"✓ train.csv ({len(train)} rows)")
print(f"✓ val.csv ({len(val)} rows)")
print(f"✓ test.csv ({len(test)} rows)")

# Summary
print("\n" + "="*60)
print("STAGE 1 COMPLETE!")
print("="*60)
print(f"Total reviews: {len(data)}")
print(f"Hotels: {data['hotel_name'].nunique()}")
print(f"\nFiles created in 'data/' folder:")
print(f"  - train.csv")
print(f"  - val.csv")
print(f"  - test.csv")
print("\n✓ Ready for Stage 2: Annotation")