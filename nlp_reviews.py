import spacy
from collections import Counter
import random

# Download and load the small English model
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'. Run 'python -m spacy download en_core_web_sm' once.")
    # Assuming the user runs the download command or has it available
    # For demonstration, we'll continue with the assumption of a loaded model.

# Simulated Amazon Review Data
amazon_reviews = [
    "I absolutely love the new 'Echo Dot'! The sound quality is great, but the delivery was late.",
    "This 'PowerBank 5000' is truly awful and disappointing. It broke on the first day.",
    "Excellent customer service from 'Aperture Corp' when my 'Zeta Phone' arrived damaged. The product itself is fine.",
    "The 'SmartWatch Pro' is amazing. Best purchase this year. Highly recommend it.",
    "Totally bad experience with 'QuickShip'. The packaging was terrible."
]

# --- 1. Named Entity Recognition (NER) ---
print("--- Named Entity Recognition (NER) ---")
entity_counter = Counter()

# Filter for entities relevant to products/brands
RELEVANT_ENTITY_TYPES = ['PRODUCT', 'ORG', 'GPE'] # ORG for brands/companies

for i, review in enumerate(amazon_reviews):
    doc = nlp(review)
    print(f"\nReview {i+1}: '{review[:50]}...'")
    
    found_entities = []
    for ent in doc.ents:
        if ent.label_ in RELEVANT_ENTITY_TYPES:
            found_entities.append((ent.text, ent.label_))
            entity_counter[ent.text] += 1
    
    if found_entities:
        print("  Extracted Entities:", found_entities)
    else:
        print("  No relevant entities found.")

print("\n--- Summary of Top Entities ---")
for entity, count in entity_counter.most_common(5):
    print(f"'{entity}': {count} mentions")


# --- 2. Sentiment Analysis (Rule-Based) ---

# Define a simple custom lexicon
POSITIVE_WORDS = {'love', 'great', 'excellent', 'amazing', 'best', 'recommend', 'fine'}
NEGATIVE_WORDS = {'awful', 'disappointing', 'broke', 'bad', 'damaged', 'late', 'terrible'}

def rule_based_sentiment(text):
    """
    Performs sentiment analysis based on a simple rule:
    Positive if (Positive Count > Negative Count), Negative otherwise.
    """
    doc = nlp(text.lower()) # Lowercase for consistent matching
    
    pos_count = 0
    neg_count = 0
    
    # Iterate through tokens/words (not entities)
    for token in doc:
        if token.text in POSITIVE_WORDS:
            pos_count += 1
        elif token.text in NEGATIVE_WORDS:
            neg_count += 1
            
    if pos_count > neg_count:
        sentiment = "Positive"
    elif neg_count > pos_count:
        sentiment = "Negative"
    else:
        sentiment = "Neutral" # Or simply default to one if the rule is strict

    return sentiment, pos_count, neg_count

print("\n--- Rule-Based Sentiment Analysis ---")
for i, review in enumerate(amazon_reviews):
    sentiment, pos_c, neg_c = rule_based_sentiment(review)
    print(f"Review {i+1}: '{review[:50]}...'")
    print(f"  Sentiment: **{sentiment}** (Pos: {pos_c}, Neg: {neg_c})")