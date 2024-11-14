import sys
import re
from collections import defaultdict

def load_data(file_path):
    with open(file_path, 'r') as file:
        return file.readlines()

def split_word_tag(word_tag):
    if '/' in word_tag:
        return word_tag.rsplit("/", 1)
    return None, None

def calculate_baseline_accuracy(train_data, test_data):
    # Count word-tag occurrences
    word_tag_counts = defaultdict(lambda: defaultdict(int))
    total_tag_counts = defaultdict(int)

    for line in train_data:
        words_tags = line.strip().split()
        for word_tag in words_tags:
            word, tag = split_word_tag(word_tag)
            if tag:  # Ensure valid tag
                word_tag_counts[word][tag] += 1
                total_tag_counts[tag] += 1

    # Find the most common tag for each word
    word_to_most_common_tag = {}
    for word, tags in word_tag_counts.items():
        if tags:  # Ensure tags are available
            most_common_tag = max(tags.items(), key=lambda x: x[1])[0]
            word_to_most_common_tag[word] = most_common_tag

    # Identify the most common tag overall for fallback
    most_common_tag = max(total_tag_counts.items(), key=lambda x: x[1])[0]

    # Prepare test sentences and true tags
    test_sentences = [re.sub(r"/[A-Z]+", "", line).strip().split() for line in test_data]
    true_tags = [
        [split_word_tag(word_tag)[1] for word_tag in line.strip().split() if '/' in word_tag]
        for line in test_data
    ]

    # Generate baseline tags
    baseline_tags = []
    for sentence in test_sentences:
        sentence_tags = []
        for word in sentence:
            # Use the most common tag for known words or fallback to the most common overall tag
            sentence_tags.append(word_to_most_common_tag.get(word, most_common_tag))
        baseline_tags.append(sentence_tags)

    # Calculate accuracy
    accuracy = calculate_accuracy(baseline_tags, true_tags)
    print(f"Baseline Accuracy: {accuracy:.2f}%")
    return accuracy

def calculate_accuracy(predicted, true):
    total_tags = sum(len(tags) for tags in true)
    correct_tags = sum(
        1 for sent_pred, sent_true in zip(predicted, true)
        for pred_tag, true_tag in zip(sent_pred, sent_true) if pred_tag == true_tag
    )
    return (correct_tags / total_tags) * 100 if total_tags > 0 else 0.0

def main(train_file, test_file):
    train_data = load_data(train_file)
    test_data = load_data(test_file)
    calculate_baseline_accuracy(train_data, test_data)

if __name__ == "__main__":
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    main(train_file, test_file)