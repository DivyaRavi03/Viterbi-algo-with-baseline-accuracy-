import sys
import numpy as np
from collections import defaultdict
import re

def load_data(file_path):
    with open(file_path, 'r') as file:
        return file.readlines()

def split_word_tag(word_tag):
    if '/' in word_tag:
        return word_tag.rsplit("/", 1)
    return None, None

def compute_counts(data):
    start_tag_counts = defaultdict(int)
    tag_counts = defaultdict(int)
    tag_word_counts = defaultdict(lambda: defaultdict(int))
    tag_transition_counts = defaultdict(lambda: defaultdict(int))

    for line in data:
        words_tags = line.strip().split()
        if not words_tags:
            continue

        prev_tag = "<START>"
        for word_tag in words_tags:
            word, tag = split_word_tag(word_tag)
            if tag:
                tag_counts[tag] += 1
                tag_word_counts[tag][word] += 1
                tag_transition_counts[prev_tag][tag] += 1
                prev_tag = tag

        tag_transition_counts[prev_tag]["<END>"] += 1
        start_tag_counts[split_word_tag(words_tags[0])[1]] += 1

    return start_tag_counts, tag_counts, tag_word_counts, tag_transition_counts

def compute_probabilities(start_tag_counts, tag_counts, tag_word_counts, tag_transition_counts, num_sentences):
    initial_probs = {tag: count / num_sentences for tag, count in start_tag_counts.items()}

    transition_probs = defaultdict(dict)
    for prev_tag, next_tag_counts in tag_transition_counts.items():
        total_transitions = sum(next_tag_counts.values())
        num_tags = len(tag_counts)
        for next_tag in tag_counts.keys():
            transition_probs[prev_tag][next_tag] = (next_tag_counts[next_tag] + 1) / (total_transitions + num_tags)
        transition_probs[prev_tag]["<END>"] = (next_tag_counts["<END>"] + 1) / (total_transitions + num_tags)

    emission_probs = defaultdict(dict)
    for tag, word_counts in tag_word_counts.items():
        total_words_for_tag = sum(word_counts.values())
        num_words = len(word_counts)
        for word in word_counts:
            emission_probs[tag][word] = (word_counts[word] + 1) / (total_words_for_tag + num_words)

    return initial_probs, transition_probs, emission_probs

def viterbi_algorithm(sentence, initial_probs, transition_probs, emission_probs, tag_list):
    num_tags = len(tag_list)
    len_sentence = len(sentence)
    score = np.full((num_tags, len_sentence), -np.inf)
    backpointer = np.zeros((num_tags, len_sentence), dtype=int)

    for t in range(num_tags):
        tag = tag_list[t]
        score[t, 0] = np.log(initial_probs.get(tag, 1e-6)) + np.log(emission_probs[tag].get(sentence[0], 1e-6))

    for w in range(1, len_sentence):
        for t in range(num_tags):
            tag = tag_list[t]
            max_score, best_prev_tag = max(
                (score[j, w - 1] + np.log(transition_probs[tag_list[j]].get(tag, 1e-6)) +
                 np.log(emission_probs[tag].get(sentence[w], 1e-6)), j)
                for j in range(num_tags)
            )
            score[t, w] = max_score
            backpointer[t, w] = best_prev_tag

    best_last_tag = np.argmax(score[:, len_sentence - 1])
    best_path = [best_last_tag]

    for w in range(len_sentence - 1, 0, -1):
        best_last_tag = backpointer[best_last_tag, w]
        best_path.insert(0, best_last_tag)

    return [tag_list[i] for i in best_path]

def calculate_accuracy(predicted, true):
    total_tags = sum(len(tags) for tags in true)
    correct_tags = sum(
        1 for sent_pred, sent_true in zip(predicted, true)
        for pred_tag, true_tag in zip(sent_pred, sent_true) if pred_tag == true_tag
    )
    return (correct_tags / total_tags) * 100 if total_tags > 0 else 0.0

def extract_true_tags(test_data):
    true_tags = []
    for line in test_data:
        tags = [split_word_tag(word_tag)[1] for word_tag in line.strip().split() if '/' in word_tag]
        true_tags.append(tags)
    return true_tags

def main(train_file, test_file):
    train_data = load_data(train_file)
    start_tag_counts, tag_counts, tag_word_counts, tag_transition_counts = compute_counts(train_data)
    num_sentences = len(train_data)
    initial_probs, transition_probs, emission_probs = compute_probabilities(
        start_tag_counts, tag_counts, tag_word_counts, tag_transition_counts, num_sentences
    )

    test_data = load_data(test_file)
    test_sentences = [re.sub(r"/[A-Z]+", "", line).strip().split() for line in test_data]
    true_tags = extract_true_tags(test_data)

    unique_tags = list(tag_counts.keys())
    predicted_tags = [
        viterbi_algorithm(sentence, initial_probs, transition_probs, emission_probs, unique_tags)
        for sentence in test_sentences
    ]

    accuracy = calculate_accuracy(predicted_tags, true_tags)
    print(f"Accuracy: {accuracy:.2f}%")

    with open("POS.test.out", 'w') as output_file:
        for sentence, tags in zip(test_sentences, predicted_tags):
            tagged_sentence = " ".join([f"{word}/{tag}" for word, tag in zip(sentence, tags)])
            output_file.write(tagged_sentence + "\n")

if __name__ == "__main__":
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    main(train_file, test_file)