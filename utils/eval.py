from utils.data import preprocess_data, universal_idx, query_expansion
import numpy as np

def evaluate_ranking(dataset, pred_func, summary_level="keyword", query_only=False, summary_only=False):

    questions, options = preprocess_data(dataset)
    if summary_level is not None:
        sentence2idx, story2idx, sentence_summary, story_summary = universal_idx()

    print(len(options))
    unique_options = list(set(options))
    print(len(unique_options))
    unique_options_to_idx = {option: idx for idx, option in enumerate(unique_options)}
    options_indices = [unique_options_to_idx[option] for option in options]

    if summary_level is not None and (not query_only):
       unique_options = query_expansion(unique_options, dataset, summary_level, sentence2idx, story2idx, sentence_summary, story_summary, summary_only=summary_only)
    
    print(len(questions))
    unique_questions = list(set(questions))
    print(len(unique_questions))
    unique_questions_to_idx = {question: idx for idx, question in enumerate(unique_questions)}
    questions_indices = [unique_questions_to_idx[question] for question in questions]

    if summary_level is not None:
        unique_questions = query_expansion(unique_questions, dataset, summary_level, sentence2idx, story2idx, sentence_summary, story_summary, summary_only=summary_only)

    predicted_labels, similarities = pred_func(unique_questions, unique_options, questions_indices=questions_indices, options_indices=options_indices)
    
    labels = dataset['Label']  # The index of the correct option
    labels = np.array([ord(label) - ord('A') for label in labels])
    total_samples = len(labels)
    correct = (predicted_labels == labels)
    precision_at_1 = sum(correct) / total_samples

    incorrect_sample_indices = []
    correct_sample_indices = []
    for i, crr in enumerate(correct):
        if not crr:
            incorrect_sample_indices.append(i)
        else:
            correct_sample_indices.append(i)

    return precision_at_1, incorrect_sample_indices, correct_sample_indices, predicted_labels, similarities
