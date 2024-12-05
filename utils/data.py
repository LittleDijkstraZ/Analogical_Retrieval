
def preprocess_options(options_str):
    splits = ['A. ', '\nB. ', '\nC. ', '\nD. ']
    final_options = []
    for sidx in range(len(splits)):
        if sidx == len(splits) - 1:
            x = options_str.split(splits[sidx])[1]
        else:
            x = options_str.split(splits[sidx])[1].split(splits[sidx+1])[0]
        final_options.append(x.strip())
    return final_options


def preprocess_data(samples):
    if 'Story' not in samples.features:
        questions_pool = samples['Sentence']
    else:
        questions_pool = samples['Story'] # story to story
    questions = [sample for sample in questions_pool]
    all_options = [sample for sample in samples['Options']]
    all_options = [preprocess_options(options) for options in all_options]
    flattened_options = [option for options in all_options for option in options]
    return questions, flattened_options


from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

import os
import json
def load_json_files_to_dict(mode='sentence'):
    result_dict = {}
    folder_path = "./{mode}".format(mode=mode)
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".json"):
            # Extract the index from the filename (e.g., "sentence_3_summaries.json" -> 3)
            index = int(file_name.split('_')[1])
            file_path = os.path.join(folder_path, file_name)
            # Read the JSON file
            with open(file_path, 'r', encoding='utf-8') as json_file:
                result_dict[index] = json.load(json_file)
    return result_dict

def universal_idx():
    queries = []
    dataset_sentence = load_dataset('jhu-clsp/AnaloBench', f'T1S1-Subset')['train']
    data_loader_sentence = DataLoader(dataset=dataset_sentence, batch_size=1, shuffle=False)
    for sample in data_loader_sentence:
        queries.append(sample["Sentence"][0])
    sentence2idx = {sentence: idx for idx, sentence in enumerate(queries)}

    queries = []
    dataset_sentence = load_dataset('jhu-clsp/AnaloBench', f'T1S10-Subset')['train']
    data_loader_sentence = DataLoader(dataset=dataset_sentence, batch_size=1, shuffle=False)
    for sample in data_loader_sentence:
        queries.append(sample["Story"][0])
    story2idx = {story: idx for idx, story in enumerate(queries)}

    sentence_summary = load_json_files_to_dict(mode='sentence')
    story_summary = load_json_files_to_dict(mode='story')
    
    return sentence2idx, story2idx, sentence_summary, story_summary


def query_expansion(queries: list, 
                    dataset,
                    summary_level, 
                    sentence2idx, 
                    story2idx, 
                    sentence_summary, 
                    story_summary, summary_only=False):
    collected = 0
    failed_options = []
    option2idx = story2idx if 'Story' in dataset.features else sentence2idx
    option_summary = story_summary if 'Story' in dataset.features else sentence_summary
    for oidx, option in enumerate(queries):
        story_idx = option2idx.get(option, None)
        if story_idx is None:
            failed_options.append(option)
            summary_prompt = ""
        else:
            collected += 1
            summary = [ss[summary_level] for ss in option_summary[story_idx]]
            summary_prompt = " ".join(summary)
        
        # specify prompt design here
        if summary_only:
            queries[oidx] = summary_prompt
        else:
            queries[oidx] = f"{option} {summary_level}: {summary_prompt}"
    print('options: collected/total', collected, len(queries))
    return queries