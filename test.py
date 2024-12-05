import openai
import json
from openai import OpenAI
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import pickle


api_key = "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"
# api_key = "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"


client = OpenAI(api_key=api_key)


def get_embedding(text: str, model="text-embedding-3-small", **kwargs):
	# replace newlines, which can negatively affect performance.
	text = text.replace("\n", " ")

	response = client.embeddings.create(input=[text], model=model, **kwargs)

	return response.data[0].embedding


def parse_set(set_str):
	"""
	Parse a string of comma-separated integers into a set of integers.
	:param set_str: A string like '144,187,73,...'
	:return: A set of integers
	"""
	return set(map(int, set_str.split(',')))

def greedy_set_cover(sets):
	"""
	Find an approximate minimal set cover using the greedy algorithm.
	:param sets: List of sets, where each set represents a collection of elements.
	:return: A set of elements representing the approximate minimal set cover.
	"""
	# Flatten all elements and create a frequency map
	element_sets = {}
	for i, s in enumerate(sets):
		for element in s:
			if element not in element_sets:
				element_sets[element] = set()
			element_sets[element].add(i)
	
	uncovered_sets = set(range(len(sets)))
	selected_elements = set()

	while uncovered_sets:
		# Select the element that covers the most uncovered sets
		best_element = max(element_sets.items(), key=lambda x: len(x[1] & uncovered_sets))[0]
		selected_elements.add(best_element)
		# Remove covered sets
		uncovered_sets -= element_sets[best_element]
		# Remove the selected element from consideration
		del element_sets[best_element]
	
	return selected_elements


# sets = []
# dataset_sentence = load_dataset('jhu-clsp/AnaloBench', f'T2S10')['train']
# data_loader_sentence = DataLoader(dataset=dataset_sentence, batch_size=1, shuffle=False)
# for sample in data_loader_sentence:
# 	set_str = sample["Indices"][0]  # Assuming Indices is a string of comma-separated numbers
# 	sets.append(parse_set(set_str))
# # senteces = senteces[10:]
# # selected_sentences = [sentences[i] for i in idx]


# minimal_cover = greedy_set_cover(sets)
# print("Approximate Minimal Set Cover:", minimal_cover)
# print(len(minimal_cover))
import pickle

# with open('sentences_emb.pkl', 'rb') as f:
#     sentences_emb = pickle.load(f)

def preprocess_options(options_str):
	splits = ['A.', 'B.', 'C.', 'D.']
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



with open('stories_emb_new.pkl', 'rb') as f:
	stories_emb = pickle.load(f)

dataset = load_dataset('jhu-clsp/AnaloBench', f'T1S10-Subset')['train']
questions, options = preprocess_data(dataset)

unique_questions = set(questions)
unique_options = set(options)
unique_stories = list(unique_questions.union(unique_options))

cnt = 0
for story in unique_stories:
	if story not in stories_emb:
		emb = get_embedding(text=story)
		stories_emb[story] = emb


with open('stories_emb_new.pkl', 'wb') as f:
	pickle.dump(stories_emb, f)

# print(sentences_emb)
# print(stories_emb)
	
