import openai
import json
from openai import OpenAI
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import pickle


api_key = "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"
# api_key = "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"


client = OpenAI(api_key=api_key)

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



def get_embedding(text: str, model="text-embedding-3-large", **kwargs):
    # replace newlines, which can negatively affect performance.
    text = text.replace("\n", " ")

    response = client.embeddings.create(input=[text], model=model, **kwargs)

    return response.data[0].embedding

if __name__ == '__main__':
	# sentences_emb  = {}
	# sentences = []
	# dataset_sentence = load_dataset('jhu-clsp/AnaloBench', f'T1S1-Subset')['train']
	# data_loader_sentence = DataLoader(dataset=dataset_sentence, batch_size=1, shuffle=False)
	# for sample in data_loader_sentence:
	# 		sentences.append(sample["Sentence"][0])
	# # sentences = sentences[:1]

	# for sentence in sentences:
	# 	sentences_emb[sentence] = get_embedding(text=sentence)

	# with open('sentences_emb.pkl', 'wb') as f:
	# 	pickle.dump(sentences_emb, f)

	# sentence_emb = {}

	# sentences = []
	# dataset = load_dataset('jhu-clsp/AnaloBench', f'T1S1-Subset')['train']
	# questions, options = preprocess_data(dataset)

	# unique_questions = set(questions)
	# unique_options = set(options)
	# unique_sentences = list(unique_questions.union(unique_options))
	# for sentence in unique_sentences:
	# 	sentence_emb[sentence] = get_embedding(text=sentence)

	# with open('sentence_emb_large.pkl', 'wb') as f:
	# 	pickle.dump(sentence_emb, f)


	stories_emb = {}

	stories = []
	dataset = load_dataset('jhu-clsp/AnaloBench', f'T1S10-Subset')['train']
	questions, options = preprocess_data(dataset)

	unique_questions = set(questions)
	unique_options = set(options)
	unique_stories = list(unique_questions.union(unique_options))
	for story in unique_stories:
		stories_emb[story] = get_embedding(text=story)

	with open('stories_emb_large.pkl', 'wb') as f:
		pickle.dump(stories_emb, f)



