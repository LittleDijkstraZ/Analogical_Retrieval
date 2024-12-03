import openai
import json
from openai import OpenAI
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader


# api_key = "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"
# api_key = "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"

import json

def convert_to_detailed_json(idx, summary, mode='story'):
	lines = summary.split('\n')
	
	# 初始化一个空列表存储结果
	result = []
	
	for line in lines:
		# 跳过空行
		if not line.strip():
			continue
		
		# 按分号分割字符串
		parts = line.split(';', 2)  # 最多分割成3部分
		
		if len(parts) == 3:
			keyword = parts[0].strip()
			keyphrase = parts[1].strip()
			summary = parts[2].strip()
			result.append({
				"keyword": keyword,
				"keyphrase": keyphrase,
				"summary": summary
			})
		else:
			print("error happened at {idx}".format(idx=idx))
			continue
	json_data = json.dumps(result, indent=4, ensure_ascii=False)
	output_file = "./{mode}/{mode}_{idx}_summaries.json".format(mode=mode, idx=idx)
	with open(output_file, "w", encoding="utf-8") as f:
		f.write(json_data)


def generate_summary_story(stories):
	prompt_template = """Please read the following story carefully. Your task is to identify and summarize all of its key ideas, capturing the significant underlying principles, themes, or morals. Focus on abstract concepts and relationships. In your summaries:

List All Significant Key Ideas:

Extract multiple key ideas if applicable.
Ensure each key idea represents a distinct theme, principle, or moral from the story.
Use Generalized Language:

Avoid specific details such as names, places, or unique objects.
Refer to characters and entities in general terms (e.g., "an individual," "a group," "a community").
Highlight Core Concepts and Relationships:

Focus on the main events, actions, and outcomes in broad terms.
Describe cause-and-effect relationships and the dynamics between characters or forces.
Emphasize Underlying Themes or Morals:

Include universal themes such as justice, irony, betrayal, redemption, self-discovery, consequences of actions, etc.
Consider both explicit messages and implicit lessons.
Be Clear and Concise:

Keep each key idea brief but comprehensive enough to convey its essence.
Ensure that each summary stands alone and makes sense without additional context.
For example, if the story involves someone deceiving others and then being deceived themselves, but also touches on themes of karma and loss, your summaries might be:

"Deception; Deception and Consequences; An individual’s persuasive deception led a group to misplace their trust, resulting in disillusionment and loss when the truth was revealed."
"Illusion; The Power of Illusion; Superficial appearances can easily mislead those who rely solely on what they see, highlighting the importance of critical examination."
"Greed; The Role of Greed; Desire for wealth and material gain can cloud judgment, making individuals vulnerable to exploitation."

Where first comes with a key word of the summary, then a key phrase of the summary, and then the whole summary sentence, separated by ";", and output each summary in its own line.
You should not add any other symbols like "*" "\"" and etc. 
Here is the story:

{story}
"""
	summaries = []
	client = OpenAI(api_key=api_key)
	for i, story in enumerate(stories):
		prompt = prompt_template.format(story=story)
		print(i)
		try:
			response = client.chat.completions.create(
				model="gpt-4o",
				messages=[
					{"role": "system", "content": "You are an expert in thematic analysis and analogical reasoning. Your task is to assist users in identifying and summarizing all significant key ideas from stories to facilitate analogical retrieval."},
					{"role": "user", "content": prompt}
				],
				temperature=0.3,
			)
			summary = response.choices[0].message.content
			convert_to_detailed_json(i, summary, mode="story")
			summaries.append(summary)
		except Exception as e:
			print(f"Error processing story: {story}\nError: {e}")
			summaries.append(None)
	
	return summaries


def generate_summary_sentence(sentences, idx=None):
	prompt_template = """Please read the following sentence carefully. Your task is to identify and summarize all the key ideas it conveys, capturing each underlying principle, theme, or moral in an abstract and generalized way. In your summaries:

Extract All Relevant Key Ideas:

Identify each distinct idea, message, or lesson the sentence conveys.
Recognize that a sentence may have multiple layers of meaning or themes.
Use Generalized Language:

Avoid mentioning specific names, places, or events.
Use broad terms like "someone," "people," "a person," "situations," etc.
Highlight Universal Themes or Morals:

Consider themes such as trust, deception, perseverance, caution, appearances, ethics, consequences, etc.
Emphasize lessons or principles that are widely applicable.

Be Concise and Clear:

Keep each summary brief—usually one sentence.
Ensure each summary is understandable on its own without additional context.
For example, if the sentence is:

"In for a penny, in for a pound, we need to stay up all night and get the report done since we cannot turn it in half-finished."
The summaries might be:

"Commitment; Commitment requires follow-through; when one has invested effort into something, they should see it through to completion."
"Persistence; Persistence in the face of challenges: Success often demands dedication, even when it requires sacrifices such as time or comfort."

Where first comes with a key word of the summary, then a key phrase of the summary, and then the whole summary sentence, separated by ";", and output each summary in its own line.
You should not add any other symbols like "*" "\"" and etc. 
Here is the sentence:

{sentence}
"""
	client = OpenAI(api_key=api_key)
	summaries = []
	for i, sentence in enumerate(sentences):
		prompt = prompt_template.format(sentence=sentence)
		# print(prompt)
		print(i)
		try:
			response = client.chat.completions.create(
				model="gpt-4o",
				messages=[
					{"role": "system", "content": "You are an expert in thematic analysis and analogical reasoning. Your task is to assist users in identifying and summarizing all significant key ideas from stories to facilitate analogical retrieval."},
					{"role": "user", "content": prompt}
				],
				temperature=0.5,
			)
			summary = response.choices[0].message.content
			if idx is not None:
				convert_to_detailed_json(idx[i], summary, mode="sentence")
			else:
				convert_to_detailed_json(i, summary, mode="sentence")
			summaries.append(summary)
		except Exception as e:
			print(f"Error processing story: {sentence}\nError: {e}")
			summaries.append(None)
	
	return summaries

if __name__ == '__main__':
	senteces  = []
	dataset_sentence = load_dataset('jhu-clsp/AnaloBench', f'T1S1-Subset')['train']
	data_loader_sentence = DataLoader(dataset=dataset_sentence, batch_size=1, shuffle=False)
	for sample in data_loader_sentence:
			senteces.append(sample["Sentence"][0])
	# senteces = senteces[10:]
	selected_sentences = [senteces[i] for i in idx]


	summmary_sentences = generate_summary_sentence(sentences=selected_sentences, idx=idx)
	print(summmary_sentences)

	# stories = []
	# dataset_story = load_dataset('jhu-clsp/AnaloBench', f'T1S10-Subset')['train']
	# data_loader_story = DataLoader(dataset=dataset_story, batch_size=1, shuffle=False)
	# for sample in data_loader_story:
	# 	stories.append(sample['Story'][0])
	# # stories = stories[10:]
	# summmary_story = generate_summary_story(stories=stories)
	# print(summmary_story)
