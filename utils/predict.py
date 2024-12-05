# model_name = 'multi-qa-mpnet-base-dot-v1'
# model_name = 'paraphrase-MiniLM-L12-v2'

# model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

import numpy as np
import torch


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



def predict_labels(questions, options, model, bs=256, questions_indices=None, options_indices=None):
    question_embeddings = model.encode(questions, show_progress_bar=True, batch_size=bs, convert_to_tensor=True)
    question_embeddings = question_embeddings.reshape(len(questions), -1)
    # question_embeddings = np.repeat(question_embeddings, 4, axis=0)
    if options_indices is None:
        question_embeddings = torch.repeat_interleave(question_embeddings, 4, dim=0)
    option_embeddings = model.encode(options, show_progress_bar=True, batch_size=bs, convert_to_tensor=True)
    print(question_embeddings.shape, option_embeddings.shape)


    if options_indices is None:
        similarities = []
        print('cossim')
        for qidx in tqdm(range(0, len(question_embeddings), bs)):
            sim = torch.nn.functional.cosine_similarity(question_embeddings[qidx:qidx+bs], option_embeddings[qidx:qidx+bs], dim=-1)
            similarities.append(sim)
        similarities = torch.hstack(similarities)
    else:
        matrix = question_embeddings @ option_embeddings.T / (question_embeddings.norm(dim=-1)[:, None] * option_embeddings.norm(dim=-1)[None, :])
        matrix = matrix.cpu()
        questions_indices = torch.tensor(questions_indices).repeat_interleave(4)
        similarities = matrix[questions_indices, options_indices]
        print(similarities.shape)

    
    similarities = similarities.reshape(-1, 4)
    print('sorting')
    # ranked_indices = torch.argsort(similarities, dim=-1, descending=True)
    predicted_labels = torch.argmax(similarities, dim=-1).cpu().numpy()
    print('sorting finished')
    return predicted_labels, similarities



def predict_labels_FollowIR(quesitons, options, model, tokenizer, bs=32):

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    token_false_id = tokenizer.get_vocab()["false"]
    token_true_id = tokenizer.get_vocab()["true"]
    template = """<s> [INST] Consider the Document to be relevant only if it can be analogous to the Query. Answer in (true/false)

    Query: {query}
    Document: {text}
    Analogically Relevant (only output one word, either "true" or "false"):  [/INST] """

    # assert bs % 4 == 0, "Batch size must be a multiple of 4"
    scores = []
    for bsidx in tqdm(range(0, len(options), bs)):
        cur_options = options[bsidx:bsidx+bs]
        cur_questions = [quesitons[opidx//4] for opidx in range(bsidx, bsidx+bs)]

        prompts = [
            template.format(query=query, text=text) for (query, text) in zip(cur_questions, cur_options)
        ]
        tokens = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            pad_to_multiple_of=None,
        )

        # move to cuda if desired
        for key in tokens:
            tokens[key] = tokens[key].cuda()

        # calculate the scores by comparing true and false tokens
        batch_scores = model(**tokens).logits[:, -1, :]
        true_vector = batch_scores[:, token_true_id]
        false_vector = batch_scores[:, token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        cur_scores = batch_scores[:, 1].exp().tolist()
        scores.extend(cur_scores)

    print(len(cur_scores))
    scores = np.array(scores)
    scores = scores.reshape(len(quesitons), 4)
    predicted_labels = np.argmax(scores, axis=1)
    return predicted_labels


def predict_labels_Promptriever(questions, options, model, bs=4, questions_indices=None, options_indices=None):

    instruction = "" 
    # 6471, 5912 (6293, 2423, 3299)
    # instruction = ""    
    # 5647 -> 5971 kw, 6176 kp, 5971 ks
    # keyphrase:
    # all: 6176
    # query only: 5588
    # query and summary only: 4471
    # both summary only: 4824
    # summary: 
    # summary only: 5206
    # keyword:
    # keyword only: 4235

    # FuLL:
    # 2583, 2621, 2366


    
    # instruction = "A relevant document would be the most analogous to the query. I don't care about semantic similarity. Think carefully about these conditions when determining relevance."
    # 6588, 5824
    # instruction = "A relevant document would be the most analogous to the query. I don't care about semantic similarity."
    # 6735, 5912
    # instruction = "A relevant document would be the most analogous to the query. I don't care about semantic similarity. Ignore concrete details and focus on the relational structure and abstract meaning."
    # 6265, 5824
    # instruction = "A relevant document would be the most analogous to the query. I care about abstraction. I don't care about semantic similarity or concrete details."
    # 6353, 5824
    # instruction = "A relevant document would be the most analogous to the query. Think carefully about these conditions when determining relevance."
    # 6647, 5794
    # instruction = "Instruction: A relevant document would be the most analogous to the query."
    # 6500, 5676

    # instruction = "Summarize the key concepts behind this query."
    # 6176, 5676
    # instruction = "Consider what documents could be analogous to this query."
    # 6382, 5941
    # instruction = "I don't care about semantic similarity."
    # 6118, 5706

    # instruction = "Happy Thanksgiving."
    # 6000, 5324
    # instruction = "Analogical sense. Analogical similarity. Not semantic similarity."
    # 5706, 5647
    # instruction = "Focus on details."
    # 6029, 5676
    # instruction = "Focus on high-level concepts, abstraction, and key ideas."
    # 5618, 5235

    # instruction = "A relevant document would be the most analogous to the query and shares the same key idea as the query."
    # 6618, 5882

    # instruction = "A relevant document would be the most analogous to the query. I don't care about semantic similarity. I don't care about semantic similarity."
    # 6824, 5971

    # instruction = "A relevant document would be the most analogous to the query. " * 2 + "I don't care about semantic similarity. " * 2
    # 6912, 6000 (2491, 3299)
    # (2520,) -> 2755 kp
    # instruction = "A relevant document would be the most analogous to the query. " * 2 + "Look at the keyphrases. " * 2
    # 2712 kp


    # instruction = "A relevant document would be the most analogous to the query. " * 3 + "I don't care about semantic similarity. " * 3
    # 6706, 6000
    # instruction = "A relevant document would be the most analogous to the query. " * 3 + "I don't care about semantic similarity. " * 2
    # 6735, 6118
    # instruction = "A relevant document would be the most analogous to the query. " * 3 + "I don't care about semantic similarity. " * 1
    # 6794, 5941

    # instruction = "A relevant document would be the most analogous to the query. " * 2 + "I don't care about semantic similarity. " * 2 + "Think carefully about these conditions when determining relevance."
    # 6647, 6118

    # instruction = "A relevant document would be the most analogous to the query. " * 2 + "Don't care about semantic similarity. " * 2 
    # 6941, 6029 (6509, 2514, 3299)

    # instruction = "A relevant document would be the most analogous to the query. " * 10 + "Don't care about semantic similarity. " * 10 
    # (2532, 3299)
    # instruction = "relevant would be the most analogous. " * 2 + " Don't care about semantic similarity. " * 2 
    # 6500, 5971
    # instruction = "A relevant document would be the most analogous to the query."
    # 6735, 5941

    # instruction = "A relevant document would be the most analogous to the query. " * 2
    # 6824, 5941

    # instruction = "A relevant document would be the most analogous to the query. " * 10
    # 6706, 5912

    # instruction = "A relevant document would reflect the same relationship described in the query, prioritizing relational over semantic similarity."
    # 5853, 5588
    # instruction = "A relevant document would reflect the same relationship or analogy described in the query. Forget about semantic similarity. Think carefully about these conditions when determining relevance."
    # 6176, 5618
    # instruction = "Retrieve documents that exhibit the same relational analogy as the query. Avoid selecting documents that merely echo the query's keywords without addressing the relational structure. Focus on those that provide analogous reasoning or context."
    # 5853, 5676
    # instruction = "A relevant document demonstrates a relational analogy to the query, focusing on parallels in context, structure, or reasoning rather than direct semantic overlap. Ensure that the documents adhere to these criteria by avoiding those that diverge into tangential or overly literal interpretations. Additionally, exclude passages from [specific field/domain] unless they offer clear analogical insights."
    # 5265, 4824
    
    '''second round'''

    # instruction = "A relevant document captures the query's analogy in structure and meaning, not surface-level overlap."
    # 5912, 5912

    # instruction = "A relevant document should focus solely on providing a clear and accurate answer to the query, without distracting or unnecessary information"
    # # 6059, 5441

    # template mod: query ... question: which passage is the most analogous to the query? instruction: ...
    # instruction = "A relevant document should focus solely on providing a clear and accurate answer to the query, without distracting or unnecessary information"
    # 6147, 5441

    # instruction = "A relevant document should be the most analogous to the query. When in doubt, prioritize documents that are analogically similar to the query."
    # 6618, 5588

    # instruction = "A relevant document should also be the most analogous to the query."
    # 6735, 5559

    input_text_list = [
        f"query: {query.strip()} {instruction.strip()}".strip() for query in questions
        # f"Instruction: {instruction.strip()} Query: {query.strip()}".strip() for query in questions # worse

        # f"query: {query.strip()} question: which passage is the most analogous to the query? {instruction.strip()}".strip() for query in questions

    ] 

    question_embeddings = model.encode(input_text_list, batch_size=bs)
    question_embeddings = question_embeddings.reshape(len(questions), -1)
    if questions_indices is None:
        question_embeddings = np.repeat(question_embeddings, 4, axis=0)
    else:
        questions_indices = np.array(questions_indices)
        questions_indices = np.repeat(questions_indices, 4)

    '''option instructions: '''

    # ====question: A relevant document would be the most analogous to the query.

    # instruction = "A relevant document would be the most analogous to the query." # same as question
    # 6853, 5706
    # instruction = "A relevant query would be the most analogous to the passage." 
    # 6824, 5824
    # instruction = "Instruction: A relevant document would be the most analogous to the query."
    # 6676, 5912
    # instruction = "A relevant document would be the most analogous to the query. A relevant query would be the most analogous to the passage." 
    # 6853, 5647
    # instruction = "A relevant document would be the most analogous to the query. I don't care about semantic similarity. Think carefully about these conditions when determining relevance."
    # 6559, 5618

    # ====question: "A relevant document demonstrates a relational analogy to the query, focusing on parallels in context, structure, or reasoning rather than direct semantic overlap. Ensure that the documents adhere to these criteria by avoiding those that diverge into tangential or overly literal interpretations. Additionally, exclude passages from [specific field/domain] unless they offer clear analogical insights."
    
    # instruction = "A relevant document demonstrates a relational analogy to the query, focusing on parallels in context, structure, or reasoning rather than direct semantic overlap. Ensure that the documents adhere to these criteria by avoiding those that diverge into tangential or overly literal interpretations. Additionally, exclude passages from [specific field/domain] unless they offer clear analogical insights."
    # 5118, 5000

    # ====question:"A relevant document would be the most analogous to the query. I don't care about semantic similarity. I don't care about semantic similarity."

    # instruction = "A relevant document would be the most analogous to the query. I don't care about semantic similarity. I don't care about semantic similarity." # same as question
    # 6824, 5971 (same)

    options = ["passage: " + option for option in options]
    options = [f"{option} {instruction}" for option in options]
    option_embeddings = model.encode(options, batch_size=bs)

    print(question_embeddings.shape, option_embeddings.shape)

    if options_indices is None:
        # similarities = (question_embeddings * option_embeddings).sum(axis=-1)
        
        similarities = []
        cur_bs = 32
        for qidx in tqdm(range(0, len(question_embeddings), cur_bs)):
            sim = (question_embeddings[qidx:qidx+cur_bs] * option_embeddings[qidx:qidx+cur_bs]).sum(axis=-1)
            similarities.append(sim)
        
        similarities = np.hstack(similarities)

    else:
        matrix = question_embeddings @ option_embeddings.T / (np.linalg.norm(question_embeddings, axis=-1)[:, None] * np.linalg.norm(option_embeddings, axis=-1)[None, :])
        similarities = matrix[questions_indices, options_indices]

    similarities = similarities.reshape(-1, 4)

    
    print('sorting')
    predicted_labels = np.argmax(similarities, axis=-1)
    print('sorting finished')
    return predicted_labels, similarities


