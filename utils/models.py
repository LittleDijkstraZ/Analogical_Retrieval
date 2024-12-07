from tqdm.auto import tqdm

from transformers import AutoTokenizer, pipeline, BitsAndBytesConfig
import torch
from accelerate.utils import BnbQuantizationConfig
from transformers import AutoModelForCausalLM

def get_model_general(model_name):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="fp4",
    )
    # quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="left",
        trust_remote_code=True
    )

    # pipe = pipeline(
    #     "text-generation",
    #     model=model_name,
    #     tokenizer=tokenizer,
    #     trust_remote_code=True,
    #     device_map="cuda:0",
    #     quantization_config=quantization_config,  # Pass the quantization config
    #     load_in_4bit=True,                     # Enable 4-bit quantization

    # )
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config, device_map="cuda:0")
    model.eval()
    # tokenizer = pipe.tokenizer
    # model = pipe.model
    return model, tokenizer



import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel, PeftConfig
import numpy as np

class Promptriever:
    def __init__(self, model_name_or_path):
        self.model, self.tokenizer = self.get_model(model_name_or_path)
        self.model.eval().cuda()
        self.instruction_list = [
            "",
            "Merry Christmas.",

            "A relevant document would be the most analogous to the query. I don't care about semantic similarity.",
            "A relevant document would be the most analogous to the query. I don't care about semantic similarity. " * 2,
            "A relevant document would be the most analogous to the query. " * 2 + "I don't care about semantic similarity. " * 2,
            "A relevant document demonstrates a relational analogy to the query, focusing on parallels in context, structure, or reasoning rather than direct semantic overlap. Ensure that the documents adhere to these criteria by avoiding those that diverge into tangential or overly literal interpretations. Additionally, exclude passages from [specific field/domain] unless they offer clear analogical insights.",
            "Focus on high-level concepts, abstraction, and key ideas.",


            "A relevant document would be the most semantically similary to the query. I don't care about analogical similarity.",
            "A relevant document would be the least semantically similary to the query. I care about analogical similarity.",
            "A relevant document would be the least analogous to the query.",
        ]

    def get_model(self, peft_model_name):
        # Load the PEFT configuration to get the base model name
        peft_config = PeftConfig.from_pretrained(peft_model_name)
        base_model_name = peft_config.base_model_name_or_path

        # Load the base model and tokenizer
        base_model = AutoModel.from_pretrained(base_model_name,
                                               device_map="auto",
                                              torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "right"

        # Load and merge the PEFT model
        model = PeftModel.from_pretrained(base_model, peft_model_name)
        model = model.merge_and_unload()

        # can be much longer, but for the example 512 is enough
        model.config.max_length = 512
        tokenizer.model_max_length = 512

        return model, tokenizer

    def create_batch_dict(self, tokenizer, input_texts):
        max_length = self.model.config.max_length
        batch_dict = tokenizer(
            input_texts,
            max_length=max_length - 1,
            return_token_type_ids=False,
            return_attention_mask=False,
            padding=False,
            truncation=True,
        )
        batch_dict["input_ids"] = [
            input_ids + [tokenizer.eos_token_id]
            for input_ids in batch_dict["input_ids"]
        ]
        return tokenizer.pad(
            batch_dict,
            padding=True,
            pad_to_multiple_of=8,
            return_attention_mask=True,
            return_tensors="pt",
        )

    def encode(self, sentences, max_length: int = 2048, batch_size: int = 4):
        all_embeddings = []
        for i in tqdm(range(0, len(sentences), batch_size)):
            batch_texts = sentences[i : i + batch_size]

            batch_dict = self.create_batch_dict(self.tokenizer, batch_texts)
            batch_dict = {
                key: value.to(self.model.device) for key, value in batch_dict.items()
            }

            with torch.amp.autocast(device_type='cuda'):
                with torch.no_grad():
                    outputs = self.model(**batch_dict)
                    last_hidden_state = outputs.last_hidden_state
                    sequence_lengths = batch_dict["attention_mask"].sum(dim=1) - 1
                    batch_size = last_hidden_state.shape[0]
                    reps = last_hidden_state[
                        torch.arange(batch_size, device=last_hidden_state.device),
                        sequence_lengths,
                    ]
                    embeddings = F.normalize(reps, p=2, dim=-1)
                    all_embeddings.append(embeddings.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)

