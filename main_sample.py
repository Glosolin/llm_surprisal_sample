"""
GPT-2 family:
"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"

GPT-Neo family:
"EleutherAI/gpt-neo-125M", "EleutherAI/gpt-neo-1.3B", "EleutherAI/gpt-neo-2.7B",
"EleutherAI/gpt-j-6B", "EleutherAI/gpt-neox-20b"

OPT family:
"facebook/opt-125m", "facebook/opt-350m", "facebook/opt-1.3b", "facebook/opt-2.7b",
"facebook/opt-6.7b", "facebook/opt-13b", "facebook/opt-30b", "facebook/opt-66b"
"""

### need to implement temperature w/ top-k and temperature w/ top-k

import os, sys, torch, transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, GPTNeoXTokenizerFast
import pandas as pd


def generate_stories(fn):
    stories = []
    f = open(fn)
    first_line = f.readline()
    assert first_line.strip() == "!ARTICLE"
    curr_story = ""

    for line in f:
        sentence = line.strip()
        if sentence == "!ARTICLE":
            stories.append(curr_story[:-1])
            curr_story = ""
        else:
            curr_story += line.strip() + " "

    stories.append(curr_story[:-1])
    return stories


def main():
    stories = generate_stories(sys.argv[1])
    model_variant = sys.argv[2].split("/")[-1]

    if "gpt-neox" in model_variant:
        tokenizer = GPTNeoXTokenizerFast.from_pretrained(sys.argv[2])
    elif "gpt" in model_variant:
        tokenizer = AutoTokenizer.from_pretrained(sys.argv[2], use_fast=False)
    elif "opt" in model_variant:
        tokenizer = AutoTokenizer.from_pretrained(sys.argv[2], use_fast=False)
    else:
        raise ValueError("Unsupported LLM variant")

    model = AutoModelForCausalLM.from_pretrained(sys.argv[2])
    model.eval()
    softmax = torch.nn.Softmax(dim=-1)
    ctx_size = model.config.max_position_embeddings
    bos_id = model.config.bos_token_id

    results_list = []
    sampling_type = sys.argv[3]
    if sampling_type == 'top-k':
        sampling_values = [100] # [10, 50, 100, 1000, 2000, 10000, 25000, 50000]
        # print('Top-k Sampling: ')
    elif sampling_type == 'top-p':
        sampling_values = [0.95, 0.9, 0.85, 0.8, 0.7]
        # print('Top-p Sampling: ')
    elif sampling_type == 'temperature':
        sampling_values = [0.1, 0.3, 0.5, 1.0, 1.5, 10]
    elif sampling_type == 'none':
        sampling_values = [0]
        # print('No Sampling: ')
    else:
        raise ValueError("Unsupported Sampling Method")
    
    for top_value in sampling_values:
        batches = []
        words = []
        for story in stories:
            words.extend(story.split(" "))
            tokenizer_output = tokenizer(story)
            ids = tokenizer_output.input_ids
            attn = tokenizer_output.attention_mask
            start_idx = 0

            # sliding windows with 50% overlap
            # start_idx is for correctly indexing the "later 50%" of sliding windows
            while len(ids) > ctx_size:
                # for GPT-NeoX (bos_id not appended by default)
                if "gpt-neox" in model_variant:
                    batches.append((transformers.BatchEncoding({"input_ids": torch.tensor([bos_id] + ids[:ctx_size-1]).unsqueeze(0),
                                                            "attention_mask": torch.tensor([1] + attn[:ctx_size-1]).unsqueeze(0)}),
                                    start_idx))
                # for GPT-2/GPT-Neo (bos_id not appended by default)
                elif "gpt" in model_variant:
                    batches.append((transformers.BatchEncoding({"input_ids": torch.tensor([bos_id] + ids[:ctx_size-1]),
                                                                "attention_mask": torch.tensor([1] + attn[:ctx_size-1])}),
                                    start_idx))
                # for OPT (bos_id appended by default)
                else:
                    batches.append((transformers.BatchEncoding({"input_ids": torch.tensor(ids[:ctx_size]).unsqueeze(0),
                                                            "attention_mask": torch.tensor(attn[:ctx_size]).unsqueeze(0)}),
                                    start_idx))

                ids = ids[int(ctx_size/2):]
                attn = attn[int(ctx_size/2):]
                start_idx = int(ctx_size/2)-1

            # remaining tokens
            if "gpt-neox" in model_variant:
                batches.append((transformers.BatchEncoding({"input_ids": torch.tensor([bos_id] + ids).unsqueeze(0),
                                                        "attention_mask": torch.tensor([1] + attn).unsqueeze(0)}),
                            start_idx))
            elif "gpt" in model_variant:
                batches.append((transformers.BatchEncoding({"input_ids": torch.tensor([bos_id] + ids),
                                                        "attention_mask": torch.tensor([1] + attn)}),
                            start_idx))
            else:
                batches.append((transformers.BatchEncoding({"input_ids": torch.tensor(ids).unsqueeze(0),
                                                            "attention_mask": torch.tensor(attn).unsqueeze(0)}),
                                start_idx))

        curr_word_ix = 0
        curr_word_surp = []
        curr_toks = ""

        print("word llm_surp")
        for batch in batches:
            batch_input, start_idx = batch
            output_ids = batch_input.input_ids.squeeze(0)[1:]

            with torch.no_grad():
                model_output = model(**batch_input)
            toks = tokenizer.convert_ids_to_tokens(batch_input.input_ids.squeeze(0))[1:]

            if sampling_type == 'none':    # original    
                index = torch.arange(0, output_ids.shape[0])
                surp = -1 * torch.log2(softmax(model_output.logits).squeeze(0)[index, output_ids]) 

            elif sampling_type == 'top-k':   # top-k 
                logits = model_output.logits
                top_k_logits, top_k_indices = torch.topk(logits, k=top_value, dim=-1)
                top_k_probs = softmax(top_k_logits)
                # new_probs = torch.zeros_like(logits)  # inifity issue
                new_probs = torch.full_like(logits, fill_value=1e-10)
                new_probs.scatter_(-1, top_k_indices, top_k_probs)
                
                index = torch.arange(0, output_ids.shape[0])
                actual_probs = new_probs[index, output_ids]
                surp = -1 * torch.log2(actual_probs)

            elif sampling_type == 'top-p':   # top-p
                logits = model_output.logits
                probs = softmax(logits)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cum_probs = torch.cumsum(sorted_probs, dim=-1)
                new_probs = torch.full_like(logits, fill_value=1e-10)
                for i in range(cum_probs.shape[0]):
                    num_inc = sum(cum_probs[i]<=top_value).item()
                    num_inc = max(1, num_inc)
                    inc_probs = sorted_probs[i, :num_inc]
                    inc_probs /=inc_probs.sum()
                    new_probs[i, sorted_indices[i, :num_inc]] = inc_probs 

                index = torch.arange(0, output_ids.shape[0])
                actual_probs = new_probs[index, output_ids]
                surp = -1 * torch.log2(actual_probs)

            elif sampling_type == 'temperature':   # temperature
                logits = model_output.logits
                scaled_logits = logits / top_value
                probs = softmax(scaled_logits)
                
                index = torch.arange(0, output_ids.shape[0])
                actual_probs = probs[index, output_ids]
                surp = -1 * torch.log2(actual_probs)

            for i in range(start_idx, len(toks)):
                # necessary for diacritics in Dundee
                cleaned_tok = toks[i].replace("Ġ", "", 1).encode("latin-1").decode("utf-8")
                # for token-level surprisal
                # print(cleaned_tok, surp[i].item())
                # for word-level surprisal
                curr_word_surp.append(surp[i].item())
                curr_toks += cleaned_tok
                # summing subword token surprisal ("rolling")
                words[curr_word_ix] = words[curr_word_ix].replace(cleaned_tok, "", 1)
                if words[curr_word_ix] == "":
                    # print(curr_toks, sum(curr_word_surp))
                    results_dict = {'token': curr_toks, f'{top_value}': sum(curr_word_surp)}
                    results_list.append(results_dict)
                    curr_word_surp = []
                    curr_toks = ""
                    curr_word_ix += 1
    # print out results
    df = pd.DataFrame(results_list)
    for col in df.columns:
        if col != 'token':
            df[col] = df[col].dropna().reset_index(drop=True)
    df = df.apply(lambda x: pd.Series(x.dropna().values))
    df = df.dropna()
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    # print(df)
    for index, row in df.iterrows():
        print(' '.join(str(value) for value in row))

if __name__ == "__main__":
    main()
