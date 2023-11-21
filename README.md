# llm_surprisal_sample
please use the following command:  
```
python main_sample.py <dataset_name>.sentitems <model_name> <sampling_method> > <dataset_name>.<model_name>.<sampling_value>.surprisal
```
An example would be: python main_sample.py naturalstories.sentitems gpt2 top-k > naturalstories.gpt2.k100.surprisal

dataset_name:  
```
naturalstories.sentitems, dundee.sentitems
```

model_name: 
```
GPT-2 family:
"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"

GPT-Neo family:
"EleutherAI/gpt-neo-125M", "EleutherAI/gpt-neo-1.3B", "EleutherAI/gpt-neo-2.7B",
"EleutherAI/gpt-j-6B", "EleutherAI/gpt-neox-20b"

OPT family:
"facebook/opt-125m", "facebook/opt-350m", "facebook/opt-1.3b", "facebook/opt-2.7b",
"facebook/opt-6.7b", "facebook/opt-13b", "facebook/opt-30b", "facebook/opt-66b"
```

sampling_method:
```
top-k: we can specify the value of k or use the default choices
top-p: we can specify the value of p or use the default choices
temperature: we can specify the value of temperature or use the default choices
none: no sampling
```

The output would be a space-delimited two-column file containing the word and LM surprisal.
```
word llm_surp
If 5.964238166809082
you 0.5626226663589478
were 5.056972503662109
to 1.9595699310302734
journey 33.21928024291992
to 1.8419939279556274
the 1.2147647142410278
North 5.670198917388916
of 7.474577903747559
England, 3.5752294063568115
```
