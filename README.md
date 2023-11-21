# llm_surprisal_sample
please use the following command:
python main_sample.py <dataset_name>.sentitems <model_name> <sampling_method> > <dataset_name>.<model_name>.<sampling_value>.surprisal

An example would be: python main_sample.py naturalstories.sentitems gpt2 top-k> naturalstories.gpt2.k100.surprisal
