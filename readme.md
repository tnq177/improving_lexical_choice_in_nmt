![Good friend lol](./images/doraemon_nobita.gif)  

This is the code for the paper **Improving Lexical Choice in NMT**. The branches are:  

* master: baseline NMT
* tied_embedding: baseline NMT with tied embedding
* fixnorm: fixnorm model in paper
* fixnorm_lex: fixnorm+lex model in paper
* arthur: apply the method of [Arthur et al.](https://arxiv.org/abs/1606.02006) on top of baseline NMT

This branch is meant to reproduce the work of Arthur et al. above (see our paper for more information since we only reproduce one model of their proposed approaches). All train/translate commands are the same, just add one more flag ```--lexical-file``` which points to the lexical table which is a numpy array of rows representing the source type ids, and columns represent target type ids (so it's a [src_vocab_size, target_vocab_size] table). We follow the [tutorial here](http://masatohagiwara.net/using-giza-to-obtain-word-alignment-between-bilingual-sentences.html) to learn a lexical table with Giza++ (with reverse order to get t(e|f), of course). Then use ```gen_lexical_table.py``` to generate the lexical table which we pass to ```--lexical-table``` flag. 

To train a model:
* write a configuration function in ```configurations.py```
* run: ```python -m nmt --proto your_config_func```  

Depending on your config function, the code generates a direction under ```nmt/saved_models/your_model_name``` and saves all dev validations there, as well as dev perplexities, train perplexities, best model checkpoint, checkpoint so far (I've tested with saving 1 best checkpoint, not sure about > 1). You should use this checkpoint to translate on any other input.

To translate with UNK replacement:
* run: ```python -m nmt --proto your_config_func --mode translate --unk-repl --model-file path_your_saved_checkpoint.cpkt --input-file path_to_input_file```  

Remember the checkpoint includes data file, meta file, ... but just link to ```.cpkt```, ignore the extension. 
