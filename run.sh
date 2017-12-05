python run.py \
--train_dir /Users/baihai/projects/char_lm/train_dir \
--model_name novel \
--batch_size 32 \
--num_layers 3 \
--use_embedding \
--use_sample_loss \
--learning_rate 0.01 \
--learning_rate_decay_factor 0.5 \
--input_file /Users/baihai/projects/char_lm/data/The_Return_of_the_Condor_Heroes.txt \
--max_train_steps 10000 \
--steps_per_sentence_length 1000 \
--steps_per_checkpoint 500 \
--steps_per_log 100


python run.py \
--train_dir /Users/baihai/projects/char_lm/train_dir \
--model_name novel \
--use_embedding \
--sampling \
--sample_length 500


















































































