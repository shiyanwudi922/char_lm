python run.py \
--train_dir /Users/baihai/projects/char_lm/train_dir \
--model_name novel_1 \
--batch_size 32 \
--max_time 100 \
--use_embedding \
--num_layers 3 \
--learning_rate 0.001 \
--input_file /Users/baihai/projects/char_lm/data/The_Return_of_the_Condor_Heroes.txt \
--max_train_steps 1000 \
--steps_per_checkpoint 100 \
--learning_rate_decay_factor 0.8 \
--immediate_learning_rate_decay


origin vocab size of the text is: 3946
actual vocabulary size is: 3947
Creating 3 layers of 128 units.
Reading model parameters from /Users/baihai/projects/char_lm/train_dir/novel_1/generate.ckpt-3000
arr shape: (32, 29601)
batch size: 32
max time: 100
batch cnt: 296
global step 3100 learning rate 0.0010 step-time 0.01 perplexity 142.57
global step 3200 learning rate 0.0010 step-time 0.01 perplexity 147.80
reset initial state
global step 3300 learning rate 0.0010 step-time 0.01 perplexity 146.76
global step 3400 learning rate 0.0010 step-time 0.01 perplexity 138.40


python run.py \
--train_dir /Users/baihai/projects/char_lm/train_dir \
--model_name novel_1 \
--use_embedding \
--sampling \
--sample_length 500