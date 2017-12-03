# 训练字符级别的语言模型，使用《神雕侠侣》进行训练，期望让模型编写武侠小说
根据https://github.com/hzy46/Char-RNN-TensorFlow 的代码进行改写，如有侵权，请及时与本人联系。
本人对架构进行了整理，包含三个模块：
数据处理模块read_util.py，模型模块model.py，运行模块run.py

以诗词语料为训练数据，训练之后的生成结果：
Reading model parameters from train_dir/poetry/generate.ckpt-3400
>
风花入山里，风下白云秋。
不是无人事，无时一未知。
山山无处去，春月入山风。
不见青山客，无来独不知。
不怜山上去，相见一人稀。
一去何年事，何来此处时。
一心如此去，相见白云时。
日日春中去，春山夜未多。
何年不为去，相见故人期。
一客不相思，一君何处情。
不应知此别，此处有人归。
不问山风上，无来有不同。
不知何处在，何事不知人。
一客多何处，人人一自归。
山光开白草，秋雨夜中深。
日暮秋花起，山花夜落春。
山风秋上月，风雨落山春。
一客长无去，何年不可知。
山风不不尽，一月自相逢。
日暮春风远，秋风夜月深。
山光生白草，云上水风飞。
何处无人事，何人有此情。
何年不为此，不见不知心。


以《神雕侠侣》为训练数据，过程为：
训练过程
python run.py \
--train_dir train_dir \
--model_name novel \
--batch_size 32 \
--num_layers 2 \
--use_embedding \
--use_sample_loss \
--set_learning_rate 0.01 \
--learning_rate_decay_factor 0.5 \
--input_file data/The_Return_of_the_Condor_Heroes.txt \
--max_train_steps 10000 \
--steps_per_sentence_length 1000 \
--steps_per_checkpoint 100


预测过程
python run.py \
--train_dir /Users/baihai/projects/char_lm/train_dir \
--model_name novel_4 \
--use_embedding \
--sampling \
--sample_length 500
Reading model parameters from /Users/baihai/projects/char_lm/train_dir/novel_4/generate.ckpt-46000
> 
不由得不知，只是他不知是何以，我也有一个人，这一招却要不知。”
杨过道：“不错，我也不知是甚么？”
郭靖心想：“那你不是你，这一句，是我这么不是？”说着伸手向他背上刺去，

心想：“我是他
们不能，我们不知你不会。”小龙女微笑道：“我是我们

了。”郭芙道：“你说我的不会，不能说话。”杨过道：“我是你的一个人，你不知我不知我的，我也不会
的，我不能不肯说？”
杨过见她不答，心想：“我这一个人的一个人，你也不能不是我的。”
杨过听她说话，心念一动。
郭芙道：“那你
我也说得甚么？”
杨过见他一惊，但见郭襄一怔，心中怦怦乱跳，心中暗想，不免不禁不忍，但见郭襄的
手中一股
鲜血淋漓，不由得脸颊中充满满满了一个
一句。他一怔，道：“你不能不肯去，这么不是你？”杨过道：“你说他不肯再去？”
小龙女道：“我这些
人是谁？”郭芙道：“你是我的
一个，我是我是你。”郭芙笑道：“你说你这么一句？”说着伸手抓起，只见他脸上大红，
一个一个
人，心中不动，只道：“我不肯说了，我是你的事，我是不是。”
郭芙心想：“你们这些
人是他不会。”
黄药师道：“你这么好，我不知是甚么？”杨过道：“我是我不是？”杨过道：



个人体会：
（1）对于诗词语料，其语法规律性比较强，字符之间的依赖长度比较短，且依赖的规律性也很强，模型很容易学习到其生成模式；但是对于长篇小说，语法比较随意，依赖长度的变化行比较大，模型学习起来比较困难
（2）对于对准确性比较高的应用，还是应该在模型中加入一些认为的规则进行约束，才有可能得到更可用的结果



问题跟踪与改进
一、问题一：训练时样本的生成方式以及指定的样本长度对模型训练结果的影响（本问题中的实验语料为诗歌语料）
1、在原作者的代码中，使用循环的样本生成方式：在一个batch中，targets是inputs右移一位，同时targets的最后一个time step是inputs的第一个time step。进行以下两个实验：
（1）设置训练数据的max_time为10，经过不同次数的训练后，生成数据的结果为
step: 300/10000...  loss: 261.2494...  0.0650 sec/batch
Restored from: /Users/ocean/projects/Char-RNN-TensorFlow/train_dir/poetry/model-300
>
锡
山山人远，一人一
人。
云人，无水，风风。
不人不去，不山。
人山，日，山人。
不人，<unk>，
。，<unk>风人<unk>，不人，
中不
中。
不人不
远，山山。不山，何日。
<unk>人山，无。山<unk>。
山人一无上，一<unk>。
不，<unk>，不<unk>，
<unk>。
不山<unk>去，不<unk><unk>
中。
一，，不山人，山人。

人人水，不山，
<unk>山人人。
山山一无日，山日一

step: 1000/10000...  loss: 250.5895...  0.0546 sec/batch
Restored from: /Users/ocean/projects/Char-RNN-TensorFlow/train_dir/poetry/model-1000
>
索，一云。
何日三无月，无日。
不来一不日，不人。
云人，云日，不来。
何人，不月，不山。
山来，不去，无风。
不日无人客，不来。
不人何山日，无来。
不来，山处，
山。
山山，不去，山来入人时。
不山，云月，山人出
人。
不日，山处，云风。云云。
何来无云日，无山。
何人，不去，云风。
何山，山水。
风山不风日，不风。
云人无云客，山日。
云山，云水，无人出

step: 2800/10000...  loss: 211.7906...  0.0563 sec/batch
Restored from: /Users/ocean/projects/Char-RNN-TensorFlow/train_dir/poetry/model-2800
>
<unk><unk>，<unk><unk>，山花。
山。
何日，山色，无人。
何。
春山，不见，人人。
何。
何日，春日。
人人何有去，不见。
归。
不知，不是，春山一
归。
江人，何处有无人。
不知春日去，不见。人来君
此路，何日自
人。
此月，无见不知来。
山里，山树入
春。
此日，春月。江云。
水水，不见白人时。
山月，何处。
人。
何处，何处。
春。
不思山山客，山水。

step: 7100/10000...  loss: 195.4819...  0.0566 sec/batch
Restored from: /Users/ocean/projects/Char-RNN-TensorFlow/train_dir/poetry/model-7100
>
巫峡日，风色，寒风。
江色，秋色，寒风。
云。
云中一一客，秋水。
云。
山水秋风雨，风深。
春。
秋来一夜月，秋日。山山。
江上，江月，秋花满
春。
秋来一不在，山水一
风。
不见，秋日，何人在
君。
一心无有别，不见。
何。
一客，相见不可归

(2）设置训练数据的max_time为50，经过不同次数的训练后，生成数据的结果为
Reading model parameters from /Users/baihai/projects/char_lm/train_dir/poetry/generate.ckpt-3400
> 
风花入山里，风下白云秋。
不是无人事，无时一未知。
山山无处去，春月入山风。
不见青山客，无来独不知。
不怜山上去，相见一人稀。
一去何年事，何来此处时。
一心如此去，相见白云时。
日日春中去，春山夜未多。
何年不为去，相见故人期。
一客不相思，一君何处情。
不应知此别，此处有人归。
不问山风上，无来有不同。
不知何处在，何事不知人。
一客多何处，人人一自归。
山光开白草，秋雨夜中深。
日暮秋花起，山花夜落春。
山风秋上月，风雨落山春。
一客长无去，何年不可知。
山风不不尽，一月自相逢。
日暮春风远，秋风夜月深。
山光生白草，云上水风飞。

(3)分析：（1）中由于使用的训练数据长度太短，模型没有学习到长依赖，所以不能生成合理的结果；（2）中使用的训练数据长度比较大，模型有效的学习到了长依赖，所以生成了合理的结果。

2、在对原代码进行修改时，对训练数据的生成方式进行了改造，使得一个batch中，targets完全是由inputs右移一位得到的。仍然进行试验：
（1）设置训练数据的max_time为10，经过不同次数的训练后，生成数据的结果为：
global step 300 learning rate 0.0050 step-time 0.09 perplexity 249.15
Reading model parameters from /Users/baihai/projects/char_lm/train_dir/poetry_2/generate.ckpt-300
> 
风无人不，无人山不，不不山上，不人人山生。
不日不人山，风月一人，风日人不，山风无山时。
一风一不上，何风无无，不不山山，山人不人心。
何人不不人，何山无山，山不人山，风山山无生。
一月不无人，何风无不人。
不月无一上，无风无无时。

lobal step 1600 learning rate 0.0050 step-time 0.08 perplexity 182.27
Reading model parameters from /Users/baihai/projects/char_lm/train_dir/poetry_2/generate.ckpt-1600
> 
知不自不难。
何日何人见，不得一云人。
风中一水日，寒云入山人。
山山无日去，天日有云山。
风山不不见，无时自此时。
风中何日远，何日有云人。
山风不可见，山山不自归。

lobal step 3900 learning rate 0.0025 step-time 0.09 perplexity 135.75
Reading model parameters from /Users/baihai/projects/char_lm/train_dir/poetry_2/generate.ckpt-3900
> 
一子何人别。
一日何人事，无君不得人。
不知何处事，不得一君期。
一日何时去，无人见不稀。
一山一山色，何必是归年。
不问山中去，高生不自归。
山风多一月，秋上入山风。
一地知何日，谁能不得还。
何人不可得，一处在云中。
不问青阳客，何因见不同。


(2）设置训练数据的max_time在[10, 30, 50, 70, 100, 150]中变换，经过不同次数的训练后，生成数据的结果为：
max time: 10
global step 300 learning rate 0.0050 step-time 0.08 perplexity 243.63
Reading model parameters from /Users/baihai/projects/char_lm/train_dir/poetry_2/generate.ckpt-300
> 
人不山，不日不风。
不人人一日，山山人山情。
何时无人在，风人不不中。
不人山无在，何，风一无中。
无山无无去，山人一不中。
不人一一去，山山一不中。
何人一人在，何日不不人。

max time: 30
global step 1600 learning rate 0.0050 step-time 0.24 perplexity 164.62
Reading model parameters from /Users/baihai/projects/char_lm/train_dir/poetry_2/generate.ckpt-1600
> 
有人不可思。
不有何处处，无处自相归。
一君何日去，何处不相知。
不有不相去，何日不相思。
自知一不去，不有白山春。
何日无时去，何时一不同。

max time: 150
global step 2500 learning rate 0.0050 step-time 1.20 perplexity 124.86
Reading model parameters from /Users/baihai/projects/char_lm/train_dir/poetry_2/generate.ckpt-2500
> 
有人何处来。
一心无处在，一去一年年。
日日无时去，春来不有心。
山山不不见，何处不知愁。
此别无人去，何年不得心。
不见山中去，何年有客心。
何处何人去，相看此夜归。
何人不可得，不是不无人。
不见天山远，无人不可知。
此来多不见，一日自无人。

(3)分析：对训练数据的生成方式进行修改以后，发现无论训练数据长度设置为多少，都可以得到理想的结果，说明把训练数据修改成符合语言模型的方式有益于模型的训练

3、结论：训练数据的生成方式和训练数据的样本长度对于模型的训练都有很大的影响，需要仔细考虑。





二、在原来的代码中，计算损失函数时，会对所有的字符进行softmax，由于字符数目比较大，导致训练速度比较慢；之后把损失函数修改为sample softmax，从而提高训练速度。（本问题中的实验预料为诗词语料）（具体的关于candidate sampling的知识，请参考tensorflow的教程）
（1）原来的训练速度：
train_sentence_length : 10
global step 3100 learning rate 0.0025 step-time 0.08 perplexity 140.95
global step 3200 learning rate 0.0025 step-time 0.08 perplexity 142.07

train_sentence_length : 12
global step 3100 learning rate 0.0050 step-time 0.10 perplexity 141.57
global step 3200 learning rate 0.0050 step-time 0.10 perplexity 142.48

train_sentence_length = [10, 30, 50, 70, 100, 150]
train_sentence_length: 30
global step 1100 learning rate 0.0050 step-time 0.23 perplexity 194.54
global step 1200 learning rate 0.0050 step-time 0.24 perplexity 187.96
train_sentence_length: 150
global step 2100 learning rate 0.0050 step-time 1.21 perplexity 140.31
global step 2200 learning rate 0.0050 step-time 1.22 perplexity 128.75
train_sentence_length: 10
global step 300 learning rate 0.0050 step-time 0.08 perplexity 243.63

（2）修改为sample softmax之后的训练速度：
train_sentence_length: 10
global step 1100 learning rate 0.0050 step-time 0.04 perplexity 68.75
global step 1200 learning rate 0.0025 step-time 0.03 perplexity 67.67

train_sentence_length: 70
global step 2100 learning rate 0.0012 step-time 0.20 perplexity 58.20
global step 2200 learning rate 0.0012 step-time 0.20 perplexity 58.84

train_sentence_length: 150
global step 3100 learning rate 0.0006 step-time 0.42 perplexity 55.02
global step 3200 learning rate 0.0006 step-time 0.43 perplexity 56.06
global step 3300 learning rate 0.0003 step-time 0.42 perplexity 54.90

（3）分析：由（1）和（2）对比可知，修改为sample softmax之后，训练时间变成了原来的三分之一，从而大幅度提高训练速度，同时最终的生成结果并没有变差。





三、源代码中的实现方式为在整个训练过程中只能使用固定学习率和固定的样本长度，会影响模型的训练速度和灵活性。之后修改为在训练过程中自适应地修改学习率，并且使用随机可变的样本长度。
