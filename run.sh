python run.py \
--train_dir /Users/baihai/projects/char_lm/train_dir \
--model_name novel_2 \
--batch_size 32 \
--max_time 32 \
--use_embedding \
--num_layers 3 \
--learning_rate 0.001 \
--input_file /Users/baihai/projects/char_lm/data/The_Return_of_the_Condor_Heroes.txt \
--max_train_steps 1000 \
--steps_per_checkpoint 100 \
--learning_rate_decay_factor 0.5 \
--immediate_learning_rate_decay


arr shape: (32, 29665)
batch size: 32
max time: 32
batch cnt: 927
global step 1500 learning rate 0.0020 step-time 0.00 perplexity 135.73
global step 1600 learning rate 0.0020 step-time 0.00 perplexity 130.99
global step 1700 learning rate 0.0020 step-time 0.00 perplexity 123.89
global step 1800 learning rate 0.0020 step-time 0.00 perplexity 119.67
global step 1900 learning rate 0.0020 step-time 0.00 perplexity 116.28
global step 2000 learning rate 0.0020 step-time 0.00 perplexity 115.24
global step 2100 learning rate 0.0020 step-time 0.00 perplexity 110.63


python run.py \
--train_dir /Users/baihai/projects/char_lm/train_dir \
--model_name novel_2 \
--use_embedding \
--sampling \
--sample_length 500

global step 1600 learning rate 0.0020 step-time 0.00 perplexity 130.99
Reading model parameters from /Users/baihai/projects/char_lm/train_dir/novel_2/generate.ckpt-1600
> 杨过锋沌栉奚辽儡君佻箩辽捱糠佻骄箩骄牟匕儡笈笈佻奚笈佻褥牟奚牟蒂牟匕辽箩辽笈匕辽餐佻箩妒牟钿箩辽笈匕栉儡牟箩色箩
寿笈匕儡笈佻牟诈佻匕辽捱骄牟儡笈箩匕儡匕餐牟匕儡儡匕箩匕牟笈牟匕骄儡儡辽儡笈倡笈奚牟牟匕辽倡倡倡倡佻匕序笈匕儡笈诈
佻诈箩辽牟妒佻奚牟蒂笈奚箩钿儡色褥奚捱诈褥佻笈奚匕儡笈奚牟箩辽箩辽儡匕箩去笈匕价儡牟儡骄牟匕箩违牟匕儡箩辽箩去笈奚
捱诈笈匕篮儡匕牟钿餐骄儡牟匕儡牟匕牟儡君笈佻褥牟匕箩钿儡辽儡辽倡奚箩儡牟奚捱匕辽餐佻牟匕骄箩匕辽儡牟匕辽笈倡佻匕辽
箩辽箩去捱妒褥匕儡箩辽箩钿牟蒂笈妒牟蒂笈匕辽倡匕儡箩匕餐匕辽箩色牟妒笈妒捱佻褥箩骄箩辽箩色笈蒂捱匕价儡色牟蒂捱奚笈
匕匕牟儡辽蒂箩钿牟邂儡骄餐笈箩匕儡辽牟奚牟匕牟牟匕君牟蒂诈箩序箩辽儡辽捱匕牟倡佻诈箩钿箩辽箩色褥奚捱诈褥奚君牟匕儡
箩去牟蒂笈诈佻笈倡妒笈笈箩色牟钿箩钿箩过箩辽儡色褥爪牟妒笈诈牟蒂捱匕辽儡奚箩钿牟邂儡骄儡儡餐奚佻箩辽笈妒捱奚箩匕餐
牟箩牟箩匕笈奚捱诈笈笈笈诈笈箩辽儡辽捱骄笈佻箩骄牟妒箩寿褥辽儡儡辽钿箩去箩寿笈妒笈妒佻诈牟妒笈笈箩去褥奚佻笈佻佻褥
诈佻笈箩寿箩匕箩色牟钿箩辽儡钿箩辽儡钿箩辽箩去箩啦笈妒笈佻佻笈箩匕辽儡笈匕匕餐笈诈诈佻牟儡匕辽牟

global step 2300 learning rate 0.0020 step-time 0.00 perplexity 104.07
Reading model parameters from /Users/baihai/projects/char_lm/train_dir/novel_2/generate.ckpt-2300
> 杨过
杨过踹心笈钿儡好牟辽儡我哦喻箩过怂诈牟诈牟你儡心儡好牟诈侦偶屎牟你殳偶瑙儡好牟奚箩上牟谁儡孙波心箩手牟过牟你磊你殳
的殳尘箩不鉴吵牟妒倡镰儡妒箩去箩我儡心牟过怂”倡佻箩你婉去牟妒婉，已兼芙郝，始却「佐锋芙”婉，始的衾上一去“无”婉，已
兼‘殳见翱违媳手褥牟牟你郝。：花喙柳唉侦万”倡佻牟我儡我倡喻倡糠”(辽郝的衾色郝。匝等站界箩去婉子牟过牟我郝王稂去潼去
倡佻箩你阂牟牟心”桌儡箩上箩了倡挫箩去婉，听痪「钉怂妒怂婉，已，始无”恍了侦糠牟心捱及侦糠”桌嚅”洒牟辽箩上倡喻侦喻箩
去(牟』(餐”婉睛”倡奚”(价婉了亩孙稂去倡痪”(价箩上潼去(牟妒婉色箩得倡奚箩得”倡喻”侦佻萌，已「佐怂芙锋”婉住”婉去”桌
嚅”（，始无胥大不好箩辽”“痪箩手婉去(辽倡佻箩去(辽牟”「，始，抹骱伸界耶，无”“痪倡痪侦偶去”桌牟儡我倡奚捱不联餐”佻
黜人牟奚捱我牟我殷得潼去”（去婉了葱你衾上郝人佐倡奠翱你婉来(辽髻挫箩”婉子”（王衾过潼上牟妒侦佻牟诈”桌嚅佻痪”桌辽
牟上箩你婉你婉，已，低伟”婉来郝了倡8”婉去捞奚辽儡妒匕杀惆我婉来(妒佻牟辽儡牟捱喻牟…(栉你“匠「叉羽是「移账甄。恍歹
婉色”稂去婉你（，却但踉去「与佐编你衾过”稂。恍歹但

global step 3400 learning rate 0.0020 step-time 0.00 perplexity 83.49
Reading model parameters from /Users/baihai/projects/char_lm/train_dir/novel_2/generate.ckpt-3400
> 杨过
杨过锋锋倡心，微术甫见婉你嘈上」崔，们柳衾一手箩手”「巴佐”婉不不，：哟踹你一颊牟手餐你锋芬你巴巴(妒”瞥抑不，都们却
们“匠霭上镂不你晋”瞥践，无”婉，：中镂一侮龟燕，听唠“无一你儡心”瞥诈惆手倍心倍，们佐们佐”佐“巴「巴霭辽”双辽，：夫洽
心辽，皱手「愁霭辽，冷漓怂，冷虑唔惆”箩我嘈臂箩上惆手儡手倡”瞥动餐喻(髻你儡手餐倡儡我燕你倡鉴餐倡”洒妈”瞥诈倡儡，：
夫拟冷颊牟手殳你凝见蔓你倡，冷馁褥惆小际却，都，：刃佐蓉霭手镂孙。佐亟仍，皱嫁锲心”尔你儡手”双辽人”瞥际一手倍，都，
们“糊臂镂手儡不，们“喁蓉鉴不我巴儡你儡你波手箩手餐鉴”瞥践儡不见衾手餐倡倡儡你燕，听咧惆手廿去牟我殳，：无辙倡莫，听
鉴蓉一颊填见倡心倍手”双辽”儡我儡心霎上(孕餐翌我儡你倡，都但佐，凄虑汨孙，埋‘”绿，砍咦(不你婉手」”桌康，：哟殳，们镂
芙，凄，但们衾喂”「焚手”(手崔去倡笑衾手倍你椽黜”侦兼芙巴殳，听鉴‘，凄，冷虑「愁「巴镂上」衾上”(妒儡心倍…瞥棍牟不主倍
，：无辙(嗯翌孙”双磨(不你鼎际倡心，：无矣翌你心磨瞥手牟手儡手餐倡(儡，听唠‘“锋霭上甄。佐“糊底霭臂，都但：”修，微虑一
颊箩手踹我衾心箩不我”瞥出”(我婉手箩手




















































































