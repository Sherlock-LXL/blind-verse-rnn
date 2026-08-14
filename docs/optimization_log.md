Run 0: 未使用外部语料库预训练

train: 1292
val: 159
vocab size: 1555

Epoch 01 | train loss: 6.9084 | val loss: 6.1071
  -> saved best model
  sample: 一有一
Epoch 02 | train loss: 5.9623 | val loss: 5.9511
  -> saved best model
Epoch 03 | train loss: 5.7884 | val loss: 5.9099
  -> saved best model
Epoch 04 | train loss: 5.6888 | val loss: 5.8779
  -> saved best model
Epoch 05 | train loss: 5.6034 | val loss: 5.8518
  -> saved best model
  sample: 我一一过的天的的
Epoch 06 | train loss: 5.5055 | val loss: 5.8160
  -> saved best model
Epoch 07 | train loss: 5.3981 | val loss: 5.7750
  -> saved best model
Epoch 08 | train loss: 5.2885 | val loss: 5.7284
  -> saved best model
Epoch 09 | train loss: 5.1826 | val loss: 5.6966
  -> saved best model
Epoch 10 | train loss: 5.0690 | val loss: 5.6607
  -> saved best model
  sample: 也有人一点着
Epoch 11 | train loss: 4.9609 | val loss: 5.6117
  -> saved best model
Epoch 12 | train loss: 4.8497 | val loss: 5.5894
  -> saved best model
Epoch 13 | train loss: 4.7392 | val loss: 5.5531
  -> saved best model
Epoch 14 | train loss: 4.6314 | val loss: 5.5206
  -> saved best model
Epoch 15 | train loss: 4.5291 | val loss: 5.4944
  -> saved best model
  sample: 我们有人着新的光过
Epoch 16 | train loss: 4.4361 | val loss: 5.4762
  -> saved best model
Epoch 17 | train loss: 4.3347 | val loss: 5.4721
  -> saved best model
Epoch 18 | train loss: 4.2343 | val loss: 5.4558
  -> saved best model
Epoch 19 | train loss: 4.1412 | val loss: 5.4390
  -> saved best model
Epoch 20 | train loss: 4.0648 | val loss: 5.4205
  -> saved best model
  sample: 我们有人匆经过
Epoch 21 | train loss: 3.9881 | val loss: 5.4158
  -> saved best model
Epoch 22 | train loss: 3.8930 | val loss: 5.4085
  -> saved best model
Epoch 23 | train loss: 3.8103 | val loss: 5.4332
Epoch 24 | train loss: 3.7407 | val loss: 5.3966
  -> saved best model
Epoch 25 | train loss: 3.6621 | val loss: 5.4228
  sample: 我想得清澈
Epoch 26 | train loss: 3.5997 | val loss: 5.3998
Epoch 27 | train loss: 3.5275 | val loss: 5.4112
Epoch 28 | train loss: 3.4548 | val loss: 5.4096
Epoch 29 | train loss: 3.3906 | val loss: 5.4158
Early stopping

===== Blind Verse =====

1
A: 有人匆匆匆匆匆
B: 我把一点一点

2
A: 我曾经过很久的光
B: 我们把后一点起一场

3
A: 我曾经过去经过去
B: 我曾经于一场过

4
A: 有些时候不再满路路
B: 我把一场的时候

5
A: 一场无数一点一点
B: 有些事情以为我们慢慢慢慢慢



Run 1: 扩大预训练库，并用大模型训练（E1024 / H1024 / L1，dropout 0.2，Adam，lr 4e-4，wd 1e-4，ReduceLROnPlateau）

train: 49932
val: 2628
vocab size: 5160

Epoch 01 | train loss: 5.7763 | val loss: 5.3878
  -> saved best model
  sample: 那些白云在那里的花
Epoch 02 | train loss: 5.2301 | val loss: 5.2424
  -> saved best model
Epoch 03 | train loss: 5.0766 | val loss: 5.1987
  -> saved best model
Epoch 04 | train loss: 5.0104 | val loss: 5.1890
  -> saved best model
Epoch 05 | train loss: 4.9797 | val loss: 5.1721
  -> saved best model
  sample: 有些不要让我们底的
Epoch 06 | train loss: 4.9502 | val loss: 5.1709
  -> saved best model
Epoch 07 | train loss: 4.9234 | val loss: 5.1654
  -> saved best model
Epoch 08 | train loss: 4.8947 | val loss: 5.1720
Epoch 09 | train loss: 4.8706 | val loss: 5.1624
  -> saved best model
Epoch 10 | train loss: 4.8500 | val loss: 5.1519
  -> saved best model
  sample: 我们的小路
Epoch 11 | train loss: 4.8317 | val loss: 5.1559
Epoch 12 | train loss: 4.8160 | val loss: 5.1631
Epoch 13 | train loss: 4.6859 | val loss: 5.1320
  -> saved best model
Epoch 14 | train loss: 4.6658 | val loss: 5.1369
Epoch 15 | train loss: 4.6606 | val loss: 5.1517
  sample: 一种不动的
Epoch 16 | train loss: 4.5831 | val loss: 5.1426
Epoch 17 | train loss: 4.5737 | val loss: 5.1510
Epoch 18 | train loss: 4.5309 | val loss: 5.1484
Early stopping

===== Pretrain Samples =====

1
A: 不见一个人
B: 不是因为这是

2
A: 他也不会再见你
B: 我也不能看见这里

3
A: 一点被一起
B: 他在一个中午的

4
A: 他不能是为它的
B: 那是我不能是你的

5
A: 那是我的爱情
B: 有时刻在我的眼睛里

Benchmark loss: 5.1320



Run 2: 在 Run 1 基础上，dropout 0.2 → 0.1

train: 49932
val: 2628
vocab size: 5160

Epoch 01 | train loss: 5.7255 | val loss: 5.3441 | lr: 4.00e-04
  sample: 不在这是我说的那个老
  -> saved best model
Epoch 02 | train loss: 5.1480 | val loss: 5.2128 | lr: 4.00e-04
  sample: 我们是否
  -> saved best model
Epoch 03 | train loss: 4.9829 | val loss: 5.1716 | lr: 4.00e-04
  sample: 一只小羊虫在地上
  -> saved best model
Epoch 04 | train loss: 4.9193 | val loss: 5.1596 | lr: 4.00e-04
  sample: 不要说话的人
  -> saved best model
Epoch 05 | train loss: 4.8889 | val loss: 5.1599 | lr: 4.00e-04
  sample: 这是我的身份
Epoch 06 | train loss: 4.8669 | val loss: 5.1649 | lr: 2.00e-04
  sample: 我要说话他是一个男人我一个人去
Epoch 07 | train loss: 4.7280 | val loss: 5.1263 | lr: 2.00e-04
  sample: 在这样的时间
  -> saved best model
Epoch 08 | train loss: 4.7045 | val loss: 5.1342 | lr: 2.00e-04
  sample: 如此之外的
Epoch 09 | train loss: 4.6967 | val loss: 5.1461 | lr: 1.00e-04
  sample: 如果能够在一个中国度
Epoch 10 | train loss: 4.6141 | val loss: 5.1370 | lr: 1.00e-04
  sample: 在一条河间中间
Epoch 11 | train loss: 4.6071 | val loss: 5.1454 | lr: 5.00e-05
  sample: 那是在那里的人间
Epoch 12 | train loss: 4.5590 | val loss: 5.1497 | lr: 5.00e-05
  sample: 我有一个人
Early stopping

===== Pretrain Samples =====

1
A: 你的脚印就像地铁
B: 它们一起的形状

2
A: 但那些在人间的呼叫
B: 他是一个老虎

3
A: 一次在我的身体里
B: 一只小鸟飞过去的

4
A: 在我的胸里
B: 我的眼睛没有一样

5
A: 如今我们在天气
B: 有时期望我们的一朵花开始

Benchmark loss: 5.1263



Run 3: 在 Run 2 基础上，Adam → AdamW

train: 49932
val: 2628
vocab size: 5160

Epoch 01 | train loss: 5.6402 | val loss: 5.2059 | lr: 4.00e-04
  sample: 我也能不能够看见了我的
  -> saved best model
Epoch 02 | train loss: 4.8446 | val loss: 5.0143 | lr: 4.00e-04
  sample: 不可能有人知道我们的家具
  -> saved best model
Epoch 03 | train loss: 4.4120 | val loss: 4.9730 | lr: 4.00e-04
  sample: 一只鸟在草丛中
  -> saved best model
Epoch 04 | train loss: 4.0470 | val loss: 4.9864 | lr: 4.00e-04
  sample: 这些人不会在一起
Epoch 05 | train loss: 3.7209 | val loss: 5.0421 | lr: 2.00e-04
  sample: 一朵荷花在水中央的
Epoch 06 | train loss: 3.2905 | val loss: 5.0897 | lr: 2.00e-04
  sample: 我们一只小小的木匠了
Epoch 07 | train loss: 3.0996 | val loss: 5.1464 | lr: 1.00e-04
  sample: 只有一个人来回去
Epoch 08 | train loss: 2.8712 | val loss: 5.1834 | lr: 1.00e-04
  sample: 我们曾经在这里
Early stopping

===== Pretrain Samples =====

1
A: 我们的时候
B: 我们都是一个人的人

2
A: 有一只鸟在窗前
B: 但是我不能说出

3
A: 而我只是看见她
B: 我们还在这个季节我们也是

4
A: 在我的身上漂泊了一张脸
B: 他们只是有一种种颜色

5
A: 就算我们也不能看见
B: 只有一朵云在暗涌的河边

Benchmark loss: 4.9730



Run 4: 在 Run 3 基础上，E1024 / H1024 → E1536 / H1536

train: 49932
val: 2628
vocab size: 5160

Epoch 01 | train loss: 5.5114 | val loss: 5.0920 | lr: 4.00e-04
  sample: 我是你的头鹰
  -> saved best model
Epoch 02 | train loss: 4.6475 | val loss: 4.9550 | lr: 4.00e-04
  sample: 他的意思是一个女人
  -> saved best model
Epoch 03 | train loss: 4.0914 | val loss: 4.9620 | lr: 4.00e-04
  sample: 我还是要活着一个人
Epoch 04 | train loss: 3.5858 | val loss: 5.0351 | lr: 2.00e-04
  sample: 我们不知道我们的家伙
Epoch 05 | train loss: 2.9484 | val loss: 5.0953 | lr: 2.00e-04
  sample: 我想起年轻的时候
Epoch 06 | train loss: 2.6563 | val loss: 5.1992 | lr: 1.00e-04
  sample: 我的眼睛挂在窗外
Epoch 07 | train loss: 2.3359 | val loss: 5.2619 | lr: 1.00e-04
  sample: 我们还能在这里面
Early stopping

===== Pretrain Samples =====

1
A: 我还不在
B: 他们都是一种

2
A: 如一个人的名字
B: 一个人们都是

3
A: 在那里面庞大的小船
B: 一切没有什么

4
A: 我们都不够我的一口
B: 我们的眼睛

5
A: 那是在我的一页
B: 那些是我们的

Benchmark loss: 4.9550



Run 5: 在 Run 4 基础上，batch_size 32 → 64

train: 49932
val: 2628
vocab size: 5160

Epoch 01 | train loss: 5.6471 | val loss: 5.1862 | lr: 4.00e-04
  sample: 如果我在一个人
  -> saved best model
Epoch 02 | train loss: 4.7999 | val loss: 4.9815 | lr: 4.00e-04
  sample: 我的脸上
  -> saved best model
Epoch 03 | train loss: 4.2866 | val loss: 4.9339 | lr: 4.00e-04
  sample: 我想给你一个人的身姿
  -> saved best model
Epoch 04 | train loss: 3.8314 | val loss: 4.9676 | lr: 4.00e-04
  sample: 就可以说话
Epoch 05 | train loss: 3.4191 | val loss: 5.0339 | lr: 2.00e-04
  sample: 我们一样把自己摧毁了
Epoch 06 | train loss: 2.9061 | val loss: 5.0926 | lr: 2.00e-04
  sample: 这是一个大雪不止的人
Epoch 07 | train loss: 2.6743 | val loss: 5.1674 | lr: 1.00e-04
  sample: 他们一定会受到责罚的人
Epoch 08 | train loss: 2.4183 | val loss: 5.2118 | lr: 1.00e-04
  sample: 我的耳朵听见
Early stopping

===== Pretrain Samples =====

1
A: 一直都没有改变
B: 一些年轻的时候

2
A: 我们的爱情和我们的爱情
B: 是否有人知道

3
A: 我们不知道自己的时间
B: 我们不知道谁是为了

4
A: 他就能看见
B: 她的手机没有被遗忘的力量

5
A: 我的爱情是一只手
B: 我们的一些

Benchmark loss: 4.9339



Run 6: 在 Run 5 基础上，扩大语料库

train: 147387
val: 7757
vocab size: 5160

Epoch 01 | train loss: 5.2192 | val loss: 4.8387 | lr: 4.00e-04
  sample: 它们是一个人
  -> saved best model
Epoch 02 | train loss: 4.5774 | val loss: 4.7094 | lr: 4.00e-04
  sample: 我们都说不出来
  -> saved best model
Epoch 03 | train loss: 4.2220 | val loss: 4.6901 | lr: 4.00e-04
  sample: 我要说话你的诗
  -> saved best model
Epoch 04 | train loss: 3.9214 | val loss: 4.7250 | lr: 4.00e-04
  sample: 我是在这个时代的夜里
Epoch 05 | train loss: 3.6591 | val loss: 4.7857 | lr: 2.00e-04
  sample: 不知是否还有这个世界
Epoch 06 | train loss: 3.2673 | val loss: 4.8150 | lr: 2.00e-04
  sample: 我想要一次呼吸
Epoch 07 | train loss: 3.0942 | val loss: 4.8740 | lr: 1.00e-04
  sample: 我们已经很多年了
Epoch 08 | train loss: 2.8757 | val loss: 4.9079 | lr: 1.00e-04
  sample: 一些事物的眼睛
Early stopping

===== Pretrain Samples =====

1
A: 我们不知道什么时候
B: 一群人在梦魇

2
A: 我的心就没有
B: 我们的心里的人啊

3
A: 她有着我和你们的孩子
B: 我的梦魇里有些感慨

4
A: 有时是在我们的心里
B: 她的命运在这里

5
A: 那些树枝条在树林中
B: 我只是一个没有人会

Benchmark loss: 4.4501



Run 7: 在 Run 6 基础上，扩大语料库

train: 612349
val: 32228
vocab size: 5160

Epoch 01 | train loss: 4.8412 | val loss: 4.5665 | lr: 4.00e-04
  sample: 那是一个人的世界
  -> saved best model
Epoch 02 | train loss: 4.4330 | val loss: 4.4917 | lr: 4.00e-04
  sample: 一个人站在那边
  -> saved best model
Epoch 03 | train loss: 4.2676 | val loss: 4.4828 | lr: 4.00e-04
  sample: 一些人在那儿
  -> saved best model
Epoch 04 | train loss: 4.1495 | val loss: 4.4965 | lr: 4.00e-04
  sample: 一个人活着
Epoch 05 | train loss: 4.0574 | val loss: 4.5168 | lr: 2.00e-04
  sample: 一个人的身体和一个个
Epoch 06 | train loss: 3.8477 | val loss: 4.4964 | lr: 2.00e-04
  sample: 我不再想起你
Epoch 07 | train loss: 3.7587 | val loss: 4.5151 | lr: 1.00e-04
  sample: 我们的爱情是
Epoch 08 | train loss: 3.6293 | val loss: 4.5201 | lr: 1.00e-04
  sample: 只有一片空白的月光
Early stopping

===== Pretrain Samples =====

1
A: 一直跟着他
B: 我的诗在那里

2
A: 一些事物都变成了
B: 在你最后的梦里

3
A: 就是把它们全部删除
B: 一直以为我是不是一个老人

4
A: 你们的眼睛像一条
B: 一个个被我们

5
A: 我一直在下午点一点
B: 只是一个自足的老哥

Benchmark loss: 4.1692



Run Final: 用原始数据对模型进行 fine-tune

train: 1292
val: 159
vocab size: 5160

Loaded pretrained model.

Epoch 01 | train loss: 4.1290 | val loss: 3.9263 | lr: 4.00e-05
  sample: 这是在我的记忆里
  -> saved best model
Epoch 02 | train loss: 3.9761 | val loss: 3.8923 | lr: 4.00e-05
  sample: 我的心就是这样一个
  -> saved best model
Epoch 03 | train loss: 3.8726 | val loss: 3.8644 | lr: 4.00e-05
  sample: 不可遏止的
  -> saved best model
Epoch 04 | train loss: 3.7369 | val loss: 3.8417 | lr: 4.00e-05
  sample: 一片叶子飘落
  -> saved best model
Epoch 05 | train loss: 3.6336 | val loss: 3.8225 | lr: 4.00e-05
  sample: 一个女孩子的小路
  -> saved best model
Epoch 06 | train loss: 3.5248 | val loss: 3.8051 | lr: 4.00e-05
  sample: 一次次的交谈
  -> saved best model
Epoch 07 | train loss: 3.4260 | val loss: 3.7910 | lr: 4.00e-05
  sample: 这样的人是一个不幸的
  -> saved best model
Epoch 08 | train loss: 3.3286 | val loss: 3.7793 | lr: 4.00e-05
  sample: 我只是在这样的夜空中
  -> saved best model
Epoch 09 | train loss: 3.2397 | val loss: 3.7698 | lr: 4.00e-05
  sample: 一个人走了回来
  -> saved best model
Epoch 10 | train loss: 3.1578 | val loss: 3.7618 | lr: 4.00e-05
  sample: 不再把这里的夜空
  -> saved best model
Epoch 11 | train loss: 3.0661 | val loss: 3.7562 | lr: 4.00e-05
  sample: 我曾经在你的手里
  -> saved best model
Epoch 12 | train loss: 2.9821 | val loss: 3.7512 | lr: 4.00e-05
  sample: 他的身影是一座村庄
  -> saved best model
Epoch 13 | train loss: 2.9003 | val loss: 3.7470 | lr: 4.00e-05
  sample: 我想我会在你的世界里
  -> saved best model
Epoch 14 | train loss: 2.8299 | val loss: 3.7432 | lr: 4.00e-05
  sample: 我已经走过这里
  -> saved best model
Epoch 15 | train loss: 2.7557 | val loss: 3.7408 | lr: 4.00e-05
  sample: 我们还给你带来的
  -> saved best model
Epoch 16 | train loss: 2.6840 | val loss: 3.7399 | lr: 4.00e-05
  sample: 我们没有说出口的话
  -> saved best model
Epoch 17 | train loss: 2.6107 | val loss: 3.7386 | lr: 4.00e-05
  sample: 我们曾经在过去
  -> saved best model
Epoch 18 | train loss: 2.5428 | val loss: 3.7379 | lr: 4.00e-05
  sample: 也许已经很久
  -> saved best model
Epoch 19 | train loss: 2.4769 | val loss: 3.7379 | lr: 4.00e-05
  sample: 一场雨停在我的肩上
Epoch 20 | train loss: 2.4082 | val loss: 3.7385 | lr: 4.00e-05
  sample: 只是把它们拉进来
Epoch 21 | train loss: 2.3430 | val loss: 3.7396 | lr: 4.00e-05
  sample: 我们都有自己的名字
Early stopping

===== Fine-tuned Samples =====

1
A: 我想把这事儿
B: 一个人等着一个人

2
A: 我把我们的名字
B: 也许有人正在走着

3
A: 看一颗一颗的小脸
B: 不必再也不能再去哪里

4
A: 我们没有说出口
B: 我曾经把手的时间

5
A: 也不会有人会有更多的地方
B: 不必知道是在哪里

Before fine-tuning | val loss: 3.9668
Best fine-tuned val loss: 3.7379

在训练期 batch-average 标准下：
Run 0 best val: 5.3966
Pretrained before FT: 3.9668
Fine-tuned best val: 3.7379

在最终 token-average 标准下：
Fine-tuned val: 3.7171
Fine-tuned test: 3.3620
Test perplexity: 28.85

注：Run 0 vocab size 为 1555，最终模型 vocab size 为 5160，两者并非严格受控的 loss 对比。