# Adder_practice

-------------------------------
Win10 home -> 
Python 3.6.5 -> 
IDE -> jupyter notebook ->
-------------------------------

共6個檔案 ->
adder(加法) -> https://nbviewer.jupyter.org/github/Hoshey/DSAI_adder_practice/blob/master/Adder.ipynb

subtractor(減法) -> 大數減小數或小數減大數皆可 ->
https://nbviewer.jupyter.org/github/Hoshey/DSAI_adder_practice/blob/master/Subtractor.ipynb

adder_subtractor (加減法) -> 上面2個結合 ->
https://nbviewer.jupyter.org/github/Hoshey/DSAI_adder_practice/blob/master/Adder_Subtractor.ipynb 

adder_subtractor-digits        -> 改變 digits ->
https://nbviewer.jupyter.org/github/Hoshey/DSAI_adder_practice/blob/master/Adder_Subtractor-digit.ipynb

adder_subtractor-epoch         -> 改變 epoch ->
https://nbviewer.jupyter.org/github/Hoshey/DSAI_adder_practice/blob/master/Adder_Subtractor-epoch.ipynb

adder_subtractor-training size -> 改變 training size ->
https://nbviewer.jupyter.org/github/Hoshey/DSAI_adder_practice/blob/master/Adder_Subtractor-training%20size.ipynb



model：LSTM(bidirection)
-> encoder (input)
-> decoder (a single layer) for training
-> output
-> 參數：digits=3, epoch=1, training size=30000, layer=1, 迭代次數=100, hidden_size=128, batch_size=128

-----------------------
參數調整，以 adder_subtractor 為例子調整 digits, epoch, training size

原始 testing accuracy >> 54.32% 

-----------------------
digits = 3 -> 5

模型可以正常運作，不過在不更改training epoch、training size 的情形下，5位數運算預測效果比較差。

testing accuracy >> 5.44%

-----------------------
epoch = 1 -> 2

不變的digit、training size下，從1變2因為改變不大，可以有更好的預測效果，不過效果不會太顯著，時間上也需要比較久，若改變幅度變大，效果更好不過可能需要注意 over-fiiting 。

testing accuracy >> 61.38%

-----------------------
training size = 30000 -> 60000

基本上多多益善，越多資料可以達成的預測效果可以越好，因為是自行生成資料，所以可以一直往上增加數字，不過在不改變training epoch、迭代次數等情況下，若資料量太龐大可能無法收斂到合適的模型。
 
testing accuracy >> 89.86%

-----------------------
原則上可以將程式設計成 multiplication ，畢竟乘法從加法而來，不過調整上比起改成減法會大很多，例如最大位數的調整上，隨著digits 的增加1，最大位數的調整至少乘以10^2。



 
