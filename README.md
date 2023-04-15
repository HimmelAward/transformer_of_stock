# The Stock Preditction of Transformer
transformer最开始是用于自然语言处理的，我们知道自然语言属于一种带有一定时序信息的的数据，那么它必然可以用于类似seq2seq的时序信息的预测，例如股票预测。
加之transformer的全局注意力机制，即可以在全局的角度进行对股票预测，所以应当在长期投票预测占据优势。
此github中利用transformer，根据输入的数据预测单个或多个股票参数，在config定义你的参数，在pre_trainer trainer中选择你想预测的参数，models保存的为模型
此次采用的数据为股票中按天的收盘价，开盘价，最高价，最低价，预测结果如下。
![9](https://user-images.githubusercontent.com/66540608/232195705-e0b7c993-4303-47d6-aef4-82f6f4eb86e3.png)![0](https://user-images.githubusercontent.com/66540608/232195714-e79bbcad-385d-4d39-a8fc-2a3fd8387ee4.png)![1](https://user-images.githubusercontent.com/66540608/232195716-59d29e2f-a482-45b2-b842-181546a7c831.png)![3](https://user-images.githubusercontent.com/66540608/232195719-aa1247c3-e012-4512-8fbd-48752d1c3b95.png)![5](https://user-images.githubusercontent.com/66540608/232195720-797dc5fb-cd46-4574-a517-787ae010b585.png)
