# Hefei note

按照原有的pipeline训练模型，生成原始数据为`data_again/npy`文件，存放在`cdf/frankaemika`目录下。processed数据(fps降采样+去除无解的点)存放在`cdf/frankaemika`目录下`data_again.pt`文件中。训练模型生成的checkpoint存放在`cdf/frankaemika/my_model_dict.pt`文件中。  

原本提供的模型`model_dict.pt`在`cdf/frankaemika`目录下。相应的可视化测试结果在`cdf/frankaemika/origin_eval`目录下。

evaluation:
model_dict_signed.pt: 训练时使用了带符号的sdf数据，正值表示点在物体外部，负值表示点在物体内部。  
model_dict.pt: 训练时使用了不带符号的sdf数据，所有点的sdf值均为正值。
model_dict.pt:  
iter 999 finished, MAE:0.03997863084077835      RMSE:0.0755908414721489 SR:0.714  
MAE:0.04766484459489584 RMSE:0.07928665001690388        SR:0.714844  
MAE:0.03856360154777018 RMSE:0.06146772432750912        SR:0.22516934441437628  
model_dict_signed.pt:   
iter 999 finished, MAE:0.10256706178188324      RMSE:0.15629196166992188        SR:0.355  
MAE:0.04908456335961819 RMSE:0.08126681592827663        SR:0.70707  
MAE:0.038315843551861954        RMSE:0.06175301788126843        SR:0.22733902238727074  
model_dict_signed_2.pt:  
在signed sdf的基础上纠正了grad的符号，与sdf符号一致。  
iter 999 finished, MAE:0.013364768587052822     RMSE:0.03129984810948372        SR:0.946  
MAE:0.04721151122474112 RMSE:0.07822127080266364        SR:0.7205940000000001  
MAE:0.03850634412244638 RMSE:0.061720250617808885       SR:0.23037556112574095  
