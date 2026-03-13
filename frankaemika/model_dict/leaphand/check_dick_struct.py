# function to check the structure of the model dictionary

dict_path = '/workspace/cdf/frankaemika/model_dict/leaphand/leaphand_finger1_mlp_base.pt'


import torch
model_dict = torch.load(dict_path, weights_only=False)
for key in model_dict:
    print(f'--- Key: {key} ---')
    # model_dict[key] is collections.OrderedDict
    for sub_key in model_dict[key]:
        print(f'  Sub-key: {sub_key}, Shape: {model_dict[key][sub_key].shape}')
        print(f'    shape: {model_dict[key][sub_key].shape}, dtype: {model_dict[key][sub_key].dtype}')
