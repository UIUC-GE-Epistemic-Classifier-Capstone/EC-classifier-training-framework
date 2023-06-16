# import torch
# from torch import nn
# import torch.nn.functional as F
# import numpy as np
# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.h, self.w = 144, 256#cfg['data']['image_size']['h'], cfg['data']['image_size']['w']
#         self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
#         self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.fc1 = nn.Linear(64 * ((self.h//2)//2) * ((self.w//2)//2), 128)
#         self.fc2 = nn.Linear(128, 2) # 3
#
#     def forward(self, x):
#         # x = x.permute(0,3,2,1)
#         x = F.relu(self.conv1(x))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = self.pool(F.relu(self.conv3(x)))
#         x = x.view(-1,64 * ((self.h//2)//2) * ((self.w//2)//2))
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
#
# # import torch
# #
# # def get_layer_output(input, model, layer_name):
# #     # Enable evaluation mode
# #     model.eval()
# #
# #     # Dictionary to store the intermediate outputs
# #     layer_outputs = {}
# #
# #     # Register a forward hook to collect the outputs of each layer
# #     def hook(module, input, output):
# #         layer_outputs[layer_name] = output
# #
# #     # Find the desired layer by name
# #     target_layer = None
# #     for name, module in model.named_modules():
# #         if name == layer_name:
# #             target_layer = module
# #             break
# #
# #     if target_layer is None:
# #         raise ValueError(f"Layer '{layer_name}' not found in the model.")
# #
# #     # Register the hook to collect the output of the desired layer
# #     hook_handle = target_layer.register_forward_hook(hook)
# #
# #     # Forward pass through the model
# #     model(input)
# #
# #     # Remove the hook
# #     hook_handle.remove()
# #
# #     # Retrieve the output of the desired layer
# #     output = layer_outputs[layer_name]
# #
# #     return output
#
#
# # import torch
# # def get_layer_output(input, model, layer_name):
# #     # Enable evaluation mode
# #     model.eval()
# #
# #     # Dictionary to store the intermediate outputs
# #     layer_outputs = {}
# #
# #     # Register a forward hook to collect the outputs of each layer
# #     def hook(module, input, output):
# #         layer_outputs[layer_name[i]] = output
# #
# #     # Find the desired layer by name
# #     target_layer = []
# #     for name, module in model.named_modules():
# #         if name in layer_name:
# #             target_layer.append(module)
# #             break
# #
# #     if target_layer is None:
# #         raise ValueError(f"Layer '{layer_name}' not found in the model.")
# #
# #     # Register the hook to collect the output of the desired layer
# #     hook_handle = []
# #     for i in range(len(target_layer)):
# #         hook_handle.append(target_layer[i].register_forward_hook(hook))
# #
# #     # Forward pass through the model
# #     model(input)
# #
# #     # Remove the hook
# #     for i in range(len(hook_handle)):
# #         hook_handle.remove()
# #
# #     # Retrieve the output of the desired layer
# #     output = []
# #     for i in range(layer_name):
# #         output.append(layer_outputs[layer_name[i]])
# #
# #     return output
#
# # import torch
#
# def get_layer_outputs(input, model, layer_names):
#     # Enable evaluation mode
#     model.eval()
#
#     # Dictionary to store the intermediate outputs
#     layer_outputs = {}
#
#     # Register a forward hook for each desired layer
#     hook_handles = []
#     for layer_name in layer_names:
#         layer_outputs[layer_name] = []
#
#         # Find the desired layer by name
#         target_layer = None
#         for name, module in model.named_modules():
#             if name == layer_name:
#                 target_layer = module
#                 break
#
#         if target_layer is None:
#             raise ValueError(f"Layer '{layer_name}' not found in the model.")
#
#         # Register the hook to collect the output of the desired layer
#         def hook_fn(module, input, output, layer_name=layer_name):
#             layer_outputs[layer_name].append(output)
#
#         hook_handles.append(target_layer.register_forward_hook(hook_fn))
#
#     # Forward pass through the model
#     model(input)
#
#     # Remove the hooks
#     for handle in hook_handles:
#         handle.remove()
#     data_all = []
#     for layer_name, outputs in layer_outputs.items():
#         print(f"Layer: {layer_name}")
#         for output in outputs:
#             if len(output.shape) == 4:
#                 res = output.reshape(input.shape[0], -1)
#             else:
#                 res = output
#             data_all.append(res)
#     return data_all
#
#
# x = torch.rand(5,3,144, 256)
# model = CNN()
# layer_name = ["conv1", "fc1", "fc2"]
# output = get_layer_outputs(x, model, layer_name)
# # for layer_name, outputs in output.items():
# #     print(f"Layer: {layer_name}")
# #     for output in outputs:
# #         print(output.shape)
# # output = np.array(output.numpy())
# print(output[2].shape)

import numpy as np
a = np.array([1,2,3,4,5])
b = [0,2]
print(a[b])