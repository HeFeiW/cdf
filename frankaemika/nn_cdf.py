# -----------------------------------------------------------------------------
# SPDX-License-Identifier: MIT
# This file is part of the CDF project.
# Copyright (c) 2024 Idiap Research Institute <contact@idiap.ch>
# Contributor: Yimming Li <yiming.li@idiap.ch>
# -----------------------------------------------------------------------------


# 7D panda robot
import numpy as np
import os
import sys
import torch
import math
import time
import argparse
CUR_PATH = os.path.dirname(os.path.realpath(__file__))
from mlp import MLPRegression
sys.path.append(os.path.join(CUR_PATH,'../../RDF/panda_layers'))
from robot_layer import RobotLayer
import bf_sdf

PI = math.pi
# torch.manual_seed(10)
np.random.seed(10)
# torch.autograd.set_detect_anomaly(True)

class CDF:
    def __init__(self,device,paths,robot,signed_distance=False,writer=None) -> None:
        # device
        self.device = device  
        self.writer = writer
        self.batch_x = 10
        self.batch_q = 100
        self.max_q_per_link = 100
        # # uncomment these lines to process the generated data and train your own CDF
        # self.raw_data = np.load(paths['raw_data'],allow_pickle=True).item()
        # self.process_data(self.raw_data)
        # self.data_path = paths['data']
        # self.data = self.load_data(self.data_path)
        # self.len_data = len(self.data['k'])
        self.signed_distance = signed_distance
        # panda robot
        self.robot = RobotLayer(device=device,paths=paths,robot=robot)
        self.paths = paths
        self.bp_sdf_model_path = paths['model']
        self.bp_sdf = bf_sdf.BPSDF(8,-1.0,1.0,self.robot,self.bp_sdf_model_path,device)
        self.bp_sdf_model = torch.load(self.bp_sdf_model_path)
        self.model_dict = paths['model_dict']
        # --- TODO ---
        if robot == 'panda':
            self.used_joints = [0,1,2,3,4,5,6]

    def process_data(self,data):
        # 从原始数据（data）中降采样（pytorch3d.ops.sample_farthest_points）每个关节的采样点,
        # "每个关节"指的是，在每个采样点上，每个关节作为“最后碰撞关节”的采样点数量不超过max_q_per_link
        # data: {key: {'x': (N,3), 'q': (N,7), 'idx': (N)}}
        # idx: (N) stands for which link is the last link that causes the collision
        # final_data: {'x': (G,3), 'q': (G,max_q_per_link,7,7), 'k': (G)}
        # G is the number of grids
        import pytorch3d.ops 
        keys = list(data.keys())  # Create a copy of the keys
        processed_data = {}
        # print('data_shape:',{k:data[k].shape for k in keys})

        for k in keys:# 每个key对应一个采样点，即Cartesian空间中的一个grid点
            if len(data[k]['q']) == 0:
                # processed_data[k] = {
                #     'x':torch.from_numpy(data[k]['x']).float().to(self.device),
                #     # 如果没有采样点，就用inf填充
                #     'q': torch.inf*torch.ones(self.max_q_per_link,7,7).to(self.device)
                # }
                data.pop(k)
                continue
            q = torch.from_numpy(data[k]['q']).float().to(self.device)
            # q:(N,7) N: number of samples for this key
            q_idx = torch.from_numpy(data[k]['idx']).float().to(self.device)
            # q_idx:(N) stands for which link is the last link that causes the collision
            q_idx[q_idx==7] = 6
            q_idx[q_idx==8] = 7
            q_lib = torch.inf*torch.ones(self.max_q_per_link,7,7).to(self.device)
            for i in range(1,8):
                mask = (q_idx==i) # find the samples where the last colliding link is i
                # 如果这个link的采样点多于max_q_per_link，就用farthest point sampling降采样，否则就直接存储
                if len(q[mask])>self.max_q_per_link:
                    # print(f'key {k}, link {i}, original num samples: {len(q[mask])}')
                    fps_q = pytorch3d.ops.sample_farthest_points(q[mask].unsqueeze(0),K=self.max_q_per_link)[0]
                    q_lib[:,:,i-1] = fps_q.squeeze()
                    # print(q_lib[:,:,i]) 
                elif len(q[mask])>0:
                    q_lib[:len(q[mask]),:,i-1] = q[mask]
            # q_lib:(max_q_per_link,7,7) 7: number of links
            # (first "7" stands for index of the q sample,
            # and second "7" stands for the link index,
            # i.e. to get all sample for link i, use q_lib[:,:,i-1])
            processed_data[k] = {
                'x':torch.from_numpy(data[k]['x']).float().to(self.device),
                'q':q_lib,
            }
        final_data = {
            'x': torch.cat([processed_data[k]['x'].unsqueeze(0) for k in processed_data.keys()],dim=0),
            'q': torch.cat([processed_data[k]['q'].unsqueeze(0) for k in processed_data.keys()],dim=0),
            'k':torch.tensor([k for k in processed_data.keys()]).to(self.device)
        }
        # print('final_data:',final_data['x'].shape,final_data['q'].shape,final_data['k'].shape)
        torch.save(final_data,os.path.join(CUR_PATH,'data_again.pt'))
        return data
    
    def load_data(self,path):
        data = torch.load(path)
        return data

    def select_data(self):
        # x_batch:(batch_x,3)
        # q_batch:(batch_q,7)
        # d:(batch_x,batch_q)
        
        x = self.data['x']
        q = self.data['q']

        idx = torch.randint(0,len(x),(self.batch_x,)) 
        # idx = torch.tensor([4000])
        x_batch,q_lib = x[idx],q[idx]
        # print(x_batch)
        q_batch = self.sample_q()   
        d,grad = self.decode_distance(q_batch,q_lib)
        return x_batch,q_batch,d,grad
    def select_data_signed(self):
        # x_batch:(batch_x,3)
        # q_batch:(batch_q,7)
        # d:(batch_x,batch_q)
        
        x = self.data['x']
        q = self.data['q']

        idx = torch.randint(0,len(x),(self.batch_x,)) 
        # idx = torch.tensor([4000])
        x_batch,q_lib = x[idx],q[idx]
        # print(x_batch)
        q_batch = self.sample_q()   
        d,grad = self.decode_distance_signed(q_batch,q_lib,x_batch)
        return x_batch,q_batch,d,grad

    def decode_distance(self,q_batch,q_lib):
        # batch_q:(batch_q,7)
        # q_lib:(batch_x,self.max_q_per_link,7,7)

        batch_x = q_lib.shape[0]
        batch_q = q_batch.shape[0]
        d_tensor = torch.ones(batch_x,batch_q,7).to(self.device)*torch.inf
        grad_tensor  = torch.zeros(batch_x,batch_q,7,7).to(self.device)
        for i in range(7):
            q_lib_temp = q_lib[:,:,:i+1,i].reshape(batch_x*self.max_q_per_link,-1).unsqueeze(0).expand(batch_q,-1,-1)
            q_batch_temp = q_batch[:,:i+1].unsqueeze(1).expand(-1,batch_x*self.max_q_per_link,-1)
            d_norm = torch.norm((q_batch_temp - q_lib_temp),dim=-1).reshape(batch_q,batch_x,self.max_q_per_link)

            d_norm_min,d_norm_min_idx = d_norm.min(dim=-1)
            # print(f'd_norm_min.shape:{d_norm_min.shape}')
            # print(d_norm_min)
            grad = torch.autograd.grad(d_norm_min.reshape(-1),q_batch_temp,torch.ones_like(d_norm_min.reshape(-1)),retain_graph=True)[0]
            grad_min_q = grad.reshape(batch_q,batch_x,self.max_q_per_link,-1).gather(2,d_norm_min_idx.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,-1,i+1))[:,:,0,:]
            grad_tensor[:,:,:i+1,i] = grad_min_q.transpose(0,1)
            d_tensor[:,:,i] = d_norm_min.transpose(0,1)

        d,d_min_idx = d_tensor.min(dim=-1)
        grad_final = grad_tensor.gather(3,d_min_idx.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,7,7))[:,:,:,0]
        return d, grad_final
    def decode_distance_signed(self,q_batch,q_lib,x_batch):
        # batch_q:(batch_q,7)
        # q_lib:(batch_x,self.max_q_per_link,7,7)
        # x_batch:(batch_x,3)
        # return d:(batch_x,batch_q)
        batch_x = q_lib.shape[0]
        batch_q = q_batch.shape[0]
        d_tensor = torch.ones(batch_x,batch_q,7).to(self.device)*torch.inf
        grad_tensor  = torch.zeros(batch_x,batch_q,7,7).to(self.device)
        for i in range(7):
            q_lib_temp = q_lib[:,:,:i+1,i].reshape(batch_x*self.max_q_per_link,-1).unsqueeze(0).expand(batch_q,-1,-1)
            q_batch_temp = q_batch[:,:i+1].unsqueeze(1).expand(-1,batch_x*self.max_q_per_link,-1)
            d_norm = torch.norm((q_batch_temp - q_lib_temp),dim=-1).reshape(batch_q,batch_x,self.max_q_per_link)
            d_norm_min,d_norm_min_idx = d_norm.min(dim=-1)
            grad = torch.autograd.grad(d_norm_min.reshape(-1),q_batch_temp,torch.ones_like(d_norm_min.reshape(-1)),retain_graph=True)[0]
            grad_min_q = grad.reshape(batch_q,batch_x,self.max_q_per_link,-1).gather(2,d_norm_min_idx.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,-1,i+1))[:,:,0,:]
            grad_tensor[:,:,:i+1,i] = grad_min_q.transpose(0,1)
            d_tensor[:,:,i] = d_norm_min.transpose(0,1)
        d,d_min_idx = d_tensor.min(dim=-1)
        d_ts = self.compute_sdf(x_batch,q_batch)
        mask =  (d_ts < 0).transpose(0,1)
        d[mask] = -d[mask]
        grad_final = grad_tensor.gather(3,d_min_idx.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,7,7))[:,:,:,0]
        grad_final[mask] = -grad_final[mask]  # 对有符号距离的梯度也取负号
        return d, grad_final
    def compute_sdf(self,x,q,return_index = False):
        # x : (Nx,3)
        # q : (Nq,7)
        # return_index : if True, return the index of link that is closest to x
        # return d : (Nq,Nx) the signed distance from x to q
        # return idx : (Nq) optional

        pose = torch.eye(4).unsqueeze(0).to(self.device).expand(len(q),4,4).float()
        if not return_index:
            d,_ = self.bp_sdf.get_whole_body_sdf_batch(x,pose, q,self.bp_sdf_model,use_derivative =False)
            # d = d.min(dim=1)[0]
            return d
        else:
            d,_,idx = self.bp_sdf.get_whole_body_sdf_batch(x,pose, q,self.bp_sdf_model,use_derivative =False,return_index = True)
            # d,pts_idx = d.min(dim=1)
            idx = idx[torch.arange(len(idx)),idx]
            return d,idx 
    def sample_q(self,batch_q = None):
        if batch_q is None:
            batch_q = self.batch_q
        q_sampled = self.robot.theta_min + torch.rand(batch_q,7).to(self.device)*(self.robot.theta_max-self.robot.theta_min)
        q_sampled.requires_grad = True
        return q_sampled
    
    def projection(self,q,d,grad):
        q_new = q - grad*d.unsqueeze(-1)
        return q_new

    def train_nn(self,epoches=500):
        # model
        # input: [x,q] (B,3+7)
        model = MLPRegression(input_dims=10, output_dims=1, mlp_layers=[1024, 512, 256, 128, 128],skips=[], act_fn=torch.nn.ReLU, nerf=True)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5000,
                                                        threshold=0.01, threshold_mode='rel',
                                                        cooldown=0, min_lr=0, eps=1e-04, verbose=True)
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        COSLOSS = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        model_dict = {}
        for iter in range(epoches):
            model.train()
            with torch.cuda.amp.autocast():
                if self.signed_distance:
                    x_batch,q_batch,d,gt_grad = self.select_data_signed()
                else:
                    x_batch,q_batch,d,gt_grad = self.select_data()
                # x_batch:(batch_x,3)
                # q_batch:(batch_q,7)
                # d:(batch_x,batch_q)
                # grad:(batch_x,batch_q,7)
                x_inputs = x_batch.unsqueeze(1).expand(-1,self.batch_q,-1).reshape(-1,3)
                q_inputs = q_batch.unsqueeze(0).expand(self.batch_x,-1,-1).reshape(-1,7)

                inputs = torch.cat([x_inputs,q_inputs],dim=-1)
                outputs = d.reshape(-1,1)
                gt_grad = gt_grad.reshape(-1,7)
                weights = torch.ones_like(outputs).to(device)
                # weights = (1/outputs).clamp(0,1)

                d_pred = model.forward(inputs)
                d_grad_pred = torch.autograd.grad(d_pred, q_inputs, torch.ones_like(d_pred), retain_graph=True,create_graph=True)[0]
                # Compute the Eikonal loss
                eikonal_loss = torch.abs(d_grad_pred.norm(2, dim=-1) - 1).mean()

                # Compute the tension loss
                dd_grad_pred = torch.autograd.grad(d_grad_pred, q_inputs, torch.ones_like(d_grad_pred), retain_graph=True,create_graph=True)[0]

                # gradient loss
                gradient_loss = (1 - COSLOSS(d_grad_pred, gt_grad)).mean()
                # tension loss
                tension_loss = dd_grad_pred.square().sum(dim=-1).mean()
                # Compute the MSE loss
                d_loss = ((d_pred-outputs)**2*weights).mean()

                # Combine the two losses with appropriate weights
                w0 = 5.0
                w1 = 0.01
                w2 = 0.01
                w3 = 0.1
                loss = w0 * d_loss + w1 * eikonal_loss + w2 * tension_loss + w3 * gradient_loss

                # # Print the losses for monitoring

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step(loss)
                if iter % 10 == 0:
                    print(f"Epoch:{iter}\tMSE Loss: {d_loss.item():.3f}\tEikonal Loss: {eikonal_loss.item():.3f}\tTension Loss: {tension_loss.item():.3f}\tGradient Loss: {gradient_loss.item():.3f}\tTotal loss:{loss.item():.3f}\tTime: {time.strftime('%H:%M:%S', time.gmtime())}")
                    model_dict[iter] = model.state_dict()
                    if self.writer is not None:
                        self.writer.add_scalar('loss/d_loss', d_loss.item(), iter)
                        self.writer.add_scalar('loss/eikonal_loss', eikonal_loss.item(), iter)
                        self.writer.add_scalar('loss/tension_loss', tension_loss.item(), iter)
                        self.writer.add_scalar('loss/gradient_loss', gradient_loss.item(), iter)
                        self.writer.add_scalar('loss/total_loss', loss.item(), iter)
                    torch.save(model_dict, os.path.join(CUR_PATH,self.model_dict))
        return model
    
    def inference(self,x,q,model):
        model.eval()
        x,q = x.to(self.device),q.to(self.device)
        # q.requires_grad = True
        # x:(len(x),3)
        # q:(len(q),7)
        x_cat = x.unsqueeze(1).expand(-1,len(q),-1).reshape(-1,3)
        q_cat = q.unsqueeze(0).expand(len(x),-1,-1).reshape(-1,7)
        inputs = torch.cat([x_cat,q_cat],dim=-1)
        cdf_pred = model.forward(inputs)
        return cdf_pred
    
    def inference_d_wrt_q(self,x,q,model,return_grad = True):
        cdf_pred = self.inference(x,q,model)
        d = cdf_pred.reshape(len(x),len(q)).min(dim=0)[0]
        if return_grad:
            grad = torch.autograd.grad(d,q,torch.ones_like(d),retain_graph=True,create_graph=True)[0]
            # dgrad = torch.autograd.grad(grad,q,torch.ones_like(grad),retain_graph=True,create_graph=True)[0]
            return d,grad
        else:
            return d

    def eval_nn(self,model,num_iter = 3):
        eval_time = False
        eval_acc = True
        if eval_time:
            x = torch.rand(100,3).to(device)-torch.tensor([[0.5,0.5,0]]).to(device)
            q = self.sample_q(batch_q=100)
            time_cost_list = []
            for i in range(100):
                t0 = time.time()
                d = self.inference_d_wrt_q(x,q,model,return_grad = False)
                t1 = time.time()
                grad = torch.autograd.grad(d,q,torch.ones_like(d),retain_graph=True,create_graph=True)[0]
                q_proj = self.projection(q,d,grad)
                t2 = time.time()
                if i >0:
                    time_cost_list.append([t1-t0,t2-t1])
            mean_time_cost = np.mean(time_cost_list,axis=0)
            print(f'inference time cost:{mean_time_cost[0]}\t projection time cost: {mean_time_cost[1]}')

        if eval_acc:
            # bp_sdf model
            bp_sdf = self.bp_sdf
            bp_sdf_model = torch.load(self.bp_sdf_model_path)

            res = []
            for i in range (1000):
                x = torch.rand(1,3).to(device)-torch.tensor([[0.5,0.5,0]]).to(device)
                q = self.sample_q(batch_q=1000)
                for _ in range (num_iter):
                    d,grad = self.inference_d_wrt_q(x,q,model,return_grad = True)
                    q = self.projection(q,d,grad)
                q,grad = q.detach(),grad.detach()   # release memory
                pose = torch.eye(4).unsqueeze(0).expand(len(q),-1,-1).to(self.device).float()
                sdf,_ = bp_sdf.get_whole_body_sdf_batch(x, pose, q, bp_sdf_model,use_derivative=False)
                
                error = sdf.reshape(-1).abs()
                MAE = error.mean()
                RMSE = torch.sqrt(torch.mean(error**2))
                SR = (error<0.03).sum().item()/len(error)
                res.append([MAE.item(),RMSE.item(),SR])
                print(f'iter {i} finished, MAE:{MAE}\tRMSE:{RMSE}\tSR:{SR}')
            res = np.array(res)
            print(f'MAE:{res[:,0].mean()}\tRMSE:{res[:,1].mean()}\tSR:{res[:,2].mean()}')
            print(f'MAE:{res[:,0].std()}\tRMSE:{res[:,1].std()}\tSR:{res[:,2].std()}')

    def eval_nn_noise(self,model,num_iter = 3):
            bp_sdf = self.bp_sdf
            bp_sdf_model = torch.load(self.bp_sdf_model_path)

            res = []
            for i in range (1000):
                x = torch.rand(1,3).to(device)-torch.tensor([[0.5,0.5,0]]).to(device)
                noise = torch.normal(0,0.03,(1,3)).to(device)
                x_noise = x + noise
                q = self.sample_q(batch_q=1000)
                for _ in range (num_iter):
                    d,grad = self.inference_d_wrt_q(x_noise,q,model,return_grad = True)
                    q = self.projection(q,d,grad)
                q,grad = q.detach(),grad.detach()   # release memory
                pose = torch.eye(4).unsqueeze(0).expand(len(q),-1,-1).to(self.device).float()
                sdf,_ = bp_sdf.get_whole_body_sdf_batch(x, pose, q, bp_sdf_model,use_derivative=False)
                
                error = sdf.reshape(-1).abs()
                MAE = error.mean()
                RMSE = torch.sqrt(torch.mean(error**2))
                SR = (error<0.03).sum().item()/len(error)
                res.append([MAE.item(),RMSE.item(),SR])
                print(f'iter {i} finished, MAE:{MAE}\tRMSE:{RMSE}\tSR:{SR}')
            res = np.array(res)
            print(f'MAE:{res[:,0].mean()}\tRMSE:{res[:,1].mean()}\tSR:{res[:,2].mean()}')
            print(f'MAE:{res[:,0].std()}\tRMSE:{res[:,1].std()}\tSR:{res[:,2].std()}')

    def check_data(self):
        # x_batch:(batch_x,3)
        # q_batch:(batch_q,7)
        # d:(batch_x,batch_q)
        # grad:(batch_x,batch_q,7)
        x_batch,q_batch,d,grad = self.select_data()
        q_proj = self.projection(q_batch,d,grad)

        # visualize
        import trimesh
        pose = torch.eye(4).unsqueeze(0).to(self.device).float()
        for q0,q1 in zip(q_batch,q_proj[1]):
            scene = trimesh.Scene()
            scene.add_geometry(trimesh.PointCloud(x_batch.data.cpu().numpy(),colors=[255,0,0]))
            robot_mesh0 = self.robot.get_forward_robot_mesh(pose, q0.unsqueeze(0))[0]
            robot_mesh0 = np.sum(robot_mesh0)
            robot_mesh0.visual.face_colors = [0,255,0,100]
            scene.add_geometry(robot_mesh0)
            robot_mesh1 = self.robot.get_forward_robot_mesh(pose, q1.unsqueeze(0))[0]
            robot_mesh1 = np.sum(robot_mesh1)
            robot_mesh1.visual.face_colors = [0,0,255,100]
            scene.add_geometry(robot_mesh1)
            scene.show()
    def sample_without_fps(self,data):
        keys = list(data.keys())  # Create a copy of the keys
        processed_data = {}
        print('data_shape:',{k:data[k]['q'].shape for k in keys})

        for k in keys:
            if len(data[k]['q']) == 0:
                data.pop(k)
                continue
            q = torch.from_numpy(data[k]['q']).float().to(self.device)
            q_idx = torch.from_numpy(data[k]['idx']).float().to(self.device)
            q_idx[q_idx==7] = 6
            q_idx[q_idx==8] = 7
            q_lib = torch.inf*torch.ones(self.max_q_per_link,7,7).to(self.device)
            for i in range(1,8):
                mask = (q_idx==i)
                if len(q[mask])>self.max_q_per_link:
                    # 不用sample_farthest_points，而是随机采样
                    sampled_q = q[mask][torch.randperm(len(q[mask]))[:self.max_q_per_link]]
                    # print(f'sampled_q:{sampled_q.shape}')
                    # fps_q = pytorch3d.ops.sample_farthest_points(q[mask].unsqueeze(0),K=self.max_q_per_link)[0]
                    fps_q = sampled_q.unsqueeze(0)
                    q_lib[:,:,i-1] = fps_q.squeeze()
                    # print(q_lib[:,:,i]) 
                elif len(q[mask])>0:
                    q_lib[:len(q[mask]),:,i-1] = q[mask]

            processed_data[k] = {
                'x':torch.from_numpy(data[k]['x']).float().to(self.device),
                'q':q_lib,
            }
        final_data = {
            'x': torch.cat([processed_data[k]['x'].unsqueeze(0) for k in processed_data.keys()],dim=0),
            'q': torch.cat([processed_data[k]['q'].unsqueeze(0) for k in processed_data.keys()],dim=0),
            'k':torch.tensor([k for k in processed_data.keys()]).to(self.device)
        }
        print('final_data:',final_data['x'].shape,final_data['q'].shape,final_data['k'].shape)
        torch.save(final_data,os.path.join(CUR_PATH,'data.pt'))
        return data
        
    def eval_model(self,model,joint_idx,q_probe):

        # model
        # input: [x,q] (B,3+7)
        model = MLPRegression(input_dims=10, output_dims=1, mlp_layers=[1024, 512, 256, 128, 128],skips=[], act_fn=torch.nn.ReLU, nerf=True)
        # 取最后一个（字典值最大的）epoch的模型
        epoches = torch.load(os.path.join(CUR_PATH,self.model_dict)).keys()
        epoches = sorted(epoches)
        model.to(device)
        print(f'epoches:{epoches[-1]}')
        model.load_state_dict(torch.load(os.path.join(CUR_PATH,self.model_dict))[epoches[-1]])

        # model.load_state_dict(torch.load(os.path.join(CUR_PATH,'model_dict.pt'))[])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5000,
                                                        threshold=0.01, threshold_mode='rel',
                                                        cooldown=0, min_lr=0, eps=1e-04, verbose=True)
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        COSLOSS = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        model_dict = {}
        for _ in range(2):
            model.train()
            with torch.cuda.amp.autocast():
                x_batch,q_batch,d,gt_grad = self.select_data()

                x_inputs = x_batch.unsqueeze(1).expand(-1,self.batch_q,-1).reshape(-1,3)
                q_inputs = q_batch.unsqueeze(0).expand(self.batch_x,-1,-1).reshape(-1,7)

                inputs = torch.cat([x_inputs,q_inputs],dim=-1)
                outputs = d.reshape(-1,1)
                gt_grad = gt_grad.reshape(-1,7)
                weights = torch.ones_like(outputs).to(device)
                # weights = (1/outputs).clamp(0,1)

                d_pred = model.forward(inputs)
                d_grad_pred = torch.autograd.grad(d_pred, q_inputs, torch.ones_like(d_pred), retain_graph=True,create_graph=True)[0]
                # Compute the Eikonal loss
                eikonal_loss = torch.abs(d_grad_pred.norm(2, dim=-1) - 1).mean()

                # Compute the tension loss
                dd_grad_pred = torch.autograd.grad(d_grad_pred, q_inputs, torch.ones_like(d_grad_pred), retain_graph=True,create_graph=True)[0]

                # gradient loss
                gradient_loss = (1 - COSLOSS(d_grad_pred, gt_grad)).mean()
                # tension loss
                tension_loss = dd_grad_pred.square().sum(dim=-1).mean()
                # Compute the MSE loss
                d_loss = ((d_pred-outputs)**2*weights).mean()

                # Combine the two losses with appropriate weights
                w0 = 5.0
                w1 = 0.01
                w2 = 0.01
                w3 = 0.1
                loss = w0 * d_loss + w1 * eikonal_loss + w2 * tension_loss + w3 * gradient_loss

                # # Print the losses for monitoring

                # scaler.scale(loss).backward()
                # scaler.step(optimizer)
                # scaler.update()
                # optimizer.zero_grad()
                # scheduler.step(loss)
                # if iter % 10 == 0:
                print(f"Epoch:{iter}\tMSE Loss: {d_loss.item():.3f}\tEikonal Loss: {eikonal_loss.item():.3f}\tTension Loss: {tension_loss.item():.3f}\tGradient Loss: {gradient_loss.item():.3f}\tTotal loss:{loss.item():.3f}\tTime: {time.strftime('%H:%M:%S', time.gmtime())}")
                    # model_dict[iter] = model.state_dict()
                    # torch.save(model_dict, os.path.join(CUR_PATH,'my_model_dict.pt'))
        return model
    def my_eval_1(self,model):
        # model
        # input: [x,q] (B,3+7)
        # 在data中取出8个构成cube的点
        cube_poses = [[0,0,0],[0,0,10],[0,10,10],[5,10,10],[10,10,10],[10,10,0]]
        # cube_poses=[[10,10,0]]
        for cube_pos in cube_poses:
        
            cube_edge = 0 # cube的边长
            print(f'data_x_shape:{self.data["x"].shape}')
            cube_points  = torch.stack([self.data['x'][20*20*x+20*y+z] \
                                                        for x in [cube_pos[0],cube_pos[0]+cube_edge] \
                                                        for y in [cube_pos[1],cube_pos[1]+cube_edge]\
                                                        for z in [cube_pos[2],cube_pos[2]+cube_edge]]).float().to(device)
            cube_ground_truth_q  = torch.stack([self.data['q'][20*20*x+20*y+z] \
                                                        for x in [cube_pos[0],cube_pos[0]+cube_edge] \
                                                        for y in [cube_pos[1],cube_pos[1]+cube_edge]\
                                                        for z in [cube_pos[2],cube_pos[2]+cube_edge]]).float().to(device)
            print(f'cube_points:{cube_ground_truth_q.shape}')
            DoF = 7
            bp_sdf = self.bp_sdf
            bdf_model = torch.load(self.bp_sdf_model_path)
            q_max = self.robot.theta_max[self.used_joints]
            q_min = self.robot.theta_min[self.used_joints]
            # device
            self.device = device
            # 在DoF维度上采样test_sample_num个点
            test_sample_num = 1000
            q_sampled = q_min + torch.rand(test_sample_num,DoF).to(self.device)*(q_max-q_min)
            
            q_sampled.requires_grad = True
            # 获得模型预测的距离和梯度
            pred_d, pred_grad = self.inference_d_wrt_q(cube_points,q_sampled,model,return_grad = True)
            # 计算ground truth的距离和梯度
            from data_generator import DataGenerator
            data_generator = DataGenerator(self.device,self.robot,self.paths,used_joints=self.used_joints)
            gt_d = data_generator.distance_q(cube_points,q_sampled)
            # print(f'pred_d:{pred_d.shape}, gt_d:{gt_d.shape}')
            d_error = pred_d - gt_d
            # 把q_sampled, d_error, pred_grad, pred_d, gt_d reshape成(batch_size, DoF), 
            # 叠在一起，然后按照d_error的大小排序
            q_sampled = q_sampled.reshape(-1,DoF)
            d_error = d_error.reshape(-1,1)
            print(f'pred_grad:{pred_grad.shape}, pred_d:{pred_d.shape}, gt_d:{gt_d.shape}')
            pred_grad = pred_grad.reshape(-1,DoF)
            pred_d = pred_d.reshape(-1,1)
            gt_d = gt_d.reshape(-1,1)
            # 按照gt_d的大小排序
            sorted_idx = torch.argsort(gt_d,dim=0)
            sorted_idx = torch.linspace(0,gt_d.shape[0]-1,gt_d.shape[0],dtype=torch.long).to(self.device)
            q_sampled = q_sampled[sorted_idx]
            d_error = d_error[sorted_idx]
            pred_grad = pred_grad[sorted_idx]
            pred_grad_norm = torch.norm(pred_grad,dim=-1,keepdim=True)
            pred_d = pred_d[sorted_idx]
            gt_d = gt_d[sorted_idx]
            print(f'q_sampled:{q_sampled.shape}, d_error:{d_error.shape}, pred_grad:{pred_grad.shape}, pred_d:{pred_d.shape}, gt_d:{gt_d.shape}, pred_grad_norm:{pred_grad_norm.shape}')
            # 取前100个点，画出pred_d, gt_d, pred_grad, d_error的散点图，
            # 并保存到当前目录下的pred_gt_distance_gradient_scatter.png
            import matplotlib.pyplot as plt
            # import seaborn as sns
            # sns.set(style="whitegrid")
            plot_range = 1000
            plt.figure(figsize=(12, 8))
            plt.subplot(2, 2, 1)
            plt.scatter(gt_d[:plot_range].cpu().detach().numpy(), pred_d[:plot_range].cpu().detach().numpy(), c='blue', label='Predicted')
            plt.scatter(gt_d[:plot_range].cpu().detach().numpy(), gt_d[:plot_range].cpu().detach().numpy(), c='yellow', label='Ground Truth')
            plt.xlabel('Ground Truth Distance')
            plt.ylabel('Predicted Distance')
            plt.title('Predicted vs Ground Truth Distance')
            plt.legend()
            plt.subplot(2, 2, 2)
            plt.scatter(gt_d[:plot_range].cpu().detach().numpy(), pred_grad_norm[:plot_range].cpu().detach().numpy(), c='red', label='Predicted Gradient')
            plt.xlabel('Ground Truth Distance')
            plt.ylabel('Predicted Gradient')
            plt.title('Predicted Gradient')
            plt.legend()
            plt.subplot(2, 2, 3)
            plt.scatter(gt_d[:plot_range].cpu().detach().numpy(), d_error[:plot_range].cpu().detach().numpy(), c='green', label='Ground Truth vs Error')
            plt.xlabel('Ground Truth Distance')
            plt.ylabel('Distance Error')
            plt.title('Ground Truth vs Distance Error')
            plt.legend()
            plt.subplot(2, 2, 4)
            # ground truth distance 的直方图
            plt.hist(gt_d[:].squeeze(-1).cpu().detach().numpy(), bins=50, alpha=0.5, label='Ground Truth Distance', color='blue')
            # predicted distance 的直方图
            plt.hist(pred_d[:].squeeze(-1).cpu().detach().numpy(), bins=50, alpha=0.5, label='Predicted Distance', color='red')
            plt.xlabel('Distance')
            plt.ylabel('Frequency')
            plt.title('Distance Histogram')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(CUR_PATH,f'my_eval_{cube_pos[0]}_{cube_pos[1]}_{cube_pos[2]}_{cube_edge}.png'))
            plt.show()
            # 计算MAE和RMSE
            pred_d = pred_d.squeeze(-1).reshape(-1).cpu().detach().numpy()
            gt_d = gt_d.reshape(-1).cpu().detach().numpy()
            pred_grad = pred_grad.reshape(-1,DoF).cpu().detach().numpy()
            MAE_d = np.mean(np.abs(pred_d - gt_d))
            RMSE_d = np.sqrt(np.mean((pred_d - gt_d)**2))
            print(f'MAE_d: {MAE_d}, RMSE_d: {RMSE_d}')
    def my_eval_2(self,model):
        # model
        # input: [x,q] (B,3+7)
        # 在data中取出8个构成cube的点
        from data_generator import DataGenerator
        data_generator = DataGenerator(self.device,self.robot.robot,self.paths,used_joints=self.used_joints)
        cube_poses = [[0,0,0],[0,0,10],[0,10,10],[5,10,10],[10,10,10],[10,10,0]]
        # cube_poses=[[10,10,0]]
        for cube_pos in cube_poses:
        
            cube_edge = 0 # cube的边长
            cube_points  = torch.stack([self.data['x'][20*20*x+20*y+z] \
                                                        for x in [cube_pos[0],cube_pos[0]+cube_edge] \
                                                        for y in [cube_pos[1],cube_pos[1]+cube_edge]\
                                                        for z in [cube_pos[2],cube_pos[2]+cube_edge]]).float().to(device)
            cube_ground_truth_q  = torch.stack([self.data['q'][20*20*x+20*y+z] \
                                                        for x in [cube_pos[0],cube_pos[0]+cube_edge] \
                                                        for y in [cube_pos[1],cube_pos[1]+cube_edge]\
                                                        for z in [cube_pos[2],cube_pos[2]+cube_edge]]).float().to(device)
            # cube_points:(N,3) (N=sample_num)
            # cube_ground_truth_q:(N,100,DoF,DoF)
            DoF = 7
            bp_sdf = self.bp_sdf
            bdf_model = torch.load(self.bp_sdf_model_path)
            q_max = self.robot.theta_max[self.used_joints]
            q_min = self.robot.theta_min[self.used_joints]
            # 在DoF维度上采样test_sample_num个点
            test_sample_num = 1000
            q_sampled = torch.rand(DoF).to(self.device).unsqueeze(0).expand(test_sample_num,-1) * (q_max-q_min) + q_min
            eval_joint_idx = torch.randint(5,DoF,(1,)).to(self.device) 
            # 在5-6之间采样一个整数作为评估的关节（因为前面几个关节对末端影响较大，后面几个关节可能过于平滑）
            print(f'eval_joint_idx:{eval_joint_idx}')
            q_sampled[:,eval_joint_idx] = torch.linspace(\
            q_min[eval_joint_idx].item(), q_max[eval_joint_idx].item(), test_sample_num).to(self.device).unsqueeze(-1)
            q_sampled.requires_grad = True
            # 获得模型预测的距离和梯度my_eval_2
            pred_d, pred_grad = self.inference_d_wrt_q(cube_points,q_sampled,model,return_grad = True)
            print(f'pred_d_min:{pred_d.min()}, pred_d_max:{pred_d.max()}, pred_d_mean:{pred_d.mean()}')
            # 计算ground truth的距离和梯度
            gt_d = data_generator.distance_q(cube_points,q_sampled)
            print(f'gt_d.min:{gt_d.min()}, gt_d.max:{gt_d.max()},gt_d.mean:{gt_d.mean()}')
            # print(f'pred_d:{pred_d.shape}, gt_d:{gt_d.shape}')
            d_error = pred_d - gt_d
            # 把q_sampled, d_error, pred_grad, pred_d, gt_d reshape成(batch_size, DoF), 
            # 叠在一起，然后按照d_error的大小排序
            q_sampled = q_sampled.reshape(-1,DoF)
            d_error = d_error.reshape(-1,1)
            # print(f'pred_grad:{pred_grad.shape}, pred_d:{pred_d.shape}, gt_d:{gt_d.shape}')
            pred_grad = pred_grad.reshape(-1,DoF)
            pred_d = pred_d.reshape(-1,1)
            gt_d = gt_d.reshape(-1,1)
            
            sorted_idx = torch.linspace(0,gt_d.shape[0]-1,gt_d.shape[0],dtype=torch.long).to(self.device)
            q_sampled = q_sampled[sorted_idx]
            d_error = d_error[sorted_idx]
            pred_grad = pred_grad[sorted_idx]
            pred_grad_norm = torch.norm(pred_grad,dim=-1,keepdim=True)
            pred_d = pred_d[sorted_idx]
            gt_d = gt_d[sorted_idx]
            configuration = (sorted_idx * q_min[eval_joint_idx] + (1 - sorted_idx) * q_max[eval_joint_idx])/len(sorted_idx)
            configuration = configuration.cpu().detach().numpy()
            import matplotlib.pyplot as plt
            # import seaborn as sns
            # sns.set(style="whitegrid")
            plot_range = 1000
            plt.figure(figsize=(12, 8))
            plt.subplot(2, 2, 1)
            plt.scatter(configuration, pred_d[:plot_range].cpu().detach().numpy(), c='blue', label='Predicted')
            plt.scatter(configuration, gt_d[:plot_range].cpu().detach().numpy(), c='yellow', label='Ground Truth')
            plt.xlabel('configuration')
            plt.ylabel('Predicted Distance')
            # y轴范围从0开始
            plt.ylim(bottom=0)
            plt.title('Predicted vs Ground Truth Distance 1')
            plt.legend()
            plt.subplot(2, 2, 2)
            plt.scatter(configuration, pred_d[:plot_range].cpu().detach().numpy(), c='blue', label='Predicted')
            plt.scatter(configuration, gt_d[:plot_range].cpu().detach().numpy(), c='yellow', label='Ground Truth')
            plt.xlabel('configuration')
            plt.ylabel('Predicted Distance')
            plt.title('Predicted vs Ground Truth Distance 2')
            plt.legend()
            plt.subplot(2, 2, 3)
            plt.scatter(configuration, d_error[:plot_range].cpu().detach().numpy(), c='green', label='Ground Truth vs Error')
            plt.xlabel('configuration')
            plt.ylabel('Distance Error')
            plt.title('Ground Truth vs Distance Error')
            plt.legend()
            plt.subplot(2, 2, 4)
            # gradient 的散点图
            plt.scatter(configuration, pred_grad_norm[:plot_range].cpu().detach().numpy(), c='red', label='Predicted Gradient')
            plt.xlabel('configuration')
            plt.ylabel('Predicted Gradient')
            plt.title('Predicted Gradient')
            plt.legend()
            plt.tight_layout()
            # 在图上添加文字
            text = f'Joint {eval_joint_idx.item()} Evaluation'+\
                f'\nCube Position: {cube_pos}\nCube Edge: {cube_edge}\n' + \
                    f"q Sampled: {q_sampled[0].cpu().detach().numpy()}"
            plt.text(0.4, 0.95, text, horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes, fontsize=10)
            plt.savefig(os.path.join(CUR_PATH,f'slice_{cube_pos[0]}_{cube_pos[1]}_{cube_pos[2]}_{cube_edge}.png'))
            # plt.show()
            # 计算MAE和RMSE
            pred_d = pred_d.squeeze(-1).reshape(-1).cpu().detach().numpy()
            gt_d = gt_d.reshape(-1).cpu().detach().numpy()
            pred_grad = pred_grad.reshape(-1,DoF).cpu().detach().numpy()
            MAE_d = np.mean(np.abs(pred_d - gt_d))
            RMSE_d = np.sqrt(np.mean((pred_d - gt_d)**2))
            print(f'MAE_d: {MAE_d}, RMSE_d: {RMSE_d}')            
            
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Panda CDF Model Training and Evaluation')
    parser.add_argument('--data_path', type=str, default='data.pt', help='Path to the data file')
    parser.add_argument('--raw', type=str, default='data_finger_no_base.npy', help='Path to the raw data file')
    parser.add_argument('--with_writer', action='store_true', help='Whether to use TensorBoard writer')
    parser.add_argument('--eval', action='store_true', help='Whether to evaluate the model')
    parser.add_argument('--train', action='store_true', help='Whether to train the model')
    parser.add_argument('--epoches', type=int, default=50000, help='Number of training epochs')
    parser.add_argument('--batch_x', type=int, default=10, help='Batch size for x')
    parser.add_argument('--batch_q', type=int, default=100, help='Batch size for q')
    parser.add_argument('--signed_distance', action='store_true', help='Whether to use signed distance')
    parser.add_argument('--max_q_per_link', type=int, default=100, help='Maximum number of q samples per link')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for training/evaluation')
    parser.add_argument('--model_dict', type=str, default='finger_no_base.pt', help='Path to save/load the model dictionary')
    parser.add_argument('--robot', type=str, default='panda', help='Robot type (e.g., panda)',choices=['panda','dexhand','leaphand'])
    args = parser.parse_args()
    print(f'args:{args}')
    c = input('press enter to continue')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hprams = {
        'batch_x': args.batch_x,
        'batch_q': args.batch_q,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'signed_distance': args.signed_distance,
        'max_q_per_link': args.max_q_per_link,
        'epoches': args.epoches,
        'optimizer': 'Adam',
        'learning_rate': 0.001,
        'scheduler': 'ReduceLROnPlateau',
        'd_loss': 5.0,
        'eikonal_loss': 0.01,
        'tension_loss': 0.01,
        'gradient_loss': 0.1
    }
    CUR_DIR = os.path.dirname(os.path.abspath(__file__))
    paths = {
        'urdf': os.path.join(CUR_DIR,f'../../RDF/descriptions/{args.robot}/*.urdf'),
        'meshes': os.path.join(CUR_DIR,f'../../RDF/descriptions/{args.robot}/meshes/*.stl'),
        'points': os.path.join(CUR_DIR,f'../../RDF/data/{args.robot}/sdf_points/'),
        'model':os.path.join(CUR_DIR, f'../../RDF/models/{args.robot}/BP_8.pt'),
        'data': os.path.join(CUR_DIR,f'data/{args.robot}/{args.data_path}'),
        'model_dict': os.path.join(CUR_DIR,f'model_dict/{args.robot}/{args.model_dict}'),
        'raw_data': os.path.join(CUR_DIR,f'data/{args.robot}/{args.raw}'),
    }
    if args.with_writer:
        from torch.utils.tensorboard import SummaryWriter
        import os
        import time
        
        i = 0
        while os.path.exists(os.path.join(CUR_PATH,'runs/panda_cdf_'+str(i))):
            i += 1
        writer = SummaryWriter(os.path.join(CUR_PATH,'runs/panda_cdf_'+str(i)))
        print(f'writer path: {os.path.join(CUR_PATH,"runs/panda_cdf_"+str(i))}')
        
        writer.add_text('info', 'This is a test for Panda CDF model training and evaluation.')
        writer.add_hparams(hparam_dict=hprams, metric_dict={})
        cdf = CDF(device,paths=paths,robot=args.robot,writer=writer,signed_distance=args.signed_distance)
    else:
        cdf = CDF(device,paths=paths,robot=args.robot,writer=None,signed_distance=args.signed_distance)
    if args.train:
        cdf.train_nn(epoches=hprams['epoches'])
    if args.eval:
        model = MLPRegression(input_dims=10, output_dims=1, mlp_layers=[1024, 512, 256, 128, 128],skips=[], act_fn=torch.nn.ReLU, nerf=True)
        model.load_state_dict(torch.load((paths['model_dict']))[49900])
        model.to(device)
        cdf.my_eval_2(model)