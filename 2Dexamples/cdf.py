# -----------------------------------------------------------------------------
# SPDX-License-Identifier: MIT
# This file is part of the CDF project.
# Copyright (c) 2024 Idiap Research Institute <contact@idiap.ch>
# Contributor: Yiming Li <yiming.li@idiap.ch>
# -----------------------------------------------------------------------------

# 2D example for configuration space distance field
import numpy as np
import os
import sys
import torch
import math
import matplotlib.pyplot as plt
from robot2D_torch import Robot2D
from primitives2D_torch import Circle
from torchmin import minimize
import time
import math
import robot_plot2D
import copy
import matplotlib.gridspec as gridspec

PI = math.pi
CUR_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(CUR_PATH,'data2D_tree.npy')

class CDF2D:
    def __init__(self,device) -> None:
        self.device = device    
        self.nbData =  50
        self.nbDiscretization = 50
        
        self.link_length = torch.tensor([[2, 2, 1, 1]]).float().to(device)
        self.link_parent_map = {
            1: 0,  # Link 1 connects to the base
            2: 1,  # Link 2 connects to Link 1
            3: 2,  # Link 3 connects to Link 2
            4: 2   # Link 4 connects to Link 2
        }
        self.num_joints = self.link_length.size(1)
        self.q_max = torch.tensor([PI]).expand(self.num_joints).to(device)
        self.q_min = torch.tensor([-PI]).expand(self.num_joints).to(device)
        self.Q_grid = self.create_grid_torch(self.nbData).to(device)

        # data generation
        self.task_space = [[-4.0,-4.0],[4.0,4.0]]
        self.batchsize = 40000       # batch size of q
        self.epsilon = 1e-3         # distance threshold to filter data

        # robot
        # self.robot = Robot2D(num_joints=self.num_joints ,init_states = self.Q_grid,link_length=self.link_length,device = device)
        self.robot = Robot2D(num_links=len(self.link_length[0]),init_states = self.Q_grid,link_lengths=self.link_length,device = device, link_parent_map=self.link_parent_map)
        # c space distance field
        save_path = DATA_PATH.replace('.npy','.pt')
        if not os.path.exists(save_path):
            self.generate_data()
        self.q_grid_template =  torch.load(save_path)

    # def create_grid(self,nb_data):
    #     # 创建一个网格，范围从-π到π
    #     # 数据格式：
    #     # q0:nb_data x nb_data 个点的横坐标
    #     # q1:nb_data x nb_data 个点的纵坐标
    #     t = np.linspace(-math.pi,math.pi, nb_data)
    #     self.q0,self.q1 = np.meshgrid(t,t)
    #     return self.q0,self.q1
    def create_grid_dof(self,nb_data,dof):
        # 把create_grid的numpy版本改成dof维度
        q_list = []
        t = np.linspace(self.q_min.cpu().numpy(),self.q_max.cpu().numpy(), nb_data)
        for i in range(dof):
            q_list.append(t[i])
        mesh = np.meshgrid(*q_list)
        Q_sets = np.stack(mesh,axis=-1).reshape(-1,dof)
        return Q_sets
    
    def create_grid_torch(self,nb_data):
        # 把create_grid的numpy版本改成torch版本
        return torch.from_numpy(self.create_grid_dof(nb_data,self.num_joints)).float()

    def inference_sdf(self,q,obj_lists,return_grad = False): 
        # 返回每个输入的q到所有障碍物总体的最小signed distance
        # sdf: (B,)/(B,2) (取决于是否return_grad)
        # using predefined object 
        # 先用正运动学计算q参数下机器人形态（然后在上面采样很多点为kpts）
        kpts = self.robot.surface_points_sampler(q)
        B,N = kpts.size(0),kpts.size(1)
        # 计算每个物体到每个采样点的signed distance
        dist = torch.cat([obj.signed_distance(kpts.reshape(-1,2)).reshape(B,N,-1) for obj in obj_lists],dim=-1)
        # print('shape of dist: ',dist.shape)
        # using closest point from robot surface
        sdf = torch.min(dist,dim=-1)[0]
        sdf = sdf.min(dim=-1)[0]
        if return_grad: 
            grad = torch.autograd.grad(sdf,q,torch.ones_like(sdf))[0]
            return sdf,grad
        return sdf
    
    def find_q(self,obj_lists,batchsize = None):
        # find q that makes d(x,q) = 0. x is the obstacle surface
        # using L-BFGS method
        # 用L-BFGS方法寻找满足d(x,q)=0的q, 
        # 即寻找障碍物表面的zero-level-set configuration space
        if not batchsize:
            batchsize = self.batchsize
            
        def cost_function(q):
            # 定义代价为所有配置点到障碍物表面的最短距离的平方和
            #  find q that d(x,q) = 0
            # q : B,2

            d = self.inference_sdf(q,obj_lists)
            cost = torch.sum(d**2)
            return cost
        
        t0 = time.time()
        # optimizer for data generation
        dof = self.q_min.size(0)
        q = torch.rand(batchsize,dof).to(self.device)*(self.q_max-self.q_min)+self.q_min
        q0 =copy.deepcopy(q)
        res = minimize(
            cost_function, 
            q, # 随机初始化q
            method='l-bfgs', 
            options=dict(line_search='strong-wolfe'),
            max_iter=50,# 限制了优化的最大迭代次数为 50
            disp=0
            )
        # res 是优化结果，包含以下信息：
            # res.x：优化后的配置空间点。
            # res.success：优化是否成功。
            # res.fun：优化后的目标函数值。
        d = self.inference_sdf(q,obj_lists).squeeze()
        # 筛选出近似满足d(x,q) = 0的q
        # q不是随机采样出来的吗？为什么要用q算的d来计算mask呢？？？#TODO
        mask = torch.abs(d) < 0.05
        q_valid,d = res.x[mask],d[mask]
        boundary_mask = ((q_valid > self.q_min) & (q_valid < self.q_max)).all(dim=1)
        final_q = q_valid[boundary_mask]
        q0 = q0[mask][boundary_mask]
        # print('number of q_valid: \t{} \t time cost:{}'.format(len(final_q),time.time()-t0))
        # print('shape of q0: {}, shape of final_q: {}'.format(q0.shape,final_q.shape))
        return q0,final_q,res.x
    
    def generate_data(self,nbDiscretization=50):
        # generate data for CDF
        # 以nbDiscretization为一维上的网格采样数，
        # 在task_space内均匀采样nbDiscretization*nbDiscretization个点
        # 对每个采样点，计算其对应的q（很大计算量！），
        # 并将其存储在DATA_PATH中。
        # 格式：字典{
        #     'p': 采样点坐标，
        #     'q': 对应的q坐标
        # }
        x = torch.linspace(self.task_space[0][0],self.task_space[1][0],self.nbDiscretization).to(self.device)
        y = torch.linspace(self.task_space[0][1],self.task_space[1][1],self.nbDiscretization).to(self.device)
        xx,yy = torch.meshgrid(x,y)
        xx,yy = xx.reshape(-1,1),yy.reshape(-1,1)
        p = torch.cat([xx,yy],dim=-1).to(self.device)

        data = {}
        for i,_p in enumerate(p):
            grids = [Circle(center=_p,radius=0.001,device=device)]
            q = self.find_q(grids)[1]
            data[i] = {# 1个grid对应1个p，多个q
                'p':_p,
                'q':q
            }
            if q.shape[0] == 0:
                print('no q found for p: {}'.format(_p))
            else:
                print('i: {} \t number of q: {}'.format(i,len(q)))

        np.save(DATA_PATH,data)
        data = np.load(DATA_PATH,allow_pickle=True).item()
        max_q_per_x = 200
        tensor_data = torch.inf*torch.ones(self.nbData,self.nbData,max_q_per_x,self.robot.num_links).to(self.device)
        for idx in data.keys():
            p = data[idx]['p']
            q = data[idx]['q']
            i = idx/50
            j = idx%50
            if len(q) > max_q_per_x:
                q = q[:max_q_per_x]
            tensor_data[int(i),int(j),:len(q),:] = q
        save_path = DATA_PATH.replace('.npy','.pt')
        torch.save(tensor_data,save_path)
        return tensor_data

    def calculate_cdf(self,q,obj_lists,method='online_computation',return_grad = False):
        # x : (Nx,2)
        # q : (Np,2)
        # return d : (Np) distance between q and x in C space. d = min_{q*}{L2(q-q*)}. sdf(x,q*)=0
        Np = q.shape[0]
        if method == None:
            method = 'online_computation'
        if method == 'offline_grid':
            if not hasattr(self,'q_list_template'):
                # 在obj_lists中采样障碍物表面点
                obj_points = torch.cat([obj.sample_surface(200) for obj in obj_lists])
                # 每个表面点找到对应的网格索引
                grid = self.x_to_grid(obj_points)   
                print('shape of grid: ',grid.shape)
                # 从q_grid_template中获取grid对应的q点集合   
                q_list_template = (self.q_grid_template[grid[:,0],grid[:,1],:,:]).reshape(-1,2)
                self.q_list_template = q_list_template[q_list_template[:,0] != torch.inf]
            # 计算q和q_list_template之间的距离矩阵
            dist = torch.norm(q.unsqueeze(1) - self.q_list_template.to(self.device).unsqueeze(0),dim=-1)
            print('shape of dist: ',dist.shape)
        if method == 'online_computation':
            if not hasattr(self,'q_0_level_set'):
                self.q_0_level_set = self.find_q(obj_lists)[1]
            print('shape of q_0_level_set: ',self.q_0_level_set.shape)
            dist = torch.norm(q.unsqueeze(1) - self.q_0_level_set.unsqueeze(0),dim=-1)
        # 取每个q点到所有障碍物表面点的最小距离
        d = torch.min(dist,dim=-1)[0]
        # compute sign of d, based on the sdf
        # exit()
        d_ts = self.inference_sdf(q,obj_lists)

        mask =  (d_ts < 0)
        d[mask] = -d[mask]
        if return_grad:
            grad = torch.autograd.grad(d,q,torch.ones_like(d))[0]
            return d,grad
        return d 
    
    def x_to_grid(self,p):
        # p: (N,2)
        # return grid index (N,2)
        x_workspace = torch.tensor([self.task_space[0][0],self.task_space[1][0]]).to(self.device)
        y_workspace = torch.tensor([self.task_space[0][1],self.task_space[1][1]]).to(self.device)

        x_grid = (p[:,0]-x_workspace[0])/(x_workspace[1]-x_workspace[0])*self.nbDiscretization
        y_grid = (p[:,1]-y_workspace[0])/(y_workspace[1]-y_workspace[0])*self.nbDiscretization

        x_grid.clamp_(0,self.nbDiscretization-1)
        y_grid.clamp_(0,self.nbDiscretization-1)
        return torch.stack([x_grid,y_grid],dim=-1).long()
    
    def projection(self,q,d,grad):
        # q : (N,2)
        # d : (N)
        # grad : (N,2)
        # return q_proj : (N,2)
        q_proj = q - grad*d.unsqueeze(-1)
        return q_proj
    
    def plot_projection(self,ax):
        q = torch.rand(1000,2).to(self.device)*2*math.pi-math.pi
        q0 = copy.deepcopy(q)
        d,grad = self.inference_c_space_sdf_using_data(q,sign=False)
        q = self.projection(q,d,grad)
        q,grad= q.detach(),grad.detach()   # release memory
        ax.set_title(f'{iter} iteractions', size=25)  # Add a title to your plot
        ax.plot(q[:,0].detach().cpu().numpy(),q[:,1].detach().cpu().numpy(),'.',color='lightgreen')
        return q0,q

    def plot_sdf(self,obj_lists, ax):
        # 初始化图像
        ax.set_aspect('equal', 'box')  # Make sure the pixels are square
        ax.set_title('Configuration space', size=30)  # Add a title to your plot
        ax.set_xlabel('q1', size=20)
        ax.set_ylabel('q2', size=20)
        axis_limits = (-PI, PI)  # Set the limits for both axes to be the same
        ax.set_xlim(axis_limits)
        ax.set_ylim(axis_limits)
        ax.tick_params(axis='both', labelsize=20)
        # 计算 SDF
        sdf = self.inference_sdf(self.Q_grid,obj_lists)
        sdf = sdf.detach().cpu().numpy()
        # 绘制等高线
        ax.contour(self.q0, self.q1, sdf.reshape(self.nbData, self.nbData), levels=[0], linewidths=6, colors='black', alpha=1.0)
        ct = ax.contourf(self.q0, self.q1, sdf.reshape(self.nbData, self.nbData), levels=6, linewidths=1, cmap='coolwarm')
        ax.clabel(ct, inline=False, fontsize=15, colors='black', fmt='%.1f')

        # fig = plt.gcf()  # Get the current figure
        # fig.set_size_inches(10, 8)  # Set the figure size
        # fig.colorbar(ct, ax=ax)  # Add a colorbar to your plot
        # # 保存 plt 图像到本地
        # fig.savefig(os.path.join(CUR_PATH,'sdf.png'), dpi=300, bbox_inches='tight')

    def shooting(self,q0,obj_lists,dt = 1e-2,timestep = 500,method = 'SDF'):
        # 沿着测地线移动（即正交于梯度方向）
        # q0: (N,2) 初始配置点
        # obj_lists: list of objects in the scene
        # dt: time step
        # timestep: number of time steps
        # method: 'SDF' or 'CDField'
        q = q0
        q.requires_grad = True
        q_list = []
        for t in range(timestep):
            if method == 'SDF':
                d,g = self.inference_sdf(q,obj_lists, return_grad=True)
            if method == 'CDField':
                d,g = self.calculate_cdf(q,obj_lists,return_grad=True)
            g = torch.nn.functional.normalize(g,dim=-1)
            g_orth = torch.stack([g[:,1],-g[:,0]],dim=-1)# 对二维向量g进行正交化（两个分量互换并取负号）
            # if g_orth[:,1] < 0:
            #     g_orth = -g_orth
            q = q + dt*g_orth
            q_list.append(q.detach().cpu().numpy())
        return np.array(q_list).transpose(1,0,2)
    
    def shooting_proj(self,q0,obj_lists,dt = 1e-2,timestep = 500,method = 'SDF'):
        q = q0
        q.requires_grad = True
        q_list = []
        if method == 'SDF':
            for t in range(timestep):
                q_list.append(q.detach().cpu().numpy())
                d,g = self.inference_sdf(q,obj_lists,return_grad=True)
                # if method == 'CDField':
                #     d,g = self.calculate_cdf(q,obj_lists,return_grad=True)
                g = torch.nn.functional.normalize(g,dim=-1)
                # if g_orth[:,1] < 0:
                #     g_orth = -g_orth
                q = q - dt*g*d.unsqueeze(-1)
                # q_list.append(q.detach().cpu().numpy())
        if method == 'CDField':
            d,g = self.calculate_cdf(q,obj_lists,return_grad=True)
            q = q - g*d.unsqueeze(-1)
            q_list = np.linspace(q0.detach().cpu().numpy(),q.detach().cpu().numpy(),timestep)
        return np.array(q_list).transpose(1,0,2)

    def plot_cdf(self,ax,obj_lists,method='online_computation'):
        d = self.calculate_cdf(self.Q_grid,obj_lists,method).detach().cpu().numpy()
        print('shape of d:{}'.format(d.shape))
        ax.set_aspect('equal', 'box')  # Make sure the pixels are square
        ax.set_title('Configuration space', size=30)  # Add a title to your plot
        ax.set_xlabel('q1', size=20)
        ax.set_ylabel('q2', size=20)
        axis_limits = (-PI, PI)  # Set the limits for both axes to be the same
        ax.set_xlim(axis_limits)
        ax.set_ylim(axis_limits)
        ax.tick_params(axis='both', labelsize=20)

        ax.contour(self.q0, self.q1, d.reshape(self.nbData, self.nbData), levels=[0], linewidths=6, colors='black', alpha=1.0)
        ct = ax.contourf(self.q0, self.q1, d.reshape(self.nbData, self.nbData), levels=8, linewidths=1, cmap='coolwarm')
        ax.clabel(ct, inline=False, fontsize=15, colors='black', fmt='%.1f')


    def plot_objects(self,ax,obj_lists):
        for obj in obj_lists:
            # plt.gca().add_patch(obj.create_patch())
            ax.add_patch(obj.create_patch())
        return ax
    
    def compare_with_lbfgs(self):
        q0,q,_ = self.find_q()
        return q0,q
    
def plot_fig1(obj_lists):
    #obj_lists: 场景中的障碍物列表
    color_list = ['magenta','orange', 'cyan', 'green',  'purple', 'navy']
    fig1, ax1 = plt.subplots(figsize=(10,8))  # Create the first plot
    fig2, ax2 = plt.subplots(figsize=(10, 8))  # Create the third plot

    fig = plt.figure(figsize=(25,20))  # Create a figure
    gs = gridspec.GridSpec(2, 6)  # Create a gridspec
    xlim=(-4.0,4.0)
    ylim=(-4.0,4.0)
    axs = []
    for i in range(12):
        ax = fig.add_subplot(gs[i // 6, i % 6])  # Add a subplot
        ax.axis('off')  # Turn off the axis
        cdf.plot_objects(ax,obj_lists)
        ax.set_aspect('equal', 'box')  # Make sure the pixels are square
        ax.set_xlim(xlim)  # Set the x limits
        ax.set_ylim(ylim)  # Set the y limits
        axs.append(ax)

    # Plot sdf on the first subplot
    import matplotlib.cm as cm
    cdf.plot_sdf(ax=ax1,obj_lists=obj_lists)
    # plot shooting
    shooting_q0 = torch.tensor([[-0.2,0.0],[1.2,1.0],[-1.0,0.5]]).to(device)
    projection_q0 = torch.tensor([[0.0,-1.5],[2.0,2.0],[-1.5,-2.0]]).to(device)
    shooting_tangent = cdf.shooting(shooting_q0,obj_lists,method='SDF')
    shooting_gradient = cdf.shooting_proj(projection_q0,obj_lists,method='SDF')
    shooting_q_sdf = np.concatenate([shooting_gradient,shooting_tangent],axis=0)
    print('shape of shooting_q_sdf: ',shooting_q_sdf.shape)

    for c,shoot in enumerate(shooting_q_sdf):
        ax1.plot(shoot[:,0],shoot[:,1],'r--',color = color_list[c],linewidth=3)
        ax1.plot(shoot[0,0],shoot[0,1],'*',color = color_list[c],markersize=10)
        # plot robot
        robot_plot2D.plot_2d_manipulators(joint_angles_batch=shoot[0:500:20],ax = axs[c*2],color = color_list[c],show_start_end=False,show_eef_traj=True)
    cdf.plot_cdf(ax2,obj_lists)

    shooting_tangent = cdf.shooting(shooting_q0,obj_lists,method='CDField')
    shooting_gradient = cdf.shooting_proj(projection_q0,obj_lists,method='CDField')
    shooting_q_cdf = np.concatenate([shooting_gradient,shooting_tangent],axis=0)    
    for c,shoot in enumerate(shooting_q_cdf):
        ax2.plot(shoot[:,0],shoot[:,1],'r--',color = color_list[c],linewidth=3)
        ax2.plot(shoot[0,0],shoot[0,1],'*',color = color_list[c],markersize=10)
        if c<3:
            # gradient projection
            robot_plot2D.plot_2d_manipulators(joint_angles_batch=shoot,ax = axs[c*2+1],color = color_list[c],show_start_end=True,show_eef_traj=True) 
        else:
            # geodesic shooting
            robot_plot2D.plot_2d_manipulators(joint_angles_batch=shoot[0:500:20],ax = axs[c*2+1],color = color_list[c],show_start_end=False,show_eef_traj=True) 
    # 保存plt图像到本地
    fig1.suptitle('Configuration Space Distance Field', size=30)
    ax1.set_title('SDF + Gradient Shooting', size=25)  # Add a title to your plot
    ax2.set_title('CDF + Gradient Shooting', size=25)  # Add a title to your plot
    fig1.tight_layout()
    fig2.tight_layout()
    fig.tight_layout()
    fig1.savefig(os.path.join(CUR_PATH,'fig1_sdf.png'), dpi=300, bbox_inches='tight')
    fig2.savefig(os.path.join(CUR_PATH,'fig1_cdf.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(CUR_PATH,'fig1_shooting.png'), dpi=300, bbox_inches='tight')

def plot_projection(obj_lists):
    fig1,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(24,8))

    # plot cdf
    cdf.plot_cdf(ax1,obj_lists)
    cdf.plot_cdf(ax2,obj_lists)
    q_random = torch.rand(1000,2,requires_grad=True).to(device)*2*math.pi-math.pi
    
    # plot CDF
    d,grad = cdf.calculate_cdf(q_random,obj_lists,return_grad=True)
    q_proj = cdf.projection(q_random,d,grad)
    ax1.plot(q_random[:,0].detach().cpu().numpy(),q_random[:,1].detach().cpu().numpy(),'.',color='lightgreen')
    ax2.plot(q_proj[:,0].detach().cpu().numpy(),q_proj[:,1].detach().cpu().numpy(),'.',color='lightgreen')
    ax1.set_title('Initial Samples', size=25)  # Add a title to your plot
    ax2.set_title('CDF + Gradient Projection', size=25)  # Add a title to your plot

    # plot SDF for comparison
    cdf.plot_sdf(obj_lists,ax3)
    q0,q,q_all = cdf.find_q(obj_lists,batchsize=1000)
    ax3.set_title(f'SDF+Optimization', size=25)  # Add a title to your plot
    ax3.plot(q_all[:,0].detach().cpu().numpy(),q_all[:,1].detach().cpu().numpy(),'.',color='lightgreen')

    # plot robot manipulator in task space for CDF
    fig, ax = plt.subplots(figsize=(8, 8))  # Create a figure 
    q_proj_np = q_proj.detach().cpu().numpy()
    for i, _q in enumerate(q_proj_np):
        robot_plot2D.plotArm(
                ax=ax,
                a=_q,
                d=cdf.link_length[0].cpu().numpy(),
                p=np.array([0.0, 0.0]),  # base position
                sz=0.05,
                label="via",
                alpha=0.05,
                zorder=2,
                xlim=None,
                ylim=None,
                robot_base=True,  # to visualize the base
                color='lightgreen'  # Set the color of the robot
            )
    cdf.plot_objects(ax,obj_lists) 
    ax.set_title('Robot Manipulator in Task Space', size=25)  # Add a title to your plot
    ax.set_xlabel('x', size=20)
    ax.set_ylabel('y', size=20)
    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-3.0, 3.0)
    ax.set_aspect('equal', 'box')  # Make sure the pixels are square
    ax.tick_params(axis='both', labelsize=20)
    fig.tight_layout()
    fig1.tight_layout()
    fig.savefig(os.path.join(CUR_PATH,'fig2_projection.png'), dpi=300,
                    bbox_inches='tight')
    fig1.savefig(os.path.join(CUR_PATH,'fig2_cdf.png'), dpi=300,
                    bbox_inches='tight')
    

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    cdf = CDF2D(device)

    scene_1 = [Circle(center=torch.tensor([2.5,2.5]),radius=0.5,device=device)]
    scene_2 = [Circle(center=torch.tensor([2.3,-2.3]),radius=0.3,device=device),
                        Circle(center=torch.tensor([0.0,2.45]),radius=0.3,device=device),
                        ]
    scene_3 = [Circle(center=torch.tensor([2.5,0.0]),radius=0.01,device=device),
                        Circle(center=torch.tensor([0.0, 1.0]),radius=0.01,attract=True,device=device)]
    # a,b,c= cdf.find_q(scene_1,2)
    # print('number of q in scene_1: ',len(b))
    plt.figure(figsize=(10,8))
    ax = plt.gca()
    cdf.plot_sdf(ax=ax,obj_lists=scene_3)
    plt.show()
    # # # plot the figure in the paper
    plot_fig1(scene_3)

    # # # plot gradient projection
    plot_projection(scene_3)
