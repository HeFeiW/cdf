# Last modified: 2026-03-03
# Construct a 2D CDF using a neural network, and compare the results with the groundtruth CDF calculated by online computation method. The model is trained on a single point, and the CDF is evaluated on a grid of points. The code also includes visualization of the CDF, groundtruth CDF, and their difference, as well as the gradient cosine similarity and L2 loss between the predicted gradient and groundtruth gradient.
import torch
import parser
import argparse
from mlp import MLPRegression
import os
from cdf import CDF2D
from tqdm import tqdm
from primitives2D_torch import Point_set
from cdf import CDF2D
import sys
sys.path.append('../../RDF')
from Siren import Siren

class Train_CDF:
    def __init__(self,device,model_type='siren')->None:
        self.device = device
        self.model_type = model_type

        self.cdf = CDF2D(device,model_type=self.model_type)

        self.q_template = self.cdf.q_grid_template.view(-1,200,2)

        x = torch.linspace(self.cdf.task_space[0][0],self.cdf.task_space[1][0],self.cdf.nbData).to(self.device)
        y = torch.linspace(self.cdf.task_space[0][1],self.cdf.task_space[1][1],self.cdf.nbData).to(self.device)
        xx,yy = torch.meshgrid(x,y)
        xx,yy = xx.reshape(-1,1),yy.reshape(-1,1)
        self.p = torch.cat([xx,yy],dim=-1).to(self.device)
        self.cdf = CDF2D(device)


    def matching_csdf(self,q):
        # q: [batchsize,2]
        # return d:[len(x),len(q)]
        dist = torch.norm(q.unsqueeze(1).expand(-1,200,-1) - self.q_template.unsqueeze(1),dim=-1)
        d,idx = torch.min(dist,dim=-1)
        q_template = torch.gather(self.q_template,1,idx.unsqueeze(-1).expand(-1,-1,2))
        return d,q_template



    def train(self,input_dim, hidden_dim, output_dim, activate, batch_size, learning_rate, weight_decay, save_path, device,
          epochs):

        # siren model
        print(len(hidden_dim)-1)
        if self.model_type == 'siren':
            net = Siren(in_features=input_dim, out_features=output_dim, hidden_features=hidden_dim[0], 
                        hidden_layers=len(hidden_dim)-1, outermost_linear=True).to(device)
        else:
            net = MLPRegression(input_dims=input_dim,
                                output_dims=output_dim, 
                                mlp_layers=hidden_dim,
                                skips=[],
                                act_fn=activate, 
                                nerf=True).to(device)
        # net.apply(model.init_weights)
        # debug: siren先把lr固定设为1e-4
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-4,
                                 weight_decay=weight_decay)
        folder = os.path.dirname(save_path)
        if not os.path.exists(folder):
            os.makedirs(folder)

        batch_p = self.p.unsqueeze(1).expand(-1,batch_size,-1).reshape(-1,2)
        # batch_p: [len(x)*batch_size,2]
        max_loss = float("inf")
        net.train()
        # debug epochs先设为5000
        epochs = 5000
        for i in tqdm(range(epochs)):
            q = torch.rand(batch_size,2,requires_grad=True).to(self.device)*2*torch.pi-torch.pi
            batch_q = q.unsqueeze(0).expand(len(self.p),-1,-1).reshape(-1,2)
            # batch_q: [len(x)*batch_size,2]
            d,q_temp = self.matching_csdf(q)
            
            q_temp = q_temp.reshape(-1,2)
            mask = d.reshape(-1)<torch.inf
            # mask = d<torch.inf
            inputs = torch.cat([batch_p,batch_q],dim=-1).reshape(-1,4)
            outputs = d.reshape(-1,1)
            # input: (len(x)*batch_size,4)
            # output: (len(x)*batch_size,1)
            inputs,outputs = inputs[mask],outputs[mask]
            q_temp = q_temp[mask]
            weights = torch.ones_like(outputs).to(device)
            # weights = (1/outputs).clamp(0,1)

            d_pred = net.forward(inputs)
            if self.model_type == 'siren':
                d_pred = d_pred[0]
            d_grad_pred = torch.autograd.grad(d_pred, batch_q, torch.ones_like(d_pred), retain_graph=True)[0]
            d_grad_pred = d_grad_pred[mask]

            # Compute the Eikonal loss
            eikonal_loss = torch.abs(d_grad_pred.norm(2, dim=-1) - 1).mean()

            # Compute the MSE loss
            d_loss = ((d_pred-outputs)**2*weights).mean()

            # Compute the projection loss
            proj_q = batch_q[mask] - d_grad_pred*d_pred
            proj_loss = torch.norm(proj_q-q_temp,dim=-1).mean()

            # Combine the two losses with appropriate weights
            w0 = 1.0
            w1 = 1.0
            w2 = 0.1
            # loss = w0 * d_loss + w1 * eikonal_loss + w2*proj_loss
            loss = w0 * d_loss + w1 * eikonal_loss
            print(f"Epoch {i+1}/{epochs}, Loss: {loss.item():.4f}, d_loss: {d_loss.item():.4f}, eikonal_loss: {eikonal_loss.item():.4f}, proj_loss: {proj_loss.item():.4f}")
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if loss.item() < max_loss:
                max_loss = loss.item()
                torch.save(net, os.path.join(save_path, args.model_path))

def inference(x,q,net):
    x_cat = x.unsqueeze(1).expand(-1,len(q),-1).reshape(-1,2)
    q_cat = q.unsqueeze(0).expand(len(x),-1,-1).reshape(-1,2)
    inputs = torch.cat([x_cat,q_cat],dim=-1)
    c_dist = net.forward(inputs)
    if isinstance(c_dist, tuple):
        c_dist = c_dist[0]
    c_dist = c_dist.squeeze()
    grad = torch.autograd.grad(c_dist, q_cat, torch.ones_like(c_dist), retain_graph=True)[0]
    return c_dist.squeeze(),grad


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="model22.pth", help="Path to the model directory")
    parser.add_argument("--model_type", type=str, default="siren", help="Type of model to use (e.g., siren, mlp)")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # check if model exists
    model_path = os.path.join('./model_dict', args.model_path)
    train_cdf = Train_CDF(device,args.model_type)
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}, starting training...")
        print(train_cdf.cdf.num_joints)
        train_cdf.train(input_dim=2+train_cdf.cdf.num_joints,
                hidden_dim=[256, 256, 128, 128, 128], 
                output_dim=1, 
                activate=torch.nn.ReLU, 
                batch_size=100,
                learning_rate=0.01, 
                weight_decay=1e-5, 
                save_path=model_path,
                device=device,
                epochs=1000)
    net = torch.load(model_path).to(device)
    net.eval()
    x = torch.tensor([[-1.0,1.0]],device=device)
    q = train_cdf.cdf.create_grid_torch(train_cdf.cdf.nbData).to(device)
    q.requires_grad = True
    q_proj = q.clone()
    for i in range (10):
        c_dist,grad = inference(x,q_proj,net)
        q_proj = train_cdf.cdf.projection(q_proj,c_dist,grad)
    # plot
    import matplotlib.pyplot as plt
    figure = plt.figure(figsize=(10,4))
    ax1,ax2,ax3 = figure.add_subplot(1,3,1),figure.add_subplot(1,3,2),figure.add_subplot(1,3,3)
    c_dist,grad = inference(x,q,net)
    cs = ax1.contourf(q[:,0].detach().cpu().numpy().reshape(50,50), q[:,1].detach().cpu().numpy().reshape(50,50), c_dist.cpu().detach().numpy().reshape(50,50), levels=20,cmap='coolwarm')
    # ax1.scatter(q_proj[:,0].detach().cpu().numpy(), q_proj[:,1].detach().cpu().numpy(), c='r', s=1)
    # 画出zero level set
    ax1.contour(q[:,0].detach().cpu().numpy().reshape(50,50), q[:,1].detach().cpu().numpy().reshape(50,50), c_dist.cpu().detach().numpy().reshape(50,50), levels=[0], colors='b')
    print(f'd statistics of inference cdf: min:{torch.min(c_dist).item()}, max:{torch.max(c_dist).item()}, mean:{torch.mean(c_dist).item()}')
    ax1.set_title('CDF')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    plt.colorbar(cs)
    
    # 对比groundtruth
    obj = Point_set(points=x)
    c_dist_gt, c_grad_gt = train_cdf.cdf.calculate_cdf(q=q, obj_lists=[obj], method='online_computation', return_grad=True)
    cs = ax2.contourf(q[:,0].detach().cpu().numpy().reshape(50,50), q[:,1].detach().cpu().numpy().reshape(50,50), c_dist_gt.cpu().detach().numpy().reshape(50,50), levels=20,cmap='coolwarm')
    # ax2.scatter(q_proj[:,0].detach().cpu().numpy(), q_proj[:,1].detach().cpu().numpy(), c='r', s=1)
    # 画出zero level set
    ax2.contour(q[:,0].detach().cpu().numpy().reshape(50,50), q[:,1].detach().cpu().numpy().reshape(50,50), c_dist_gt.cpu().detach().numpy().reshape(50,50), levels=[0], colors='b')
    print(f'd statistics of groundtruth cdf: min: {torch.min(c_dist_gt).item()}, max: {torch.max(c_dist_gt).item()}, mean: {torch.mean(c_dist_gt).item()}')
    ax2.set_title('Groundtruth CDF')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    plt.colorbar(cs)
    
    diff = torch.abs(c_dist_gt - c_dist)
    cs = ax3.contourf(q[:,0].detach().cpu().numpy().reshape(50,50), q[:,1].detach().cpu().numpy().reshape(50,50), diff.cpu().detach().numpy().reshape(50,50), levels=20,cmap='viridis')
    ax3.set_title('Difference between CDF and Groundtruth')
    ax3.set_xlabel('x')
    ax3.set_title('Difference between CDF and Groundtruth')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    plt.colorbar(cs)
    plt.savefig('nn_cdf_result.png')

    # 画出grad 余弦相似度误差图
    fig2 = plt.figure(figsize=(10,4))
    ax4 = fig2.add_subplot(1,2,1)
    cos = torch.nn.CosineSimilarity(dim=-1)
    cos_sim_loss = 1 - cos(grad, c_grad_gt)
    cs = ax4.contourf(q[:,0].detach().cpu().numpy().reshape(50,50), q[:,1].detach().cpu().numpy().reshape(50,50), cos_sim_loss.cpu().detach().numpy().reshape(50,50), levels=20,cmap='viridis')
    ax4.set_title('Cosine Similarity between CDF and GT Gradient')
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    plt.colorbar(cs)
    print(f'Gradient cosine similarity statistics: min: {torch.min(cos_sim_loss).item()}, max: {torch.max(cos_sim_loss).item()}, mean: {torch.mean(cos_sim_loss).item()}')
    plt.savefig('nn_cdf_grad_result.png')
    
    # 画出grad eikonal误差图
    ax5 = fig2.add_subplot(1,2,2)
    l2_loss = torch.abs(torch.norm(grad, dim=-1)-1)
    cs2 = ax5.contourf(q[:,0].detach().cpu().numpy().reshape(50,50), q[:,1].detach().cpu().numpy().reshape(50,50), l2_loss.cpu().detach().numpy().reshape(50,50), levels=20,cmap='rainbow')
    ax5.set_title('L2 Loss between CDF and GT Gradient')
    ax5.set_xlabel('x')
    ax5.set_ylabel('y')
    plt.colorbar(cs2)
    print(f'Gradient L2 loss statistics: min: {torch.min(l2_loss).item()}, max: {torch.max(l2_loss).item()}, mean: {torch.mean(l2_loss).item()}')
    plt.savefig('nn_cdf_grad_result.png')