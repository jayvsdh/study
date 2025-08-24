# 扩散模型

1. 正向扩散（加噪声）

   从$x_0$ 到$x_T$其中这个T表示的步长，也就是这一共进行了多少步

   开始是一张干净的图片，一步步的加入噪声，最终会是一张纯噪声图片

   那么这个噪声是怎么加入的，大概就是一个高斯分布的噪声吧。会有一个$\beta$从0.0001到0.002，表示的是噪声比例，这会是已知的。

   $\alpha = 1 - \beta$，这个$\alpha$表示的原图保存的比例，这也是已知的

   最终的图像：$x_t = \sqrt{\alpha _t}x_{t-1} + \sqrt{\beta_t}z_t$ 

   $x_{t-1} = \sqrt{\alpha_{t-1}}x_{t-2}+\sqrt{\beta_{t-1}z_{t-1}}$

   依次类推，最终可以得出$x_t = \sqrt{\overline{\alpha_t}}x_0+\sqrt{1-\overline{\alpha_t}}z$
   这个公式有啥作用呢？

2. 反向过程
  目的是让模型学会如何从当前步恢复到前面一步，也就是做噪声预测
  公式：$p_\theta(x_{t-1}|x_t) =N(x_{t-1};\hat\mu_\theta(x_t,t),\hat\sigma_\theta(t))$ 
  模型通过以下公式去除噪音：$x_{t-1}=\hat\mu_\theta(x_t,t)+\hat\sigma_\theta(t)z$  其中：
  	$x_t$表示当前t步的噪音图像
  	$\hat\mu_\theta$ is the  result of the model
  	$\hat\sigma_\theta(t)$ is the noise level of the model predict

3. 一个非常重要的公式

  <img src="/Users/ding/Library/Application Support/typora-user-images/image-20250823235543188.png" alt="image-20250823235543188" style="zoom:40%;" />

4. trainning process
  ```python
  # 训练循环
  num_epochs = 10
  for epoch in range(num_epochs):
   optimizer.zero_grad()
  
   # 1. 随机生成一个时间步数
   t = torch.randint(0, timesteps, (batch_size,))  # 每个样本对应一个随机的时间步
  
   # 2. 前向扩散过程：加噪声
   x_t = forward_diffusion(x0, timesteps, beta_schedule)
  
   # 3. 生成噪声（真实噪声）
   noise = torch.randn_like(x0)
  
   # 4. 使用模型预测噪声
   predicted_noise = model(x_t)
  
   # 5. 计算损失：均方误差
   loss = loss_fn(predicted_noise, noise)
  
   # 6. 反向传播和优化
   loss.backward()
   optimizer.step()
  
   if epoch % 10 == 0:
       print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}"
  ```

4.源码

```python
class Diffusion:
  def __init__(self,noise_steps = 1000,beta_start = 1e-4,beta_end = 0.02,img_size = 256,device = "cuda"):
    # 自带的参数
    self.noise_step = noise_steps
    self.beta_start = beta_start
    self.beta_end = beta_end
    self.img_size = img_size
    self.device = device
    
    self.beta = self.prepare_noise_schedule().to(device)
    self.alpha = 1. - self.beta
    self.alpha_hat = torch.comprod(self.alpha,dim = 0)
    
    
  def prepare_noise_schedule(self):
    return torch.linespace(self.beta_start,self.beta_end,self.noise_steps)
  
  def noise_images(self,x,t):
    sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:,None,None,None]
    sqrt_one_minus_alpha_hat = torch.sqrt(1-self.alpha_hat[t][:,None,None,None])
    epsilon = torch.randn_like(x)
    return sqrt_alpha_hat*x + sqrt_one_minus_alpha_hat * epsilon,epsilon
  
  def sample_timesteps(self,n):
    return torch.randint(low = 1,hign = self.noise_steps,size = (n,))
  
  def sample(self,model,n):
    logging.info(f"Sampling {n} new images....")
    model.eval()
    with torch.no_grad():
      x = torch.randn((n,3,self.img_size,self.img_size)).to(self.device) # 生成初始噪声
      for i in tqdm(reversed(range(1,self.noise_steps),position = 0)):# 逆向去噪循环
        # 去噪计算
        	t = (torch.ones(n)*i).long().to(self.device)
          predicted_noise = model(x,t)
          alpha = self.alpha[t][:,None,None,None]
          alpha_hat = self.alpha_hat[t][:,None,None,None]
          if i>1 :
            noise = torch.randn_like(x)
          else:
            noise = torch.zero_like(x)
            x = 1/torch.sqrt(alpha)*(x-((1-alpha)/(torch.sqrt(1-alpha_hat)))*predicted_noise)+ torch.sqrt(beta)*noise
            # 值的后处理与返回
            model.train()
            x = (x.clamp(-1,1)+1)/2
            x = (x*255).type(torch.uint8)
            return x
      
      
```

