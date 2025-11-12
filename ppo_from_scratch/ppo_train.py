from transformers import AutoModelForCausalLM, AutoModel, AutoModelForSequenceClassification, AutoTokenizer
from dataclasses import dataclass
from typing import Optional, Union, Tuple
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

# 构建dataset
class PromptDataset(Dataset):
    def __init__(self, prompts, tokenizer, apply_chat_template=False):
        self.prompts = prompts
        self.tokenizer = tokenizer
        
        self.final_prompts = []
        
        for prompt in prompts:
            if apply_chat_template:
                content = [{"role": "user", "content": prompt}]
                prompt = self.tokenizer.apply_chat_template(content, tokenize=False, add_generation_prompt=True)
            else:
                prompt = self.tokenizer.bos_token + prompt
                
            self.final_prompts.append(prompt)
        
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, index):
        return self.final_prompts[index]

# 价值（评论家）模型，用于预测每一步（生成token）的动作产生的收益，使用演员模型进行初始化，并外加一个回归头，输出shape为：(batch_size, seq_len， 1)
class Critic(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.base_model.eval()
        self.value_head = nn.Linear(base_model.config.hidden_size, 1)
        
    def forward(self, input_ids, attention_mask, num_actions):
        
        
        hidden_state = self.base_model(input_ids, attention_mask=attention_mask).last_hidden_state # (batch_size, seq_len, hidden_size)
        value_model_output = self.value_head(hidden_state) #(batch_size, seq_len， 1)
        values = value_model_output.squeeze(-1)[:, -num_actions:] # (batch_size, seq_len)
        return values



def compute_policy_loss(log_probs, old_log_probs, advantages, action_mask=None, clip_eps=0.2):
    ratio = (log_probs - old_log_probs).exp()
    surr1 = ratio * advantages
    surr2 = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * advantages
    loss = -torch.min(surr1, surr2)
    if action_mask is None:
        return loss.mean(-1).mean()
    return ((loss * action_mask).sum(-1) / action_mask.sum(-1)).mean()


def compute_value_loss(values, old_values, returns, action_mask=None, clip_eps: float = None):
    if clip_eps is not None:
        values_clipped = old_values + (values - old_values).clamp(-clip_eps, clip_eps)
        surr1 = (values_clipped - returns) ** 2
        surr2 = (values - returns) ** 2
        loss = torch.max(surr1, surr2)
    else:
        loss = (values - returns) ** 2
        
    if action_mask is None:
        return loss.mean(-1).mean()
    return ((loss * action_mask).sum(-1) / action_mask.sum(-1)).mean()


class ExperienceBuffer:
    def __init__(self, limit):
        self.limit = limit
        self.buffer = []
    
    def append(self, experiences):
        batch = [{} for _ in range(len(experiences))]
        keys = (
        "seqs",
        "action_log_probs",
        "values",
        "returns",
        "advantages",
        "attention_mask",
        "action_mask",
        "num_actions"
    )
        for key in keys:
            for i, experience in enumerate(experiences):
                value = getattr(experience, key)
                batch[i][key] = value
          
        self.buffer.extend(batch)
        if len(self.buffer) >= self.limit:
            self.buffer = self.buffer[len(self.buffer)-self.limit:]
        
    def get_batches(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def clear(self):
        self.buffer = []
        
    def __len__(self):
        return len(self.buffer)
    
    def __getitem__(self, index):
        return self.buffer[index]
    

@dataclass
class Samples:
    seqs: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    num_actions: Union[int, torch.Tensor]
    packed_seq_lens: Optional[torch.Tensor]
    response_length: torch.Tensor
    total_length: torch.Tensor

@dataclass
class Experience:

    seqs: torch.Tensor
    action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: Optional[torch.Tensor]
    advantages: Optional[torch.Tensor]
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    reward: torch.Tensor
    response_length: torch.Tensor
    total_length: torch.Tensor
    num_actions: Union[int, torch.Tensor]
    kl: Optional[torch.Tensor] = None

def compute_approx_kl(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    action_mask: Optional[torch.Tensor] = None,
):
    #	•	log_probs: (B, T) —— 来自 actor 的逐 token 对数概率
	#•	ref_log_probs: (B, T) —— 来自参考模型的逐 token 对数概率
	#•	action_mask（可选）: (B, T)，或任何可与前两者在最后一维广播对齐的形状（如 (B, T, 1)）
    log_ratio = log_probs.float() - ref_log_probs.float()
    if action_mask is not None:
        log_ratio = log_ratio * action_mask

    return log_ratio #(B, T)

# A(t) = R(t) + gam*V(t+1) - V(t)
# gae:A(t) = R(t) + gam*V(t+1) - V(t) + gam*lam*A(t+1)
# 最后一个时刻的未来优势和未来收益为0：A(T+1) = 0, V(T+1) = 0,  则A(T) = R(T) - V(T), 得出A(T)
# A(T-1) = R(T-1) + gam*V(T) - V(T-1) + gam*lam*A(T) 知道A(T)可计算A(T-1) 依次类推
# returns(t) = A(t) + V(t) = = R(t) + gam * (V(t+1) + lam * A(t+1))
def get_advantages_and_returns(
        values: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
        lambd: float):
    
    lastgaelam = 0
    advantages_reversed = []
    response_length = rewards.size(1)
    
    if action_mask is not None:
        values = action_mask * values
        rewards = action_mask * rewards

    for t in reversed(range(response_length)):
        nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
        delta = rewards[:, t] + gamma * nextvalues - values[:, t]
        lastgaelam = delta + gamma * lambd * lastgaelam
        advantages_reversed.append(lastgaelam)
    advantages = torch.stack(advantages_reversed[::-1], dim=1)
    returns = advantages + values
    return advantages.detach(), returns

def generate_samples(prompts, model, max_length, max_new_tokens, n_samples_per_prompt, micro_rollout_batch_size):
    samples_list = []
    model.eval()
    all_prompts = sum([[prompt]*n_samples_per_prompt for prompt in prompts], [])
    for i in range(0, len(all_prompts), micro_rollout_batch_size):
        prompts = all_prompts[i:i+micro_rollout_batch_size]
        inputs = actor_tokenizer(prompts, padding='max_length', max_length=max_length, truncation=True, return_tensors='pt')
        input_ids = inputs['input_ids']

        # 这里可能还需要再看看，transformers的推理
        # 这里返回结果是(B, M+T)包含原始 prompt（前 M 位）+ 生成（后 T 位）。
        seqs = model.generate(**inputs.to(device), 
                            max_new_tokens = max_new_tokens, 
                            eos_token_id = eos_token_id, 
                            pad_token_id = pad_token_id)
        if seqs.size(1) >= max_new_tokens + max_length:
            seqs = seqs[:, :max_new_tokens + max_length]
        else:
            seqs = torch.cat([seqs, torch.full((seqs.size(0), max_new_tokens + max_length - seqs.size(1)), fill_value=pad_token_id, device=seqs.device)], dim=1)

        # 这里是需要识别是否是padding字符，eos_token并不算padding 
        attention_mask = (seqs.ne(pad_token_id)).to(dtype=torch.long)
        ans = seqs[:, input_ids.size(1):]
        # 这里计算的是需要进行梯度更新的动作，因此eostoken并不参与到梯度更新中
        action_mask = (ans.ne(eos_token_id) & ans.ne(pad_token_id)).to(dtype=torch.long)
       

        samples = Samples(
            seqs=seqs,
            attention_mask=attention_mask,
            action_mask=action_mask,
            num_actions=action_mask.size(1),
            packed_seq_lens=None,
            response_length=action_mask.float().sum(dim=-1),
            total_length=attention_mask.float().sum(dim=-1),
        )
        samples_list.append(samples)

    return samples_list


def compute_rewards(kl, r, action_mask, kl_ctl, clip_reward_value):

        # 把kl散度的负数，分配到奖励中
        # [b，acttion_num]
        kl_divergence_estimate = -kl_ctl * kl #逐元素相乘 → kl_divergence_estimate：(B, T)
        rewards = kl_divergence_estimate

        # (B,)，表示每条样本有效 token 数 K
        ends = action_mask.sum(1) + 1
        if not isinstance(clip_reward_value, torch.Tensor):
            clip_reward_value = torch.tensor(clip_reward_value).to(r.device)

        # 对 r 做逐元素截断（clamp），把奖励值限制在 [-clip_reward_value, +clip_reward_value] 区间内，防止奖励过大/过小导致不稳定：
        # r : (B, 1)
        reward_clip = torch.clamp(r, -clip_reward_value,
                                  clip_reward_value)
        batch_size = r.size(0)
        for j in range(batch_size):
            # 这里应该是在《eos》这个token上加reward-model的结果
            rewards[j, :ends[j]][-1] += reward_clip[j, 0]

        return rewards

def generate_experiences(samples_list):

    actor_model.eval()
    ref_model.eval()
    reward_model.eval()
    critic_model.eval()

    experiences = []
    
    for samples in samples_list:
        seqs = samples.seqs
        attention_mask = samples.attention_mask
        action_mask = samples.action_mask
        num_actions = samples.num_actions
        with torch.no_grad():
            ## 计算策略模型输出token的概率
            # seqs: [batch_size, seq_len]
            # 这里调用的是forward函数，返回维度为【batch_size, seq_len, vocab_size】，下一个token的概率
            # 如果是generate，则是在内部调用多次forward，返回结果是(B×num_return_sequences, M+T)包含原始 prompt（前 M 位）+ 生成（后 T 位）。
            output = actor_model(seqs, attention_mask=attention_mask)
            logits = output.logits # (B, L, V) 
 
            # 需要排除掉最后一个token
            log_probs = F.log_softmax(logits[:, :-1, :], dim=-1) #(B, L-1, V) 
            # 向后取一位
            # seqs[:, 1:].unsqueeze(-1) ⇒ (B, L-1, 1)，且元素是 token id，范围在 [0, V-1]
            # log_probs_labels[b, t, 0] = log_probs[b, t, y[b, t]]
            # 每一个token对应的label（下一个token）的概率
            log_probs_labels = log_probs.gather(dim=-1, index=seqs[:, 1:].unsqueeze(-1)) #(B, L-1, 1)
            action_log_probs = log_probs_labels.squeeze(-1)[:, -num_actions:]

            ##计算参考模型输出token的概率
            ref_output = ref_model(seqs, attention_mask=attention_mask)
            ref_logits = ref_output.logits
            ref_log_probs = F.log_softmax(ref_logits[:, :-1, :], dim=-1)
            ref_log_probs_labels = ref_log_probs.gather(dim=-1, index=seqs[:, 1:].unsqueeze(-1))
            ref_action_log_probs = ref_log_probs_labels.squeeze(-1)[:, -num_actions:]

            ## 计算价值
            # batch_size, seq_len
            value = critic_model.forward(seqs, attention_mask, num_actions).to(device)
            # 转换成文本
            seq_texts = actor_tokenizer.batch_decode(seqs, skip_special_tokens=True)
            # 计算奖励模型的奖励值 
            # 这里需要先decode再encode，是因为reward model和actor model的词表可能不太一样
            reward_model_inputs = reward_tokenizer(seq_texts, return_tensors="pt", padding=True)

            # 维度猜测是 【b，1】
            r = reward_model(**reward_model_inputs.to(device)).logits # 奖励模型的输出，相当于生成最后一个token的奖励（结果奖励模型）
            # 计算kl散度

            # 结果是 [b，acttion_num]
            kl = compute_approx_kl(
                    action_log_probs,
                    ref_action_log_probs,
                    action_mask=action_mask).to(device)
            # 计算实际奖励
            rewards = compute_rewards(kl, r, action_mask, kl_ctl=0.1, clip_reward_value=0.2)
            # 计算优势和回报
            advantages, returns = get_advantages_and_returns(value, rewards, action_mask, gamma=0.1, lambd=0.2)
        # actor_model.train()
        # critic_model.train()

        experiences.append(Experience(seqs,
                    action_log_probs.detach(),
                    value.detach(),
                    returns.detach(),
                    advantages.detach(),
                    attention_mask,
                    action_mask,
                    r.detach(),
                    samples.response_length,
                    samples.total_length,
                    num_actions,
                    kl.detach(),
        ))

    return experiences

@dataclass
class BufferItem:

    seqs: torch.Tensor
    action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    attention_mask: torch.Tensor
    action_mask: torch.Tensor
    num_actions: Union[int, torch.Tensor]

def collate_fn(batch):

    seqs = []
    action_log_probs = []
    values = []
    returns = []
    advantages = []
    attention_mask = []
    action_mask = []
    
    for x in batch:
        seqs.append(x['seqs'])
        action_log_probs.append(x['action_log_probs'])
        values.append(x['values'])
        returns.append(x['returns'])
        advantages.append(x['advantages'])
        attention_mask.append(x['attention_mask'])
        action_mask.append(x['action_mask'])

    seqs = torch.cat(seqs, dim=0)
    action_log_probs = torch.cat(action_log_probs, dim=0)
    values = torch.cat(values, dim=0)
    returns = torch.cat(returns, dim=0)
    advantages = torch.cat(advantages, dim=0)
    attention_mask = torch.cat(attention_mask, dim=0)
    action_mask = torch.cat(action_mask, dim=0)
    
    return BufferItem(seqs, action_log_probs, values, returns, advantages, attention_mask, action_mask, action_mask.size(1))
    
def train_step(experience, steps):
    
    actor_model.train()
    optimizer_actor.zero_grad()

    
    sequences = experience.seqs
    old_action_log_probs = experience.action_log_probs
    advantages = experience.advantages
    num_actions = experience.num_actions
    attention_mask = experience.attention_mask
    action_mask = experience.action_mask
    old_values = experience.values
    returns = experience.returns
    
    logits = actor_model(
            sequences,
            attention_mask=attention_mask).logits
    
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=sequences[:, 1:].unsqueeze(-1))
    action_log_probs = log_probs_labels.squeeze(-1)[:, -num_actions:]
  

    
    policy_loss = compute_policy_loss(action_log_probs, old_action_log_probs, advantages,action_mask=action_mask)
    policy_loss.backward()
    optimizer_actor.step()  
    writer.add_scalar("policy_loss", policy_loss.item(), steps)
    
    critic_model.train()
    optimizer_critic.zero_grad()
    values = critic_model.forward(sequences, attention_mask, num_actions)
    value_loss = compute_value_loss(values, old_values, returns, action_mask)
    value_loss.backward()
    optimizer_critic.step()
    writer.add_scalar("value_loss", value_loss.item(), steps)
    print(f"step: {steps}  policy_loss: {policy_loss.item():.4f}  value_loss: {value_loss.item():.4f}")
    

def train():
    # 初始化经验池
    buffer = ExperienceBuffer(limit=100)
    steps = 0
    for episode in range(episodes):
        for rand_prompts in prompts_dataloader:
            # 这里取rollout_batch_size个样本
            # 生成样本（获取模型推理结果）
            samples = generate_samples(rand_prompts, actor_model, max_length, max_new_tokens, n_samples_per_prompt, micro_rollout_batch_size)
            # 生成经验（获取优势、奖励、回报等）
            experiences = generate_experiences(samples)
            buffer.append(experiences)
            dataloader = DataLoader(buffer, batch_size=micro_train_batch_size, shuffle=True, collate_fn=collate_fn)
            torch.cuda.empty_cache()
            for epoch in range(max_epochs):
                for experience in dataloader:
                    train_step(experience, steps)
                    steps += 1
            
            buffer.clear()
        
            torch.cuda.empty_cache()
            

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 一共迭代多少轮
    episodes = 3
    # 生成一次经验，训练的轮数
    max_epochs = 5
    # 一次从提示词数据集中取多少条数据用于生成经验
    rollout_batch_size = 8
    # 一次取多少条数据生成经验（生成经验需要多个模型推理，对显存要求高）
    micro_rollout_batch_size = 2
    # 一个提示词生成多少个样本
    n_samples_per_prompt = 2
    # 生成的最大长度，相当于最大动作数，数值越大，模型探索的可能性越多
    max_new_tokens = 50
    # 最大长度
    max_length = 256
    # 实际训练的batch_size大小，一次取多少条数据用于更新参数
    micro_train_batch_size = 2
    # 记录日志
    writer = SummaryWriter('./runs')
    # 策略模型
    actor_model = AutoModelForCausalLM.from_pretrained('/home/user/Downloads/Qwen2.5-0.5B-Instruct').to(device)
    # 参考模型
    ref_model = AutoModelForCausalLM.from_pretrained('/home/user/Downloads/Qwen2.5-0.5B-Instruct').to(device)
    # 奖励模型
    reward_model = AutoModelForSequenceClassification.from_pretrained('/home/user/Downloads/reward-model-deberta-v3-large-v2').to(device)
    actor_tokenizer = AutoTokenizer.from_pretrained('/home/user/Downloads/Qwen2.5-0.5B-Instruct')
    reward_tokenizer = AutoTokenizer.from_pretrained('/home/user/Downloads/reward-model-deberta-v3-large-v2')
    # 价值模型

    #	你传入的是 actor_model.base_model。在 🤗Transformers 里，AutoModelForCausalLM 由两部分构成：
	# base_model：Transformer 主体，输出 last_hidden_state ∈ ℝ^{B×L×H}；
	# lm_head：把隐藏维 H 投到词表维 V 的线性层（或 MLP），输出 logits。
    critic_model = Critic(actor_model.base_model).to(device)
    
    # 初始化优化器
    optimizer_actor = torch.optim.Adam(actor_model.parameters(), lr=0.00005)
    optimizer_critic = torch.optim.Adam(critic_model.parameters(), lr=0.00005)
    
    # 填充方式为左填充
    actor_tokenizer.padding_side = 'left'
    eos_token_id = actor_tokenizer.eos_token_id
    pad_token_id = actor_tokenizer.pad_token_id
    prompt_list = [
        '请问1+1等于多少？',
        'PowerShell，如何知道BIOS中的虚拟化是否已禁用',
        '为什么人们喜欢在水族馆里游泳，而不是在游泳池里？',
        '你是一位营销专家。为Instagram reels写30个带有营销技巧的脚本。',
        '你是一位营销专家。为Instagram reels写30个带有营销技巧的脚本。',
        '你是一位营销专家。为Instagram reels写30个带有营销技巧的脚本。',
        '为什么所有的镜子都是矩形的？',
        '我们在受感染的植物根部可以找到哪一种，臭氧还是金子？'
    ]
    prompts_dataset = PromptDataset(prompt_list, actor_tokenizer, apply_chat_template=True)
    prompts_dataloader = DataLoader(prompts_dataset, batch_size=rollout_batch_size, shuffle=True)
   
    train()
    

