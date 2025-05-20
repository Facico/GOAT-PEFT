import math
from abc import ABC
from typing import Optional
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# from ..lora import LoraLayer
from peft.tuners.tuners_utils import BaseTunerLayer
from functools import reduce

class TopKGOATLayer(nn.Module):
    def __init__(self, experts: nn.ModuleList, gate: nn.Module, top_k: int):
        super().__init__()
        self.experts = experts
        self.gate = gate
        self.top_k = top_k
        self.layer_loss = None
        self.expert_sum = torch.zeros((1, len(self.experts)))
        self.expert_sum_now = torch.zeros((1, len(self.experts)))
        self.aux_tot = 0
        self.merge_tot = 0
    
    def get_expert_similarity_loss(self):
        lora_A_flatten = [torch.flatten(expert.lora_A.weight).view(1, -1) for expert in self.experts]
        lora_A_flatten = torch.cat(lora_A_flatten, dim=0)
        norm_A = torch.norm(lora_A_flatten, dim=1, keepdim=True) #[8, 1]
        lora_A_score = lora_A_flatten@lora_A_flatten.T / (norm_A@norm_A.T) #[8,8]
        sim_loss = lora_A_score.fill_diagonal_(0).sum()
        return sim_loss
    
    def get_layer_loss(self, gate_logits: torch.Tensor, selected_experts: torch.Tensor) -> torch.Tensor:
        num_inputs = gate_logits.shape[0]
        num_experts = len(self.experts)
        expert_counts = torch.bincount(selected_experts.reshape(-1), minlength=num_experts)
        expert_fractions = expert_counts / num_inputs
        expert_probs = torch.sum(gate_logits, dim=0) / num_inputs
        layer_loss = num_experts * torch.sum(expert_fractions * expert_probs)
        return layer_loss
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        flattened_inputs = inputs.view((-1, inputs.shape[-1]))
        gate_logits = F.softmax(self.gate(flattened_inputs), dim=-1)
        # [batch size * seq len, num expert]
        weights, selected_experts = torch.topk(input=gate_logits, k=self.top_k, dim=-1)
        # [batch size * seq len, top k]
        weights = weights / torch.sum(weights, dim=-1, keepdim=True, dtype=inputs.dtype)
        
        shared_coefficent = 0
        results = torch.zeros_like(self.experts[0](flattened_inputs))
        shared_coefficent = 0
        for i, expert in enumerate(self.experts):
            batch_idx, nth_expert = torch.where(selected_experts == i)
            results[batch_idx] += \
                (1-shared_coefficent)*weights[batch_idx, nth_expert, None] * expert(flattened_inputs[batch_idx])
    
        results = results.view((*inputs.shape[:-1], results.shape[-1]))
        self.layer_loss = self.get_layer_loss(gate_logits=gate_logits, selected_experts=selected_experts)
        return results

class GOATExpert(nn.Module):
    def __init__(self, lora_A: nn.Module, lora_B: nn.Module, lora_dropout: nn.Module, scaling: float):
        super().__init__()
        self.lora_A = lora_A
        self.lora_B = lora_B
        self.lora_dropout = lora_dropout
        self.scaling = scaling

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.lora_B(self.lora_A(self.lora_dropout(inputs))) * self.scaling
        return outputs

class GOATLayer(BaseTunerLayer, ABC):
    def __init__(self, base_layer: nn.Module, **kwargs):
        self.base_layer = base_layer
        self.lora_rank = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_dropout = nn.ModuleDict({})
        self.lora_A = nn.ModuleDict({})
        self.lora_B = nn.ModuleDict({})
        self.kwargs = kwargs
        
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        else:
            raise ValueError(f"Unsupported layer type {type(base_layer)}")

        self.in_features = in_features
        self.out_features = out_features
        
        self.lora_gating = nn.ModuleDict({})
        self.moe_layer = nn.ModuleDict({})

    def update_layer(
        self, adapter_name: str, lora_rank: int, lora_alpha: int, lora_dropout: float, init_lora_weights: bool,
        num_experts: int, top_k: int, init_type: bool, init_cof: float=None,
        scaling_factor: int = None,         
    ) -> None:
        """
        Update the layer
        """
        if lora_rank <= 0:
            raise ValueError(f"The rank `r` should be a positive integer value but the value passed is {lora_rank}.")


        rank_list = [lora_rank // num_experts for i in range(num_experts)]
        assert 0 not in rank_list
        self.lora_rank[adapter_name] = lora_rank
        self.lora_alpha[adapter_name] = lora_alpha

        if lora_dropout > 0.0:
            self.lora_dropout[adapter_name] = nn.ModuleList(nn.Dropout(p=lora_dropout) for _ in range(num_experts))
        else:
            self.lora_dropout[adapter_name] = nn.ModuleList(nn.Identity(p=lora_dropout) for _ in range(num_experts))
            
        self.lora_A[adapter_name] = nn.ModuleList(
            nn.Linear(self.in_features, rank_list[i], bias=False) for i in range(num_experts))
        self.lora_B[adapter_name] = nn.ModuleList(
            nn.Linear(rank_list[i], self.out_features, bias=False) for i in range(num_experts))
        
        need_svd = "pissa" in init_type or "milora" in init_type or "goat" in init_type or "svd" in init_type
        assert need_svd or "hydralora" in init_type or "mole" in init_type or "lora" in init_type, f"{init_type} not implemented"
        rho = 10
        eta = float(os.getenv("ETA", 1.0))
        if "scale" in init_type:
            self.scaling[adapter_name] = [math.sqrt(3*eta*self.in_features / rank_list[i]) for i in range(num_experts)]
        elif init_type == "guassian" or init_type == "bert":
            self.scaling[adapter_name] = [1 for i in range(num_experts)]
        elif init_type == "pissa":
            self.scaling[adapter_name] = [2 for i in range(num_experts)]
        elif os.getenv("SCALING"):
            if "," in os.getenv("SCALING"):
                self.scaling[adapter_name] = list(map(float, os.getenv("SCALING").split(",")))
            else:
                self.scaling[adapter_name] = [float(os.getenv("SCALING"))] * num_experts
        else:
            if "rs" in init_type:
                self.scaling[adapter_name] = [lora_alpha / math.sqrt(rank_list[i]) for i in range(num_experts)]
            else:
                self.scaling[adapter_name] = [lora_alpha / rank_list[i] for i in range(num_experts)]
        
        self.lora_gating[adapter_name] = nn.Linear(self.in_features, num_experts, bias=False)
            
        if init_type == "mole" or init_type == "lora" or init_type == "lora_scale":
            for i in range(len(self.lora_A[adapter_name])):
                nn.init.kaiming_uniform_(self.lora_A[adapter_name][i].weight, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B[adapter_name][i].weight)
        elif "hydralora" in init_type:
            for i in range(num_experts):
                if i > 0:
                    del self.lora_A[adapter_name][-1]
                nn.init.zeros_(self.lora_B[adapter_name][i].weight)
            nn.init.kaiming_uniform_(self.lora_A[adapter_name][0].weight, a=math.sqrt(5))
        elif need_svd:
            def svd_init():
                weight = self.get_base_layer().weight
                dtype = weight.dtype
                weight = weight.to(torch.float32)
                V, S, Uh = torch.linalg.svd(weight.data, full_matrices=False)
                scaling = self.scaling[adapter_name][0]
                
                if init_type == "milora":
                    Vr = V[:, -lora_rank:]
                    Sr = S[-lora_rank:]
                    Uhr = Uh[-lora_rank:]
                    Sr /= scaling * rho
                elif init_type == "pissa" or init_type == "pissa_scale":
                    Vr = V[:, :lora_rank]
                    Sr = S[: lora_rank]
                    Uhr = Uh[:lora_rank]
                    Sr /= scaling * rho
                elif init_type == "pissa_milora":
                    Vr = torch.cat((V[:, :lora_rank//2], V[:, -lora_rank//2:]), dim=1)
                    Sr = torch.cat((S[:lora_rank//2], S[-lora_rank//2:]))
                    Uhr = torch.cat((Uh[:lora_rank//2], Uh[-lora_rank//2:]), dim=0)
                    Sr /= scaling * rho
                elif init_type == "goat":
                    Vlen = V.shape[-1]//num_experts
                    Mlen = lora_rank//num_experts
                    V_piece = [V[:, i*Vlen:i*Vlen+Mlen] for i in range(num_experts)]
                    S_piece = [S[i*Vlen:i*Vlen+Mlen] for i in range(num_experts)]
                    U_piece = [Uh[i*Vlen:i*Vlen+Mlen] for i in range(num_experts)]
                    Vr = torch.cat(V_piece, dim=1)
                    Sr = torch.cat(S_piece)
                    Uhr = torch.cat(U_piece, dim=0)
                    Sr /= scaling * rho
                elif init_type == "goat_mini":
                    Vlen = V.shape[-1]//num_experts
                    Mlen = lora_rank//num_experts
                    V_piece = [V[:, (i+1)*Vlen-Mlen:(i+1)*Vlen] for i in range(num_experts)]
                    S_piece = [S[(i+1)*Vlen-Mlen:(i+1)*Vlen] for i in range(num_experts)]
                    U_piece = [Uh[(i+1)*Vlen-Mlen:(i+1)*Vlen] for i in range(num_experts)]
                    Vr = torch.cat(V_piece, dim=1)
                    Sr = torch.cat(S_piece)
                    Uhr = torch.cat(U_piece, dim=0)
                    Sr /= scaling * rho
                else:
                    raise NotImplementedError(f"{init_type} not implemented")
                
                lora_A = torch.diag(torch.sqrt(Sr)) @ Uhr
                lora_B = Vr @ torch.diag(torch.sqrt(Sr))
                sum_rank = 0
                for i in range(num_experts):
                    self.lora_A[adapter_name][i].weight.data=lora_A[sum_rank:sum_rank+rank_list[i], :].contiguous()
                    self.lora_B[adapter_name][i].weight.data=lora_B[:, sum_rank:sum_rank+rank_list[i]].contiguous()
                    sum_rank += rank_list[i]
                self.get_base_layer().weight.data -= init_cof * scaling * lora_B @ lora_A
            svd_init()

        experts = nn.ModuleList(GOATExpert(
                self.lora_A[adapter_name][i],
                self.lora_B[adapter_name][i],
                self.lora_dropout[adapter_name][i],
                self.scaling[adapter_name][i],
        ) for i in range(num_experts))
        if top_k is not None:
            self.moe_layer[adapter_name] = TopKGOATLayer(
                experts=experts, gate=self.lora_gating[adapter_name], top_k=top_k)
        else:
            raise ValueError("Either top_k or threshold must be specified.")
        self.set_adapter(self.active_adapters)

    def reset_parameters(self, adapter_name: str, init_lora_weights: bool) -> None:
        if init_lora_weights is False:
            return
        elif adapter_name in self.lora_A.keys():
            for i in range(len(self.lora_A[adapter_name])):
                nn.init.kaiming_uniform_(self.lora_A[adapter_name][i].weight, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B[adapter_name][i].weight)


class LinearGOATLayer(nn.Module, GOATLayer):
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        lora_rank: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        init_lora_weights: bool = True,
        num_experts: int = 8,
        top_k: int = 2,
        init_type = None,
        init_cof = 1.0,
        **kwargs,
    ) -> None:
        super().__init__()
        GOATLayer.__init__(self, base_layer=base_layer, **kwargs)
        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name, lora_rank, lora_alpha, lora_dropout, init_lora_weights, num_experts, top_k, init_type, init_cof)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        previous_dtype = x.dtype
        result = self.base_layer(x, *args, **kwargs)
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue

            moe_layer = self.moe_layer[active_adapter]
            x = x.to(moe_layer.experts[0].lora_A.weight.dtype)
            result += moe_layer(x)

        result = result.to(previous_dtype)
        return result