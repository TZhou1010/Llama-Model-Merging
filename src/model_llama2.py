# model_llama2.py

import torch
import torch.nn as nn

def base_to_lora_key(base_key, suffix):
    assert base_key.startswith('model.')
    lora_key = 'base_model.model.model.' + base_key[len('model.'):]
    lora_key = lora_key.replace('weight', suffix)
    return lora_key

def softmax_entropy(logits):
    return -(logits.softmax(dim=-1) * logits.log_softmax(dim=-1)).sum(dim=-1).mean()

class AdaMergingLlama2(nn.Module):
    def __init__(self, model_structure, paramslist, names, device='cuda'):
        super().__init__()
        self.model_structure = model_structure.to(device)
        self.paramslist = paramslist
        self.names = names
        self.device = device
        self.n_models = len(paramslist)

        prior = 0.3
        self.lambdas_raw = nn.Parameter(torch.ones(2, self.n_models - 1) * prior)
        self.pretrain_lambdas = torch.ones(1, 1)

    def lambdas(self):
        task_lambdas = torch.clamp(self.lambdas_raw, min=0.0, max=1.0)
        return task_lambdas

    def collect_trainable_params(self):
        return [self.lambdas_raw]

    def set_attr(self, obj, names, val):
        if len(names) == 1:
            target = getattr(obj, names[0])
            if isinstance(target, nn.Parameter):
                target.data.copy_(val.to(target.data.dtype))
            else:
                setattr(obj, names[0], val)
        else:
            self.set_attr(getattr(obj, names[0]), names[1:], val)

    def load_weights(self):
        with torch.no_grad():
            alphas = self.lambdas().cpu()
            lambda_a = alphas[0]
            lambda_b = alphas[1]

            for idx, name in enumerate(self.names):
                base_weight = self.paramslist[0][idx].cpu()
                final_weight = base_weight

                if 'q_proj' in name or 'v_proj' in name:
                    A_list, B_list = [], []

                    for lora_idx in range(1, self.n_models):
                        lora_dict = self.paramslist[lora_idx]

                        key_A = base_to_lora_key(name, 'lora_A.weight')
                        key_B = base_to_lora_key(name, 'lora_B.weight')
                        if key_A in lora_dict and key_B in lora_dict:
                            A_list.append(lora_dict[key_A].cpu())
                            B_list.append(lora_dict[key_B].cpu())

                    if A_list and B_list:
                        A_stack = torch.stack(A_list, dim=0)
                        B_stack = torch.stack(B_list, dim=0)

                        merged_A = (lambda_a[:, None, None] * A_stack).sum(dim=0)
                        merged_B = (lambda_b[:, None, None] * B_stack).sum(dim=0)

                        delta = merged_B @ merged_A

                        if delta.shape == base_weight.shape:
                            final_weight = base_weight + delta

                        del A_list, B_list, merged_A, merged_B, delta  # <-- only inside here

                # Move only final tensor back to GPU
                final_weight = final_weight.to(self.device, dtype=torch.float16)
                self.set_attr(self.model_structure, name.split('.'), final_weight)

                del base_weight, final_weight
                torch.cuda.empty_cache()

    def forward(self, input_ids, attention_mask=None, generate_kwargs=None):
        outputs = self.model_structure(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True
        )
        if generate_kwargs is not None:
            generated = self.model_structure.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generate_kwargs
            )
            return outputs, generated
        return outputs
