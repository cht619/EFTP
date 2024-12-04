import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import List
from torch.func import functional_call
import functools
from copy import deepcopy


def get_named_submodule(model, sub_name: str):
    names = sub_name.split(".")
    module = model
    for name in names:
        module = getattr(module, name)

    return module


def set_named_submodule(model, sub_name, value):
    names = sub_name.split(".")
    module = model
    for i in range(len(names)):
        if i != len(names) - 1:
            module = getattr(module, names[i])

        else:
            setattr(module, names[i], value)


def del_attr(obj, names: List[str]):
    """
    Deletes an attribute from an object recursively.

    Args:
        obj (object): Object to delete attribute from.
        names (list): List of attribute names to delete recursively.
    """
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        del_attr(getattr(obj, names[0]), names[1:])


def set_attr(obj, names: List[str], val):
    """
    Sets an attribute of an object recursively.

    Args:
        obj (object): Object to set attribute of.
        names (list): List of attribute names to set recursively.
        val (object): Value to set the attribute to.
    """
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        set_attr(getattr(obj, names[0]), names[1:], val)


def get_attr(obj, names: List[str]):
    """
    Gets an attribute of an object recursively.

    Args:
        obj (object): Object to get attribute of.
        names (list): List of attribute names to get recursively.

    Returns:
        object: The attribute of the object.
    """
    if len(names) == 1:
        return getattr(obj, names[0])
    else:
        return get_attr(getattr(obj, names[0]), names[1:])


class Router(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.fc2 = nn.Linear(hidden_size, num_experts, bias=True)

    def init_weight(self, init_lambda: float):
        nn.init.normal_(self.fc1.weight, std=0.01)
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, std=0.01)
        nn.init.constant_(self.fc2.bias, init_lambda)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = F.relu(self.fc1(hidden_states))
        return self.fc2(hidden_states)


class MoE(nn.Module):
    # variable to store the merged state dict temporarily
    _merged_state_dict = None

    def __init__(
        self,
        base_model: nn.Module,
        init_lambda: float = 0.2,
        batch_first: bool = False,
        router_hidden_layers: int = 1,  #
        batch_reduce: bool = False,
    ):

        super().__init__()
        self.num_experts = 2  #  local
        self.in_features = base_model.in_features
        self.out_features = base_model.out_features

        self.hidden_size = self.in_features
        self.batch_first = batch_first
        self.batch_reduce = batch_reduce

        # linear layers. get experts' weight  [bs, n_experts]

        # compute the task vectors
        self.base_model = base_model.requires_grad_(False)  # global model
        self.expert_model = deepcopy(self.base_model)  # local model
        self.expert_model.requires_grad_(False)
        self.task_vectors = nn.ModuleList([self.expert_model])

    def update_expert_model(self, state_dict):
        self.expert_model.load_state_dict(state_dict)

    @property
    def forward_model(self):
        return functools.partial(
            functional_call,
            self.base_model,
            self._merged_state_dict,
        )

    def merge_weights(self, expert_weights):
        state_dict = self.base_model.state_dict(keep_vars=True)
        for task_vector in self.task_vectors:
            for name, param in task_vector.named_parameters():
                state_dict[name] = expert_weights[0] * state_dict[name] + expert_weights[1] * param
        self._merged_state_dict = state_dict
        return state_dict

    def forward(self, hidden_states: Tensor):
        if self.router.num_hidden_layers == 0:
            gate_weights = self.router()
        else:
            gate_weights = self.router(hidden_states)  # [bs, n_experts]
            if self.batch_first:  # default True
                # the input is in the shape of (batch_size, seq_len, hidden_size)
                gate_weights = gate_weights.mean(dim=1)
            else:
                # the input is in the shape of (seq_len, batch_size, hidden_size)
                gate_weights = gate_weights.mean(dim=0)

        # gate_weights.shape=[bs, n_models]
        gate_weights = gate_weights.softmax(1)  # model_dim
        gate_weights = gate_weights.mean(dim=0)
        self.merge_weights(gate_weights)  # 利用gate weight来得到merge_weight
        output_hidden_states = self.forward_model(hidden_states)


        self._merged_state_dict = None
        return output_hidden_states


class WMoE(MoE):
    # expert --> test data
    def __init__(self, base_model, **kwargs):
        super().__init__(base_model, **kwargs)
        self.inference = False
        self.update_count = 0  #
        self.num_experts = 15
        # set router
        self.router = Router(self.hidden_size, self.num_experts)
        self.router.num_hidden_layers = 2
        self.router.init_weight(0.2)

        self.expert_models = [deepcopy(self.expert_model) for _ in range(self.num_experts)]
        self.task_vectors = nn.ModuleList(self.expert_models)

    def update_expert_model(self, state_dict):
        self.task_vectors[self.update_count].load_state_dict(state_dict)
        if self.update_count + 1 == self.num_experts:
            self.update_count = 0
        else:
            self.update_count += 1

    def merge_weights(self, expert_weights):
        state_dict = self.base_model.state_dict(keep_vars=True)

        for weight, task_vector in zip(expert_weights, self.task_vectors):
            for name, param in task_vector.named_parameters():
                state_dict[name] = state_dict[name] + weight * (param - state_dict[name])

        self._merged_state_dict = state_dict
        return state_dict

    def forward(self, hidden_states: Tensor):
        if self.inference:
            return self.expert_model(hidden_states)

        else:  #
            if self.router.num_hidden_layers == 0:
                gate_weights = self.router()
            else:
                gate_weights = self.router(hidden_states)  # [bs, n_experts]
                if self.batch_first:  # default True
                    # the input is in the shape of (batch_size, seq_len, hidden_size)
                    gate_weights = gate_weights.mean(dim=1)
                else:
                    # the input is in the shape of (seq_len, batch_size, hidden_size)
                    gate_weights = gate_weights.mean(dim=0)

            # gate_weights.shape=[bs, n_models]
            gate_weights = gate_weights.softmax(1)  # model_dim
            gate_weights = gate_weights.mean(dim=0)
            self.merge_weights(gate_weights)  #
            output_hidden_states = self.forward_model(hidden_states)

            self._merged_state_dict = None
            return output_hidden_states


# model merging continual TTA
def get_continual_dataloaders(cfg, runner, domains):
    train_dataloader_cfg = deepcopy(cfg.test_dataloader)
    train_dataloader_cfg.sampler.shuffle = False  # test

    dataloaders_dict = {}  # multi domains
    for domain in domains:
        if domain in ['fog', 'snow', 'rain', 'night']:
            train_dataloader_cfg.dataset.data_prefix.img_path = 'rgb_anon/{}/train'.format(domain)
            train_dataloader_cfg.dataset.data_prefix.seg_map_path = 'gt/{}/train'.format(domain)
        elif domain in ['Rio', 'Rome', 'Taipei', 'Tokyo']:
            train_dataloader_cfg.dataset.data_prefix.img_path = '{}/Images/Test'.format(domain)
            train_dataloader_cfg.dataset.data_prefix.seg_map_path = '{}/Labels/Test'.format(domain)

        train_dataloader_cfg.dataset.pipeline = cfg.test_pipeline
        train_dataloader = runner.build_dataloader(dataloader=train_dataloader_cfg)
        dataloaders_dict[domain] = train_dataloader

    return dataloaders_dict
