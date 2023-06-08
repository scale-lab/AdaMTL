import torch
from .gumbel import gumbel_sigmoid


def get_random_policy(policy, ratio):
    # add policy * 0.0 into the loop of loss calculation to avoid the DDP issue
    random_p = torch.empty_like(policy).fill_(ratio).bernoulli() + policy * 0.0
    return random_p


class TokensSelect(torch.nn.Module):
    def __init__(self, num_embeddings, num_patches, hidden_sz=100, random=False, random_ratio=.6):
        super(TokensSelect, self).__init__()
        self.random = random
        self.random_ratio = random_ratio
        self.num_embeddings = num_embeddings
        self.num_patches = num_patches
        self.hidden_sz = hidden_sz
        self.mlp = torch.nn.Linear(num_embeddings, hidden_sz)
        self.fc = torch.nn.Linear(hidden_sz, 1)

    def unfreeze_controllers(self):
        # Allow training of the token select
        for layer in [self.mlp, self.fc]:
            for p in layer.parameters():
                p.requires_grad = True

    def freeze_controllers(self):
        # Allow training of the token select
        for layer in [self.mlp, self.fc]:
            for p in layer.parameters():
                p.requires_grad = False

    # Defining the forward pass
    def forward(self, x, hard=True):
        x = torch.nn.functional.relu(self.mlp(x))
        x = self.fc(x)

        tokens_mask = gumbel_sigmoid(x, hard=hard, tau=5)
        if self.random:
            tokens_mask = get_random_policy(tokens_mask, self.random_ratio)

        return tokens_mask

    def flops(self):
        flops = 0
        # mlp
        flops += self.num_patches*self.num_embeddings*self.hidden_sz
        # fc
        flops += self.num_patches*self.hidden_sz*1
        return flops

class BlockSelect(torch.nn.Module):
    def __init__(self, num_embeddings, num_patches, num_blocks, hidden_sz=100, random=False, random_ratio=.9):
        super(BlockSelect, self).__init__()
        self.random = random
        self.random_ratio = random_ratio
        self.num_embeddings = num_embeddings
        self.num_patches = num_patches
        self.hidden_sz = hidden_sz
        self.num_blocks = num_blocks
        self.mlp = torch.nn.Linear(num_embeddings, hidden_sz)
        self.fc = torch.nn.Linear(hidden_sz*num_patches, num_blocks)

    def unfreeze_controllers(self):
        # Allow training of the block select
        for layer in [self.mlp, self.fc]:
            for p in layer.parameters():
                p.requires_grad = True

    def freeze_controllers(self):
        # Allow training of the block select
        for layer in [self.mlp, self.fc]:
            for p in layer.parameters():
                p.requires_grad = False

    # Defining the forward pass
    def forward(self, x, hard=True):
        b_sz = x.shape[0]
        x = torch.nn.functional.relu(self.mlp(x))
        x = x.view(b_sz, -1)
        x = self.fc(x)

        block_mask = gumbel_sigmoid(x, hard=hard, tau=5)
        if self.random:
            block_mask = get_random_policy(block_mask, self.random_ratio)

        return block_mask

    def flops(self):
        flops = 0
        # mlp
        flops += 2*self.num_patches*self.num_embeddings*self.hidden_sz
        # fc
        flops += 2*self.num_patches*self.hidden_sz*self.num_blocks
        return flops
