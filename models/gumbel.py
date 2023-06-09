import torch


def gumbel_softmax(logits, tau=1, hard=False, dim: int = -1):
    gumbels = (
        -torch.empty_like(logits,
                          memory_format=torch.legacy_contiguous_format).exponential_().log()
    )  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(
            logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret

# This implementation is borrowed from https://github.com/MengLcool/AdaViT
def gumbel_sigmoid(logits, tau=1, hard=False, eps=1e-10, training=True, threshold=0.5):
    if training:
        # ~Gumbel(0,1)`
        gumbels1 = (
            -torch.empty_like(logits,
                              memory_format=torch.legacy_contiguous_format)
            .exponential_()
            .log()
        )
        gumbels2 = (
            -torch.empty_like(logits,
                              memory_format=torch.legacy_contiguous_format)
            .exponential_()
            .log()
        )
        gumbels1 = (logits + gumbels1 - gumbels2) / tau
        y_soft = gumbels1.sigmoid()
    else:
        y_soft = logits.sigmoid()

    if hard:
        y_hard = torch.zeros_like(
            logits, memory_format=torch.legacy_contiguous_format
        ).masked_fill(y_soft > threshold, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret
