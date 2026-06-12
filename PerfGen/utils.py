import copy

import torch


class EMA:
    """Exponential Moving Average for model parameters."""

    def __init__(self, model, decay=0.99):
        self.decay = decay
        self.shadow = copy.deepcopy(model.state_dict())
        for k in self.shadow:
            self.shadow[k] = self.shadow[k].float()

    @torch.no_grad()
    def update(self, model):
        for name, param in model.state_dict().items():
            self.shadow[name].mul_(self.decay).add_(param.float(), alpha=1 - self.decay)

    def apply(self, model):
        """Swap model weights with EMA weights. Call restore() after."""
        self.backup = copy.deepcopy(model.state_dict())
        model.load_state_dict({k: v.to(model.parameters().__next__().dtype)
                               for k, v in self.shadow.items()})

    def restore(self, model):
        """Restore original model weights after apply()."""
        model.load_state_dict(self.backup)
        del self.backup

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


def save_checkpoint(path, model, ema, optimizer, scheduler, epoch):
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "ema": ema.state_dict() if ema is not None else None,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }, path)


def load_checkpoint(path, model, ema=None, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    if ema is not None and "ema" in ckpt:
        ema.load_state_dict(ckpt["ema"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt["epoch"]
