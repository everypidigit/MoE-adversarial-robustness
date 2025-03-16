from UNetTorch import UNet
import torch.nn as nn
import torch


class UNetEnsemble(nn.Module):
    def __init__(
        self,
        model_paths=None,
        in_channels=3,
        out_channels=1,
        device="mps"
    ):
        super(UNetEnsemble, self).__init__()

        # Create 3 separate UNet models
        self.models = nn.ModuleList([
            UNet(
                in_channels=in_channels, out_channels=out_channels
            ).to(device),
            UNet(
                in_channels=in_channels, out_channels=out_channels
            ).to(device),
            UNet(
                in_channels=in_channels, out_channels=out_channels
            ).to(device),
        ])

        # Load pre-trained models if paths are provided
        if model_paths:
            for model, path in zip(self.models, model_paths):
                checkpoint = torch.load(path, map_location=device)
                model.load_state_dict(checkpoint["model_state_dict"])
                model.eval()

    def forward(self, x):
        preds = [model(x) for model in self.models]
        avg_pred = torch.stack(preds, dim=0).mean(dim=0)

        return avg_pred
