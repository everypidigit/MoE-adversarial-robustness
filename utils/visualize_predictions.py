import matplotlib.pyplot as plt
import torch


def visualize_prediction(model, dataset, idx=0, device="cpu"):
    model.eval()

    sample = dataset[idx]
    image = sample["image"].unsqueeze(0).to(device)
    gt_mask = sample["road_gt"].numpy()

    with torch.no_grad():
        pred_mask = model(image)  # Forward pass
        pred_mask = torch.argmax(pred_mask.squeeze(0), dim=0).cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(image.squeeze(0).permute(1, 2, 0).cpu())
    axes[0].set_title("Original Image")

    axes[1].imshow(gt_mask, cmap="gray")
    axes[1].set_title("Ground Truth Mask")

    axes[2].imshow(pred_mask, cmap="gray")
    axes[2].set_title("Model Prediction")

    for ax in axes:
        ax.axis("off")

    plt.show()
