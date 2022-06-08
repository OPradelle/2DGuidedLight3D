import torch


def PointNetLoss(input, target, criterion, transf, device):
    """
        Args:
            input (torch.tensor):      The image predicted by the network. (B x C x H x W)
            target (torch.tensor):     The ground truth images. (B x H x W)
            criterion (torch.nn.Loss): The training loss.
            transf (torch.tensor):     The ptnet features rotation. (B x 64 x 64)
            device (torch.device):     The network and data's device.
        Returns:
            float: The batch's loss.
    """
    lossGT = criterion(input, target)

    if transf is not None:
        I = torch.eye(transf.shape[2])[None, :, :].to(device)
        lossTransf = 0.

        for id in range(transf.shape[1]):
            lossTransf += torch.mean(
                torch.norm(torch.bmm(transf[:, id, :, :], transf[:, id, :, :].transpose(2, 1)) - I, dim=(1, 2)))

        # see pointnet for regularization terms (weight 0.001)
        return lossGT + lossTransf * 0.001

    return lossGT
