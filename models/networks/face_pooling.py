import torch
from torch_scatter import scatter_max, scatter_add, scatter_mean


class FacePooling(torch.nn.Module):
    def __init__(self, pool_type="max"):
        super(FacePooling, self).__init__()
        self.pool_type = pool_type
        assert self.pool_type in ["max", "sum", "average"]
        # Select appropriate function
        if self.pool_type == "max":
            self.pool_func = scatter_max
        elif self.pool_type == "average":
            self.pool_func = scatter_mean
        elif self.pool_type == "sum":
            self.pool_func = scatter_add
        else:
            raise NotImplementedError

    def forward(self, img, index, max_index):
        """
        Args:
            img: [B,F,H,W]
            spx: [B,H,W] (idx=0 is ignored)
            max_index
        Return:
            face_pooling_feature: [B,F,max_index]
        """
        bsize, fsize, _, _ = img.shape

        feature_list = []
        for b_idx in range(bsize):
            # Flatten for operation
            b_flat_img = img[b_idx].view(fsize, -1)  # [F,H*W]
            b_flat_index = index[b_idx].view(-1)  # [H*W]

            # Execute pooling ([F,max_index+1])
            if self.pool_type == "max":
                b_feature = self.pool_func(b_flat_img, b_flat_index.long(),
                                           dim_size=max_index + 1, fill_value=0)[0]
            else:
                b_feature = self.pool_func(b_flat_img, b_flat_index.long(),
                                           dim_size=max_index + 1, fill_value=0)

            # Remove background id (num=0)
            feature_list.append(b_feature[:, 1:])
        # Feature integration ([B,F,max_index])
        final_feature = torch.stack(feature_list, 0)

        return final_feature
