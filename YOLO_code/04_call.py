#from utils/loss.py, class v8DetectionLoss

def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        #print("[Edit call in class v8DetectionLoss in utils/loss.py]")
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        debug = False
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)


        # [Edited] added weights
        if "weight" in batch: 
            weights = batch["weight"].view(-1, 1).to(self.device, dtype=dtype)
            # Ensure weights align with targets (some may have zero boxes)
            #if weights.shape[0] != gt_labels.numel():
            #    min_len = min(weights.shape[0], gt_labels.numel())
            #    weights = weights[:min_len]
        else:
            weights = torch.ones_like(gt_labels, device=self.device, dtype=dtype)
        #weights = weights.view(batch_size, -1, 1)  # (bs, n_max_boxes, 1)
        if debug: print(f"Loaded {weights}")

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        # dfl_conf = pred_distri.view(batch_size, -1, 4, self.reg_max).detach().softmax(-1)
        # dfl_conf = (dfl_conf.amax(-1).mean(-1) + dfl_conf.amax(-1).amin(-1)) / 2

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            # pred_scores.detach().sigmoid() * 0.8 + dfl_conf.unsqueeze(-1) * 0.2,
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        '''
        print("target_gt_idx: for each predicted anchor, the index of the GT box it’s matched to")
        print(target_gt_idx.shape)
        print(target_gt_idx[0, :5])
        print("target_bboxes: for each predicted anchor, the GT box it should regress to")
        print(pred_scores.shape)
        print(pred_scores[0, :5])
        print("target_bboxes: for each predicted anchor, the GT box it should regress to")
        print(target_bboxes.shape)
        print(target_bboxes[0,:5])
        print("target_scores: for each predicted anchor, the target class probabilities (one-hot for positives?)")
        print(target_scores.shape)
        print(target_scores[0,:5])
        print("fg_mask: boolean mask, True where anchor corresponds to a real object (foreground)")
        print(fg_mask[0, :5])
        print(fg_mask.sum())
        '''
        
        #[Edit] add weighted class loss
        #initialize with 1, which will be kept for background detections
        anchor_weights = torch.ones_like(target_gt_idx, dtype=dtype, device=self.device)
        
        if debug:
            for b in range(batch_size):
                fg_idx = target_gt_idx[b][fg_mask[b]]
                print(f"Image {b} foreground GT indices: {fg_idx}")

        #weights are given as 1D tensor per batch, here start and end for the images are computed
        batch_idx = batch["batch_idx"]  # shape: (num_total_gt,)
        #batch_size = preds[0].shape[0]
        # Count how many GTs per image
        num_gt_per_image = [(batch_idx == b).sum().item() for b in range(batch_size)]
        # Compute cumulative sum to get start/end indices
        gt_start = [0] + list(torch.cumsum(torch.tensor(num_gt_per_image), dim=0).tolist())  # len=batch_size+1

        #replace the 1 with the weigths for foreground detections
        for b in range(batch_size):
            if num_gt_per_image[b] == 0:
                continue  # skip images with no GTs

            if fg_mask[b].any():
                start_idx = gt_start[b]
                end_idx = gt_start[b+1]
                weights_per_image = weights[start_idx:end_idx, 0]  # shape: (num_gt_in_image,)

                fg_indices = target_gt_idx[b][fg_mask[b]]

                if fg_indices.numel() == 0:
                    continue  # skip if no foreground anchors

                fg_weights = weights_per_image[fg_indices]
                anchor_weights[b, fg_mask[b]] = fg_weights

                if debug:
                    print(f"Image {b}")
                    print(f"  FG anchor GT indices: {fg_indices}")
                    print(f"  FG anchor weights: {fg_weights}")

        anchor_weights = anchor_weights.unsqueeze(-1) 
        
        if debug:
            fg_anchor_weights = anchor_weights[fg_mask]
            print("Foreground anchor weights stats:")
            print(f"  min={fg_anchor_weights.min().item():.6f}")
            print(f"  max={fg_anchor_weights.max().item():.6f}")
            print(f"  mean={fg_anchor_weights.mean().item():.6f}")
            print(f"  count={fg_anchor_weights.numel()}")
            
        bce_loss = self.bce(pred_scores, target_scores.to(dtype))
        weighted_bce = bce_loss * anchor_weights
        #loss[1] = weighted_bce.sum() / (anchor_weights.sum() + 1e-9) #maybe change to normalize by target_scores_sum
        #print(f"Weighted loss with anchor_weights normalization: {loss[1]}")
        
        target_scores_sum = max(target_scores.sum(), 1)
        loss[1] = weighted_bce.sum() / target_scores_sum
        if debug: print(f"Weighted loss with target_scores_normalization: {weighted_bce.sum() / target_scores_sum}")

        # [Edit] Commented out unweighted Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        #loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE (unweighted loss)
        if debug: print(f"Default loss: {self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum}" )
        
    
        # Bbox loss
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain
        return loss * batch_size, loss.detach()  # loss(box, cls, dfl)