"""
YOLO Implementation for Object Detection

YOLOv5-inspired architecture for traffic sign detection and object detection
in autonomous driving scenarios.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional
import math


class Conv(nn.Module):
    """Standard convolution with batch normalization and activation."""
    
    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p: Optional[int] = None, 
                 g: int = 1, act: bool = True):
        """
        Initialize convolution layer.
        
        Args:
            c1: Input channels
            c2: Output channels
            k: Kernel size
            s: Stride
            p: Padding (auto if None)
            g: Groups
            act: Whether to use activation
        """
        super(Conv, self).__init__()
        
        self.conv = nn.Conv2d(c1, c2, k, s, k // 2 if p is None else p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Standard bottleneck block."""
    
    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """
        Initialize bottleneck.
        
        Args:
            c1: Input channels
            c2: Output channels
            shortcut: Whether to use shortcut connection
            g: Groups
            e: Expansion ratio
        """
        super(Bottleneck, self).__init__()
        
        c_ = int(c2 * e)  # Hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""
    
    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, 
                 g: int = 1, e: float = 0.5):
        """
        Initialize C3 block.
        
        Args:
            c1: Input channels
            c2: Output channels
            n: Number of bottlenecks
            shortcut: Whether to use shortcuts
            g: Groups
            e: Expansion ratio
        """
        super(C3, self).__init__()
        
        c_ = int(c2 * e)  # Hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer."""
    
    def __init__(self, c1: int, c2: int, k: int = 5):
        """
        Initialize SPPF layer.
        
        Args:
            c1: Input channels
            c2: Output channels
            k: Kernel size for max pooling
        """
        super(SPPF, self).__init__()
        
        c_ = c1 // 2  # Hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class Detect(nn.Module):
    """YOLOv5 detection head."""
    
    def __init__(self, nc: int = 80, anchors: List = (), ch: List = ()):
        """
        Initialize detection head.
        
        Args:
            nc: Number of classes
            anchors: Anchor boxes
            ch: Input channels for each detection layer
        """
        super(Detect, self).__init__()
        
        self.nc = nc  # Number of classes
        self.no = nc + 5  # Number of outputs per anchor
        self.nl = len(anchors)  # Number of detection layers
        self.na = len(anchors[0]) // 2  # Number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # Init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # Init anchor grid
        
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)
        self.inplace = True  # Use in-place ops for speed
    
    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        """Forward pass through detection head."""
        z = []  # Inference output
        
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # Conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            
            if not self.training:  # Inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
                
                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))
        
        return x if self.training else (torch.cat(z, 1), x)
    
    def _make_grid(self, nx: int = 20, ny: int = 20, i: int = 0):
        """Create grids for anchor boxes."""
        d = self.anchors[i].device
        yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)])
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        anchor_grid = (self.anchors[i].clone() * self.stride[i]).view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()
        return grid, anchor_grid


class YOLOv5(nn.Module):
    """
    YOLOv5 model for object detection.
    
    Adapted for traffic sign detection in autonomous driving scenarios.
    """
    
    def __init__(self, nc: int = 43, depth_multiple: float = 1.0, 
                 width_multiple: float = 1.0):
        """
        Initialize YOLOv5 model.
        
        Args:
            nc: Number of classes
            depth_multiple: Model depth multiple
            width_multiple: Model width multiple
        """
        super(YOLOv5, self).__init__()
        
        self.nc = nc
        self.model = self._build_model(nc, depth_multiple, width_multiple)
        self.stride = torch.tensor([8., 16., 32.])  # Strides computed during build
        
        # Initialize detection head
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            m.stride = self.stride
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._check_anchor_order(m)
    
    def _build_model(self, nc: int, depth_multiple: float, width_multiple: float):
        """Build YOLOv5 model architecture."""
        # Model configuration (simplified YOLOv5s)
        anchors = [
            [10, 13, 16, 30, 33, 23],  # P3/8
            [30, 61, 62, 45, 59, 119],  # P4/16
            [116, 90, 156, 198, 373, 326]  # P5/32
        ]
        
        # Backbone and head configuration
        backbone = [
            [-1, 1, Conv, [64, 6, 2, 2]],  # 0-P1/2
            [-1, 1, Conv, [128, 3, 2]],    # 1-P2/4
            [-1, 3, C3, [128]],
            [-1, 1, Conv, [256, 3, 2]],    # 3-P3/8
            [-1, 6, C3, [256]],
            [-1, 1, Conv, [512, 3, 2]],    # 5-P4/16
            [-1, 9, C3, [512]],
            [-1, 1, Conv, [1024, 3, 2]],   # 7-P5/32
            [-1, 3, C3, [1024]],
            [-1, 1, SPPF, [1024, 5]]       # 9
        ]
        
        head = [
            [-1, 1, Conv, [512, 1, 1]],
            [-1, 1, nn.Upsample, [None, 2, 'nearest']],
            [[-1, 6], 1, Concat, [1]],  # cat backbone P4
            [-1, 3, C3, [512, False]],  # 13
            
            [-1, 1, Conv, [256, 1, 1]],
            [-1, 1, nn.Upsample, [None, 2, 'nearest']],
            [[-1, 4], 1, Concat, [1]],  # cat backbone P3
            [-1, 3, C3, [256, False]],  # 17 (P3/8-small)
            
            [-1, 1, Conv, [256, 3, 2]],
            [[-1, 14], 1, Concat, [1]],  # cat head P4
            [-1, 3, C3, [512, False]],  # 20 (P4/16-medium)
            
            [-1, 1, Conv, [512, 3, 2]],
            [[-1, 10], 1, Concat, [1]],  # cat head P5
            [-1, 3, C3, [1024, False]],  # 23 (P5/32-large)
            
            [[17, 20, 23], 1, Detect, [nc, anchors]]  # Detect(P3, P4, P5)
        ]
        
        # Build model layers
        layers = []
        ch = [3]  # Input channels
        
        for i, (f, n, m, args) in enumerate(backbone + head):
            m = eval(m) if isinstance(m, str) else m  # String to class
            
            # Scale channels by width_multiple
            if args:
                args = list(args)
                if m in [Conv, C3, SPPF]:
                    args[0] = max(round(args[0] * width_multiple / 8) * 8, 1)  # Round to 8
                elif m is Detect:
                    args[1] = anchors
                    args[2] = [ch[x] for x in f]
            
            # Create layer
            n = max(round(n * depth_multiple), 1) if n > 1 else n  # Scale depth
            
            if m in [Conv, Bottleneck, C3, SPPF]:
                c1, c2 = ch[f], args[0]
                args = [c1, c2, *args[1:]]
            elif m is Concat:
                c2 = sum([ch[x] for x in f])
            elif m is Detect:
                args.append([ch[x] for x in f])
            
            module = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)
            
            # Add to model
            layers.append(module)
            ch.append(c2 if hasattr(module, 'out_channels') else args[0] if args else ch[f])
        
        # Create final model
        model = nn.Sequential(*layers)
        return model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through YOLOv5."""
        return self._forward_once(x)
    
    def _forward_once(self, x: torch.Tensor) -> torch.Tensor:
        """Single forward pass."""
        y = []  # Outputs
        
        for m in self.model:
            if m.f != -1:  # If not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
        
        return x
    
    def _check_anchor_order(self, m: Detect):
        """Check anchor order against stride order for YOLOv5."""
        a = m.anchor_grid.prod(-1).view(-1)  # Anchor area
        da = a[-1] - a[0]  # Delta a
        ds = m.stride[-1] - m.stride[0]  # Delta s
        if da.sign() != ds.sign():  # Same order
            m.anchors[:] = m.anchors.flip(0)
            m.anchor_grid[:] = m.anchor_grid.flip(0)


class Concat(nn.Module):
    """Concatenate tensors along specified dimension."""
    
    def __init__(self, dimension: int = 1):
        """Initialize concatenation layer."""
        super(Concat, self).__init__()
        self.d = dimension
    
    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        """Forward pass."""
        return torch.cat(x, self.d)


class YOLOLoss(nn.Module):
    """
    YOLOv5 loss function.
    
    Combines classification, objectness, and bounding box regression losses.
    """
    
    def __init__(self, model, autobalance: bool = False):
        """
        Initialize YOLO loss.
        
        Args:
            model: YOLOv5 model
            autobalance: Whether to use automatic loss balancing
        """
        super(YOLOLoss, self).__init__()
        
        device = next(model.parameters()).device
        h = model.hyp if hasattr(model, 'hyp') else {}  # Hyperparameters
        
        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h.get('cls_pw', 1.0)], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h.get('obj_pw', 1.0)], device=device))
        
        # Class label smoothing
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))
        
        # Focal loss
        g = h.get('fl_gamma', 0.0)  # Focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)
        
        det = model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, 0.02])
        self.ssi = list(det.stride).index(16) if autobalance else 0  # Stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))
    
    def forward(self, p: List[torch.Tensor], targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate loss.
        
        Args:
            p: Predictions
            targets: Ground truth targets
            
        Returns:
            Tuple of (total_loss, loss_components)
        """
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # Targets
        
        # Losses
        for i, pi in enumerate(p):  # Layer index, layer predictions
            b, a, gj, gi = indices[i]  # Image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # Target obj
            
            n = b.shape[0]  # Number of targets
            if n:
                ps = pi[b, a, gj, gi]  # Prediction subset corresponding to targets
                
                # Regression
                pxy = ps[:, :2].sigmoid() * 2 - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # Predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # IoU
                lbox += (1.0 - iou).mean()  # IoU loss
                
                # Objectness
                score_iou = iou.detach().clamp(0).type(tobj.dtype)
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # IoU ratio
                
                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # Targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE
            
            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()
        
        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        
        lbox *= self.hyp.get('box', 0.05)
        lobj *= self.hyp.get('obj', 1.0)
        lcls *= self.hyp.get('cls', 0.5)
        bs = tobj.shape[0]  # Batch size
        
        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()
    
    def build_targets(self, p: List[torch.Tensor], targets: torch.Tensor):
        """Build targets for loss calculation."""
        # This is a simplified version - full implementation would be more complex
        na, nt = self.na, targets.shape[0]  # Number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        
        gain = torch.ones(7, device=targets.device)  # Normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # Same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # Append anchor indices
        
        # Process each detection layer
        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain
            
            # Match targets to anchors
            t = targets * gain
            
            # Simple assignment (full version would use more sophisticated matching)
            if nt:
                # Simplified target assignment
                j = torch.ones(t.shape[0], dtype=torch.bool)
                t = t[j]  # Filter
                
                # Compute grid coordinates
                gxy = t[:, 2:4]  # Grid xy
                gxi = gain[[2, 3]] - gxy  # Inverse
                j, k = ((gxy % 1 < 0.5) & (gxy > 1)).T
                l, m = ((gxi % 1 < 0.5) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + 
                          torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], 
                                     device=targets.device).float()[:, None])[j]
            else:
                t = targets[0]
                offsets = 0
            
            # Define targets
            if t.shape[0]:
                b, c = t[:, :2].long().T  # Image, class
                gxy = t[:, 2:4]  # Grid xy
                gwh = t[:, 4:6]  # Grid wh
                gij = (gxy - offsets).long()
                gi, gj = gij.T  # Grid indices
                
                # Append
                a = t[:, 6].long()  # Anchor indices
                indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))
                tbox.append(torch.cat((gxy - gij, gwh), 1))  # Box
                anch.append(anchors[a])  # Anchors
                tcls.append(c)  # Class
            else:
                indices.append((torch.zeros(0), torch.zeros(0), torch.zeros(0), torch.zeros(0)))
                tbox.append(torch.zeros(0, 4))
                anch.append(torch.zeros(0, 2))
                tcls.append(torch.zeros(0))
        
        return tcls, tbox, indices, anch


def smooth_BCE(eps: float = 0.1):
    """Return positive, negative label smoothing BCE targets."""
    return 1.0 - 0.5 * eps, 0.5 * eps


def bbox_iou(box1: torch.Tensor, box2: torch.Tensor, x1y1x2y2: bool = True, 
             GIoU: bool = False, DIoU: bool = False, CIoU: bool = False, eps: float = 1e-7):
    """
    Calculate IoU between bounding boxes.
    
    Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    """
    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2
    
    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    
    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps
    
    iou = inter / union
    
    if CIoU or DIoU or GIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if CIoU:
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU
    
    return iou  # IoU


class FocalLoss(nn.Module):
    """Focal Loss implementation."""
    
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        """Initialize focal loss."""
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'
    
    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """Calculate focal loss."""
        loss = self.loss_fcn(pred, true)
        pred_prob = torch.sigmoid(pred)
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss