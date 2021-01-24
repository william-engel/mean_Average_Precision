import numpy as np

def calculate_iou(gt_boxes, pred_boxes):

  pred_boxes = pred_boxes.astype('float')
  gt_boxes = gt_boxes.astype('float')
  
  x1 = np.maximum(pred_boxes[:,None,0],gt_boxes[:,0])
  y1 = np.maximum(pred_boxes[:,None,1],gt_boxes[:,1])
  x2 = np.minimum(pred_boxes[:,None,2],gt_boxes[:,2])
  y2 = np.minimum(pred_boxes[:,None,3],gt_boxes[:,3])
  
  #Intersection area
  intersectionArea = np.maximum(0.0, x2-x1) * np.maximum(0.0, y2-y1)

  #Union area
  pred_Area = (pred_boxes[:,2]-pred_boxes[:,0])*(pred_boxes[:,3]-pred_boxes[:,1])
  gt_Area = (gt_boxes[:,2]-gt_boxes[:,0])*(gt_boxes[:,3]-gt_boxes[:,1])

  unionArea = np.maximum(1e-10, pred_Area[:,None] + gt_Area - intersectionArea)

  iou = intersectionArea/unionArea
  return np.clip(iou,0.0,1.0)

def calculate_mAP(gt, pred, iou_thresholds, interpolation = '11_point'):
  # gt   [x1, y1, x2, y2, class] Nx5
  # pred [x1, y1, x2, y2, class, score] Nx6

  gt_classes = gt[:,4]
  pred_classes = pred[:,4]

  present_classes = np.unique(gt_classes)

  mAP = []
  for cls in present_classes:
    gt_ = gt[gt_classes == cls][:,:4]
    pred_ = np.delete(pred[pred_classes == cls], np.s_[4:5], axis=1) 
    for iou_threshold in iou_thresholds:
      AP = calculate_AP(gt_, pred_, iou_threshold, interpolation = '11_point')
      mAP.append(AP)
  
  return np.mean(mAP)

def calculate_AP(gt, pred, iou_threshold, interpolation = '11_point'):
  # gt   [x1, y1, x2, y2] Nx4
  # pred [x1, y1, x2, y2, score] Nx5

  score = pred[:,4]
  pred  = pred[:,:4]

  if len(pred) == 0: return 0.0 # AP = 0 

  # 1. sort predictions by score value in descending order
  pred = pred[score.argsort()][::-1]

  # 2. Calculate IoU Matrix
  iou_matrix = calculate_iou(gt, pred) # num_pred x num_gt

  # 3. Match gt boxes to pred boxes
  pred_index, gt_index = np.where(iou_matrix > iou_threshold) # TP if iou > threshold

  # remove multiple detections of the same object
  _, index = np.unique(gt_index, return_index = True) 
  pred_index = pred_index[index] # predictions with match (TP)

  # 4. Set True Positive (TP) or False Positive (FP) for each prediction
  cond = np.zeros(shape = (len(pred),1) ) # all FP (0)
  cond[pred_index] = 1 # predictions with match TP (1)

  # 5. Calculate Precision an Recall
  tp_cum = np.cumsum(cond) # TP cumulated sum

  all_pred = np.arange(1, len(pred)+1) # All predictions
  all_gt = len(gt)                     # All ground truth

  precision = tp_cum / all_pred
  recall    = tp_cum / all_gt

  # 6. 11 Point AUC Interpolation
  if interpolation == '11_point':
    recall_levels = np.linspace(0.0, 1.0, 11)
    precision_levels = [np.max(np.append(precision[recall >= level],0)) for level in recall_levels]
    return 1/11 * np.sum(precision_levels)

  # 7. All Point Interpolation
  if interpolation == 'all_point':
    precision_levels = [np.max(np.append(precision[recall >= level],0)) for level in recall][1:]
    recall_dist = recall[1:] - recall[:-1]
    return np.sum(precision_levels*recall_dist) 