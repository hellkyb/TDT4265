import numpy as np
import matplotlib.pyplot as plt
import json
import copy
from task2_tools import read_predicted_boxes, read_ground_truth_boxes
import math

def calculate_iou(prediction_box, gt_box):
    """Calculate intersection over union of single predicted and ground truth box.

    Args:
        prediction_box (np.array of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (np.array of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]

        returns:
            float: value of the intersection of union for the two boxes.
    """
    # YOUR CODE HERE
    x1 = np.maximum(prediction_box[0], gt_box[0])
    y1 = np.maximum(prediction_box[1], gt_box[1])
    x2 = np.minimum(prediction_box[2], gt_box[2])
    y2 = np.minimum(prediction_box[3], gt_box[3])
    # REturn 0 if x2 < x1 and y2 < y1
    #print(type(x1))
    #print(x1)
    if((x2 < x1) or (y2 < y1)):
        return 0

    
    overlap = (x2 - x1)*(y2 - y1)
    area_1 = (prediction_box[2] - prediction_box[0])*(prediction_box[3] - prediction_box[1])
    area_2 = (gt_box[2] - gt_box[0])*(gt_box[3] - gt_box[1])
    
    iou = overlap / (area_1 + area_2 - overlap)
    
    return iou

def calculate_precision(num_tp, num_fp, num_fn):
    """ Calculates the precision for the given parameters.
        Returns 1 if num_tp + num_fp = 0

    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of precision
    """
    if (num_tp + num_fp) == 0:
        return 1
    return (num_tp / (num_tp + num_fp))


def calculate_recall(num_tp, num_fp, num_fn):
    """ Calculates the recall for the given parameters.
        Returns 0 if num_tp + num_fn = 0
    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of recall
    """
    if (num_tp + num_fn) == 0:
        return 0
    return (num_tp / (num_tp + num_fn))


def get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold):
    #print(prediction_boxes)
    #print(gt_boxes)
    """Finds all possible matches for the predicted boxes to the ground truth boxes.
        No bounding box can have more than one match.

        Remember: Matching of bounding boxes should be done with decreasing IoU order!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns the matched boxes (in corresponding order):
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of box matches, 4].
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of box matches, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    """
    #print(prediction_boxes)
    num_pred_match = np.empty([0, 3])
    for i in range(0, len(prediction_boxes)):
        for x in range(0, len(gt_boxes)):
            iou = calculate_iou(prediction_boxes[i], gt_boxes[x])
            if(iou >= iou_threshold):
                num_pred_match = np.append(num_pred_match, [[iou, i, x]], axis=0)
                
    num_pred_match = num_pred_match[num_pred_match[:,0].argsort()[::-1]]
    
    new_pred_match = np.empty([0,4])
    new_gt = np.empty([0,4])
    gt_index = np.empty([0])
    pred_index = np.empty([0])
    
    for j in range(0,len(num_pred_match)):
        if(not np.isin(num_pred_match[j][1], pred_index) and not np.isin(num_pred_match[j][2], gt_index)):
            gt_index = np.append(gt_index, num_pred_match[j][2])
            pred_index = np.append(pred_index, num_pred_match[j][1])
            new_pred_match = np.append(new_pred_match, [prediction_boxes[int(num_pred_match[j][1])]], axis=0)
            new_gt = np.append(new_gt, [gt_boxes[int(num_pred_match[j][2])]], axis=0)
            
    return new_pred_match, new_gt   


    # Find all possible matches with a IoU >= iou threshold
    #for i in np.nditer(prediction_boxes):
       
    # Sort all matches on IoU in descending order

    # Find all matches with the highest IoU threshold


def calculate_individual_image_result(
    prediction_boxes, gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes,
       calculates true positives, false positives and false negatives
       for a single image.
       NB: prediction_boxes and gt_boxes are not matched!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        dict: containing true positives, false positives, true negatives, false negatives
            {"true_pos": int, "false_pos": int, "false_neg": int}
    """
    box_matches, new_gt = get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold)
            

    final_dict = {'true_pos':len(box_matches),'false_pos':len(prediction_boxes)-len(box_matches),'false_neg':len(gt_boxes)-len(new_gt)}
    return final_dict
    # Find the bounding box matches with the highes IoU threshold

    # Compute true positives, false positives, false negatives


def calculate_precision_recall_all_images(
        all_prediction_boxes, all_gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates recall and precision over all images
       for a single image.
       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
    Returns:
        tuple: (precision, recall). Both float.
    """
    tp = 0
    fp = 0
    fn = 0
    i = 0
    for c in all_gt_boxes:
        dict_true_vs_false = calculate_individual_image_result(all_prediction_boxes[i], all_gt_boxes[i], iou_threshold)
        #print(dict_true_vs_false)
        i += 1
        tp += dict_true_vs_false['true_pos']
        fp += dict_true_vs_false['false_pos']
        fn += dict_true_vs_false['false_neg']
    
    
    #print(tp,fp,fn)
    precision = calculate_precision(tp, fp, fn)
    recall = calculate_recall(tp, fp, fn)
    
    
    return (precision, recall)
    # Find total true positives, false positives and false negatives
    # over all images

    # Compute precision, recall
    

def get_precision_recall_curve(all_prediction_boxes, all_gt_boxes,
                               confidence_scores, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates the recall-precision curve over all images.
       for a single image.

       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
        scores: (list of np.array of floats): each element in the list
            is a np.array containting the confidence score for each of the
            predicted bounding box. Shape: [number of predicted boxes]

            E.g: score[0][1] is the confidence score for a predicted bounding box 1 in image 0.
    Returns:
        tuple: (precision, recall). Both np array of floats floats.
    """
    # Instead of going over every possible confidence score threshold to compute the PR
    # curve, we will use an approximation
    # DO NOT CHANGE. If you change this, the tests will not pass when we run the final
    # evaluation
    confidence_thresholds = np.linspace(0, 1, 500)
    # YOUR CODE HERE
    # Iterate over all confidence thesholds, and filter out predicted bounding boxes with lower score
      
    
    precision = []
    recall = []
    
    for c_t in confidence_thresholds:
        img_pred_array = []
        for img_num, pred_boxes in enumerate(all_prediction_boxes):
            pred_array = []
            for box_num, pred_box in enumerate(pred_boxes):
                if(confidence_scores[img_num][box_num] >= c_t):
                    pred_array.append(pred_box)
            pred_array = np.array(pred_array)
            img_pred_array.append(pred_array)
        img_pred_array = np.array(img_pred_array)
        
        prc, rcl = calculate_precision_recall_all_images(img_pred_array, all_gt_boxes, iou_threshold)
        precision.append(prc)
        recall.append(rcl)
        
    return (np.array(precision), np.array(recall))
                    


def plot_precision_recall_curve(precisions, recalls):
    """Plots the precision recall curve.
        Save the figure to precision_recall_curve.png:
        'plt.savefig("precision_recall_curve.png")'

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        None
    """
    # No need to edit this code.
    plt.figure(figsize=(20, 20))
    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0.8, 1.0])
    plt.ylim([0.8, 1.0])
    plt.savefig("precision_recall_curve.png")


def calculate_mean_average_precision(precisions, recalls):
    """ Given a precision recall curve, calculates the mean average
        precision.

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        float: mean average precision
    """
    # Calculate the mean average precision given these recall levels.
    # DO NOT CHANGE. If you change this, the tests will not pass when we run the final
    # evaluation
    recall_levels = np.linspace(0, 1.0, 11)
    # YOUR CODE HERE
    max_prc_array = []
    
    for r in recall_levels:
        max_prc = 0
        for prc, r_marked in zip(precisions, recalls):
            if(r_marked >= r and prc >= max_prc):
                max_prc = prc
        max_prc_array.append(max_prc)
        
    mAP = 1.0 / len(recall_levels)*np.sum(max_prc_array)
    return mAP


def mean_average_precision(ground_truth_boxes, predicted_boxes):
    """ Calculates the mean average precision over the given dataset
        with IoU threshold of 0.5

    Args:
        ground_truth_boxes: (dict)
        {
            "img_id1": (np.array of float). Shape [number of GT boxes, 4]
        }
        predicted_boxes: (dict)
        {
            "img_id1": {
                "boxes": (np.array of float). Shape: [number of pred boxes, 4],
                "scores": (np.array of float). Shape: [number of pred boxes]
            }
        }
    """
    # DO NOT EDIT THIS CODE
    all_gt_boxes = []
    all_prediction_boxes = []
    confidence_scores = []

    for image_id in ground_truth_boxes.keys():
        pred_boxes = predicted_boxes[image_id]["boxes"]
        scores = predicted_boxes[image_id]["scores"]

        all_gt_boxes.append(ground_truth_boxes[image_id])
        all_prediction_boxes.append(pred_boxes)
        confidence_scores.append(scores)
    iou_threshold = 0.5
    precisions, recalls = get_precision_recall_curve(all_prediction_boxes,
                                                     all_gt_boxes,
                                                     confidence_scores,
                                                     iou_threshold)
    plot_precision_recall_curve(precisions, recalls)
    mean_average_precision = calculate_mean_average_precision(precisions,
                                                              recalls)
    print("Mean average precision: {:.4f}".format(mean_average_precision))


if __name__ == "__main__":
    ground_truth_boxes = read_ground_truth_boxes()
    predicted_boxes = read_predicted_boxes()
    keys_gt = list(ground_truth_boxes.keys())
    keys_pred = list(predicted_boxes.keys())
    mean_average_precision(ground_truth_boxes, predicted_boxes)
    #iou_threshold = 0.001
    #match, gt = get_all_box_matches(predicted_boxes[keys_pred[0]]['boxes'], ground_truth_boxes[keys_gt[0]], iou_threshold)
