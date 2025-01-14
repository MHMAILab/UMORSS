import warnings
warnings.filterwarnings("ignore")

def doc_ai_orads(doc_orads, ai_preds, ai_uncertain, pred_ori, func):
    """
    Calculate the final O-RADS score based on single data point from doctor and AI model predictions, and their uncertainty.
    Parameters:
    doc_orads (float): O-RADS score from the doctor.
    ai_preds (float): AI model's prediction probability.
    ai_uncertain (float): AI model's uncertainty value.
    pred_ori (float): Original prediction value used to determine confidence.
    func (int): Method to combine predictions (1, 2, or 3).
    Returns:
    float: Final orads score.
    """
    # Convert the AI prediction to O-RADS score
    if ai_preds < 0.04:
        ai_orads = 2
    elif ai_preds < 0.09:
        ai_orads = 3
    elif ai_preds < 0.43:
        ai_orads = 4
    else:
        ai_orads = 5

    # Compute the final O-RADS score based on provided function choice
    orads = doc_orads
    if doc_orads == 0:
        orads = ai_orads
    else:
        if func == 1:
            # Method 1: Average of AI and doctor scores
            orads = 0.5 * (ai_orads + doc_orads)
        elif func == 2:
            # Method 2: Confidence-weighted average based on pred_ori
            conf = pred_ori - 0.25 if pred_ori > 0.5 else 0.75 - pred_ori
            orads = conf * ai_orads + (1 - conf) * doc_orads
        elif func == 3:
            # Method 3: Uncertainty-weighted average
            t5 = 0.246
            t95 = 0.602
            # Clip and normalize the uncertainty
            values = max(min(ai_uncertain, t95), t5)
            new_min = 0.25
            new_max = 0.75
            values = (values - t5) / (t95 - t5) * (new_max - new_min) + new_min
            orads = (1 - values) * ai_orads + values * doc_orads
        else:
            print("func error")

    return orads
    
doc_orads_example = 3           # O-RADS grade from the doctor
ai_preds_example = 0.08         # prediction value from AI model
ai_uncertain_example = 0.3      # uncertainty value from AI model
pred_ori_example = 0.55         # original prediction value(if predtion from phase2, original prediction is equal to the prediction value)

final_orads1 = doc_ai_orads(doc_orads_example, ai_preds_example, 
                           ai_uncertain_example, pred_ori_example, 1)
print(f"Average O-RADS: {final_orads1}")
final_orads2 = doc_ai_orads(doc_orads_example, ai_preds_example, 
                           ai_uncertain_example, pred_ori_example, 2)
print(f"Confidence-weighted O-RADS: {final_orads2}")
final_orads3 = doc_ai_orads(doc_orads_example, ai_preds_example, 
                           ai_uncertain_example, pred_ori_example, 3)
print(f"Uncertainty-weighted O-RADS: {final_orads3}")
