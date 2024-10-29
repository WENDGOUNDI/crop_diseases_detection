# Libraries importation
from ultralytics import YOLO

# Load our trained model
model = YOLO("path to your trained best model")

# Create a function for predicting crop disease
def cropPrediction(model, input_image):
  """ 
  function for predicting crop disease
  Input Parameters:
  model --> the trained model for prediction --> .pt file
  input_image --> image path of the image to be predicted --> str
  Output:
  predicted_label --> returning the crop disease --> str
  """
  # Predict with the model
  results = model.predict(input_image, save=False, verbose=False)
  labels_names = results[0].names
  pred_cls_idx = [result.probs.top1 for result in results]
  pred_cls_idx = pred_cls_idx[0]
  predicted_label = labels_names[pred_cls_idx]
  return predicted_label

if __name__ == "__main__":
  predImage = "./split_data/test/Gray_Leaf_Spot/Corn_Gray_Spot (11).jpg" # adjust the path based on your local dir
  predlabel = cropPrediction(model, predImage)
  print(f"Predicted Label: {predlabel}")