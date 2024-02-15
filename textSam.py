from PIL import Image
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6]) #predefined color (light-blue)
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='red', facecolor=(0,0,0,0), lw=2))  

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='.', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='.', s=marker_size, edgecolor='white', linewidth=1.25)   

def get_boxes(obj):
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
    url = sys.argv[1]
    image = cv2.imread(url)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)

    inputs = processor(text=obj, images=image, return_tensors="pt")
    outputs = model(**inputs)

    # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
    target_sizes = torch.Tensor([image.size[::-1]])
    # Convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
    results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)
    i = 0  # Retrieve predictions for the first image for the corresponding text queries
    boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
    for box, score, label in zip(boxes, scores, labels):
        box = [round(i, 2) for i in box.tolist()]
        print(f"Detected {obj} with confidence {round(score.item(), 3)} at location {box}")

    boxes = boxes.tolist()
    if len(boxes) > 0:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(image)
        for box in boxes:
            rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        ax.axis('off')
        plt.title("Detected objects", fontsize=20)
        plt.show()

    # Convert boxes to the desired format
    input_boxes = torch.tensor(boxes, device=model.device)
    labels = [1]*len(input_boxes)
    return input_boxes, labels


if(len(sys.argv) < 3 or len(sys.argv) > 3):
    print("Usage: python textSam.py <path to image> <'s'/'a'>\ns: view one mask at time\na: view all masks in one image")
    sys.exit(1)

if(sys.argv[2] != "a" and sys.argv[2] != "s"):
    print("Usage: python textSam.py <path to image> <'s'/'a'>\ns: view one mask at time\na: view all masks in one image")
    sys.exit(1)

image = cv2.imread(sys.argv[1])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

sam_checkpoint = "models/sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)
predictor.set_image(image)


obj = input("What's the object you want to segment (type 'exit' to stop the demo): ")

while obj != "exit":
    input_boxes, input_label = get_boxes(obj)

    if len(input_boxes) == 0:
        print("No object found")
    
    else:

        transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])

        if sys.argv[2] == "s":
            for i,box in enumerate(input_boxes):
                input_box = np.array(box.tolist())
                masks, _, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=False,
                )
                plt.figure(figsize=(10, 10))
                plt.imshow(image)
                show_mask(masks[0], plt.gca())
                show_box(input_box, plt.gca())
                plt.axis('off')
                plt.show()

        elif sys.argv[2] == "a":
            input_boxes = torch.tensor(input_boxes, device=predictor.device)
            transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
            masks, _, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            for mask in masks:
                show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
            for box in input_boxes:
                show_box(box.cpu().numpy(), plt.gca())
            plt.axis('off')
            plt.show()

    obj = input("Write another object to segment (type 'exit' to stop the demo): ")