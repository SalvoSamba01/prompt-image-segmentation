import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys 
from PIL import Image
import matplotlib.patches as patches
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import math

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6]) #predefined color (light-blue)
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='.', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='.', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  


def get_coordinates(image_path):
    img = Image.open(image_path)
    fig, ax = plt.subplots()
    ax.imshow(img)
    coordinates = []

    def onclick(event):
        ix, iy = event.xdata, event.ydata
        coordinates.append([ix, iy])
        print(f'x = {ix}, y = {iy}')
        fig.canvas.mpl_disconnect(cid)
        plt.close(fig)

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    return np.array(coordinates)


def get_coordinates_and_labels(image_path):
    img = Image.open(image_path)
    fig, ax = plt.subplots()
    ax.imshow(img)
    coordinates = []
    labels = []
    circles = []

    def onclick(event):
        ix, iy = event.xdata, event.ydata
        coordinates.append([ix, iy])

        if event.button == 1:
            labels.append(1)
            print(f'x = {ix}, y = {iy}, label = {labels[-1]}')
            circle = ax.scatter(ix, iy, c='green')
            circles.append(circle)
        elif event.button == 3:
            labels.append(0)
            print(f'x = {ix}, y = {iy}, label = {labels[-1]}')
            circle = ax.scatter(ix, iy, c='red')
            circles.append(circle)
        elif event.button == 2:
            coordinates.clear()
            labels.clear()
            for circle in circles:
                circle.remove()
            circles.clear()
            print('Points and labels cleared')

        fig.canvas.draw()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    return np.array(coordinates), np.array(labels)


def get_box_coordinates(image_path):
    img = Image.open(image_path)
    fig, ax = plt.subplots()
    ax.imshow(img)
    coordinates = []

    def onpress(event):
        ix, iy = event.xdata, event.ydata
        coordinates.append([ix, iy])

    def onrelease(event):
        ix, iy = event.xdata, event.ydata
        coordinates.append([ix, iy])
        fig.canvas.mpl_disconnect(cid_press)
        fig.canvas.mpl_disconnect(cid_release)
        plt.close(fig)

    def onmove(event):
        if len(coordinates) == 1:
            ax.clear()
            ax.imshow(img)
            x0, y0 = coordinates[0]
            x1, y1 = event.xdata, event.ydata
            width = x1 - x0
            height = y1 - y0
            rect = patches.Rectangle((x0, y0), width, height, linewidth=1, edgecolor='g', facecolor='none')
            ax.add_patch(rect)
            fig.canvas.draw()

    cid_press = fig.canvas.mpl_connect('button_press_event', onpress)
    cid_release = fig.canvas.mpl_connect('button_release_event', onrelease)
    cid_move = fig.canvas.mpl_connect('motion_notify_event', onmove)

    plt.show()

    x0, y0 = coordinates[0]
    x1, y1 = coordinates[1]
    box_coordinates = np.array([x0, y0, x1, y1])

    return box_coordinates


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


if(len(sys.argv)<3 or len(sys.argv)>3):
    print("usage: python SAM.py <image_path> <model('h', 'b' or 'l')>")
    sys.exit(0)

image = cv2.imread(sys.argv[1])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

sam_checkpoint=""
model_type=""
if(sys.argv[2]=="h"):
    sam_checkpoint = "models/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
elif(sys.argv[2]=="b"):
    sam_checkpoint = "models/sam_vit_b_01ec64.pth"
    model_type = "vit_b"
elif(sys.argv[2]=="l"):
    sam_checkpoint = "models/sam_vit_l_0b3195.pth"
    model_type = "vit_l"
else:
    print("Invalid model type")
    sys.exit(0)
    
device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)
predictor.set_image(image)


print("\n\n-----------------SEGMENTING SINGLE POINT-----------------\n\n")
input_point = get_coordinates(sys.argv[1])
input_label = np.array([1])

masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)

for i, (mask, score) in enumerate(zip(masks, scores)):
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    show_mask(mask, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    plt.axis('off')
    plt.show() 



print("\n\n-----------------SEGMENTING MULTIPLE POINTS-----------------\n\n")
input_point, input_label = get_coordinates_and_labels(sys.argv[1])

mask_input = logits[np.argmax(scores), :, :]  #sceglie la mascherina con il punteggio più alto
masks, _, _ = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    mask_input=mask_input[None, :, :],
    multimask_output=True,
)

for i, (mask, score) in enumerate(zip(masks, scores)):
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    show_mask(mask, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    plt.axis('off')
    plt.show() 


print("\n\n-----------------SEGMENTING BOX-----------------\n\n")

input_box = get_box_coordinates(sys.argv[1])

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

# GENERATE ALL MASKS (DEFAULT PARAMETERS)

'''print("-----------------GENERATING ALL MASKS (DEFAULT PARAMETERS) -----------------")

image = cv2.imread(sys.argv[1])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(image)

plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.show() '''

# GENERATE ALL MASKS (MODIFIED PARAMETERS)

print("-----------------GENERATING ALL MASKS-----------------")

mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=10,
    pred_iou_thresh=0.995,
    stability_score_thresh=0.95,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100
)

masks = mask_generator.generate(image)

print("Number of masks: ",len(masks),"\n\n")
for mask in masks:
    print("Mask shape: ",mask['segmentation'].shape, " <------ (width,height)")
    print("Mask area: ",mask['area'])
    print("Mask stability score: ",mask['stability_score'])
    print("Mask IoU score: ",mask['predicted_iou'])
    print("\n")

plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.show()

mask_used = [
    mask['segmentation']
    for mask
    in sorted(masks, key=lambda x: x['area'], reverse=True)
]

images = mask_used
num_masks = len(masks)
grid_size = (int(math.ceil(math.sqrt(num_masks))), int(math.ceil(math.sqrt(num_masks))))

fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(16, 16))

for i, ax in enumerate(axes.flat):
    if i < num_masks:
        ax.imshow(images[i])
        ax.axis('off')
    else:
        ax.axis('off')
fig.suptitle("All masks considered", fontsize=15)
for i, mask in enumerate(masks):
    axes.flat[i].set_title(f"Mask {i+1} - IoU score: {mask['predicted_iou']:.3f}", fontsize=12)
plt.show()