import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import cv2
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchsummary import summary
from matplotlib import pyplot as plt
from torchvision import transforms
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import ttk
import xiapi
from PIL import Image, ImageTk
import torchvision.transforms as transforms
import os
from PIL import Image as PILImage
from collections import deque
import time
#import rospy
#from std_msgs.msg import Int32

class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        pos = 25.0 * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        neg = 1.0 * torch.pow(euclidean_distance, 2)
        loss_contrastive = torch.mean((1 - label) * pos + label * neg)
        return loss_contrastive


def get_pairs(dataset, batch_size):
    indices = np.random.choice(range(len(imglist)), size=int(len(imglist)/2), replace=False)

    original = []
    positive_counterparts = []
    negative_counterparts = []
    o_stack = []
    p_stack = []
    n_stack = []

    for index in indices:
        img = dataset[index]
        if len(o_stack) == batch_size or index == indices[-1]:
            original.append(torch.stack(o_stack, dim=0))
            o_stack = []
        o_stack.append(img[0])

        label = img[1] - 1  
        choice = index

        while (label != img[1] and choice == index) or index == indices[-1]:
            choice = np.random.randint(0, len(dataset))
            label = dataset[choice][1]
        if len(p_stack) == batch_size:
            positive_counterparts.append(torch.stack(p_stack, dim=0))
            p_stack = []
        p_stack.append(dataset[choice][0])
        label = img[1]
        negative = index 
        while label == img[1]:
            negative = np.random.randint(0, len(dataset))
            label = dataset[negative][1]  
        if len(n_stack) == batch_size or index == indices[-1]:
            negative_counterparts.append(torch.stack(n_stack, dim=0))
            n_stack = []
        n_stack.append(dataset[negative][0]) 

    return (original, positive_counterparts, negative_counterparts)


def evaluate_pair(output1, output2, target, threshold):
    euclidean_distance = F.pairwise_distance(output1, output2)
    print(euclidean_distance)
    cond = euclidean_distance < threshold
    same_distances = []
    diff_distances = []


    
    same_sum = 0  
    diff_sum = 0  
    same_acc = 0 
    diff_acc = 0 

    for i in range(len(cond)):
        if target[i] == 0:
            same_sum += 1
            if cond[i]:
                same_acc += 1
                same_distances.append(euclidean_distance[i].item())
        elif target[i] == 1:
            diff_sum += 1
            if not cond[i]:
                diff_acc += 1
                diff_distances.append(euclidean_distance[i].item())

    print("Same pair distances:", same_distances[:10])
    print("Different pair distances:", diff_distances[:10])
    return same_acc, same_sum, diff_acc, diff_sum



def initialize_weights(m):
    classname = m.__class__.__name__

    if (classname.find('Linear') != -1):
        m.weight.data.normal_(mean = 0, std = 0.01)
    if (classname.find('Conv') != -1):
        m.weight.data.normal_(mean = 0.5, std = 0.01)


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), 
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(50176, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 128),
        )

    def encode(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

    def forward(self, input1, input2):
        return self.encode(input1), self.encode(input2)


face_proto = "./deploy.prototxt"  
face_model = "./res10_300x300_ssd_iter_140000.caffemodel" 
face_net = cv2.dnn.readNetFromCaffe(face_proto, face_model)

def detect_faces(image, conf_threshold=0.3):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(300, 300),
                                 mean=(104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w, endX), min(h, endY)
            face = image[startY:endY, startX:endX]
            faces.append(face)
    return faces

def detect_face_from_PIL(pil_image, conf_threshold=0.3):
    image_np = np.array(pil_image)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    (h, w) = image_cv.shape[:2]
    
    blob = cv2.dnn.blobFromImage(image_cv, scalefactor=1.0, size=(300, 300),
                                 mean=(104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()
    
    faces = []
    boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w, endX), min(h, endY)
            face = image_cv[startY:endY, startX:endX]
            faces.append(face)
            boxes.append((startX, startY, endX, endY))
    
    if len(faces) > 0:
        face_rgb = cv2.cvtColor(faces[0], cv2.COLOR_BGR2RGB)
        return PILImage.fromarray(face_rgb), boxes[0]
    else:
        return pil_image, None


class Contrast(torch.utils.data.Dataset):

    def __init__(self,atat_dataset):
        self.classes = atat_dataset.classes
        self.imgs = atat_dataset.imgs
        self.transform = atat_dataset.transform

    def __getitem__(self,index):
        self.target = np.random.randint(0,2)
        img1,label1 = self.imgs[index]
        new_imgs = list(set(self.imgs) - set(self.imgs[index]))
        length = len(new_imgs)
        # print(length)
        random = np.random.RandomState(42)
        if self.target == 1:
            label2 = label1
            while label2 == label1:
                choice = random.choice(length)
                img2,label2 = new_imgs[choice]
        else:
            label2 = label1 + 1
            while label2 != label1:
                choice = random.choice(length)
                img2,label2 = new_imgs[choice]

        img1 = Image.open(img1)
        img2 = Image.open(img2)
        
        img1 = detect_face_from_PIL(img1)
        img2 = detect_face_from_PIL(img2)
        
        img1 = img1.resize((224, 224))
        img2 = img2.resize((224, 224))

        img1 = img1.convert("L")
        img2 = img2.convert("L")
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return (img1,img2,self.target)

    def __len__(self):
        return(len(self.imgs))


def show(img, ax, d):
    npimg = img.numpy()
    ax.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    
    if d[0] < 0.3:
        title = f"YES (Dissimilarity: {d[0]:.4f})"
    else:
        title = f"NO (Dissimilarity: {d[0]:.4f})"
    
    ax.set_title(title, fontweight="bold", size=24)
    ax.set_xticks([])
    ax.set_yticks([])

path = './'

bs = 64
lr = 1e-4
threshold = 0.25
margin = 2.5
epochs = 50


model = SiameseNetwork()
model = model.cuda()
model.apply(initialize_weights)
optim = torch.optim.Adam(model.parameters(),lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optim,8)
criterion = ContrastiveLoss(margin)
#criterion = torch.nn.BCEWithLogitsLoss()

valid_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),])

valid_ds = Contrast(ImageFolder(root = path + './operators',transform=valid_transforms))
valid_dl = DataLoader(valid_ds,batch_size=bs)

checkpoint = torch.load('./model_PAcc0.6471_NAcc0.9756.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optim.load_state_dict(checkpoint['optimizer_state_dict'])
best_epoch = checkpoint['epoch']
best_val_loss = checkpoint['val_loss']

print(f"Loaded best model from epoch {best_epoch} with validation loss: {best_val_loss}")
model.eval()  

import tkinter as tk
from tkinter import ttk
import cv2
import xiapi
from PIL import Image, ImageTk
import torch.nn.functional as F
import torch
import torchvision.transforms as transforms
import os
from PIL import Image as PILImage
import numpy as np
from collections import deque
import time

MASTER_PASSWORD = "admin123"  
valid_path = './operators'
CAM_WIDTH, CAM_HEIGHT = 640, 480
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

model.eval()
class_embeddings = {}

label_text = None  
label_name = None  

PREDICTION_BUFFER_SIZE = 3
predictions = deque(maxlen=PREDICTION_BUFFER_SIZE)
confirmed_name = "Recognizing..."
confirmed_dist = None

def load_embeddings():
    global class_embeddings
    class_embeddings = {}
    for class_name in os.listdir(valid_path):
        class_dir = os.path.join(valid_path, class_name)
        img_files = [f for f in os.listdir(class_dir) if f.endswith('.jpg')]
        if not img_files:
            continue
        img_path = os.path.join(class_dir, img_files[0])
        img = PILImage.open(img_path)
        face, _ = detect_face_from_PIL(img)
        img = face.resize((224, 224)).convert("L")
        img_tensor = transform(img).unsqueeze(0).cuda()
        emb, _ = model(img_tensor, img_tensor)
        class_embeddings[class_name] = emb.detach()

load_embeddings()

cam = xiapi.Camera()
cam.open_device()
cam.set_exposure(8000)
cam.start_acquisition()
xi_img = xiapi.Image()

root = tk.Tk()
root.title("Face Recognition")
root.attributes('-fullscreen', False)

main_frame = ttk.Frame(root)
main_frame.pack(fill="both", expand=True)

left_panel = ttk.Frame(main_frame, width=300)
left_panel.pack(side='left', fill='y', padx=20, pady=20)
left_panel.pack_propagate(False)  

canvas = tk.Canvas(main_frame, width=CAM_WIDTH, height=CAM_HEIGHT)
canvas.pack(side='left', expand=True)


def quit_app():
    print("Shutting down...")
    try:
        cam.stop_acquisition()
        cam.close_device()
    except Exception as e:
        print(f"Error closing camera: {e}")
    root.destroy()

def back_to_main():
    for widget in left_panel.winfo_children():
        widget.destroy()
    setup_main_ui()

def setup_main_ui():
    global label_name, label_text
    label_text = tk.StringVar(value="Recognizing...")

    btn_quit = ttk.Button(left_panel, text="Quit", command=quit_app, width=25)
    btn_quit.pack(pady=10, anchor='nw', fill='x')

    btn_add_user = ttk.Button(left_panel, text="Add new user", command=setup_add_user_ui, width=25)
    btn_add_user.pack(pady=10, anchor='nw', fill='x')

    label_name = ttk.Label(left_panel, textvariable=label_text, font=("Arial", 10),
                           anchor='center', justify='center')
    label_name.pack(pady=20, anchor='center', fill='x')


def setup_add_user_ui():
    for widget in left_panel.winfo_children():
        widget.destroy()

    name_var = tk.StringVar()
    password_var = tk.StringVar()
    error_label = ttk.Label(left_panel, text="", foreground="red",
                            anchor='center', justify='center')
    ttk.Label(left_panel, text="Enter name:", anchor='center', justify='center')\
        .pack(pady=5, anchor='center', fill='x')
    name_entry = ttk.Entry(left_panel, textvariable=name_var, justify='center')
    name_entry.pack(pady=5, anchor='center', fill='x')
    ttk.Label(left_panel, text="Enter password:", anchor='center', justify='center')\
        .pack(pady=5, anchor='center', fill='x')
    password_entry = ttk.Entry(left_panel, textvariable=password_var, show="*", justify='center')
    password_entry.pack(pady=5, anchor='center', fill='x')
    status_text = tk.StringVar()
    status_label = ttk.Label(left_panel, textvariable=status_text, font=("Arial", 10),
                             anchor='center', justify='center')
    status_label.pack(pady=5, anchor='center', fill='x')
    progress = ttk.Progressbar(left_panel, orient='horizontal', length=200,
                               mode='determinate', maximum=15)
    progress.pack(pady=5, anchor='center', fill='x')

    def start_capture():
        user_name = name_var.get().strip()
        password = password_var.get().strip()

        if not user_name:
            error_label.config(text="The name input field is empty")
            return
        if not password:
            error_label.config(text="The password input field is empty")
            return

        if password != MASTER_PASSWORD:
            error_label.config(text="Incorrect password!")
            return

        error_label.config(text="")
        status_text.set("Look into the camera please!")

        save_path = os.path.join(valid_path, user_name)
        os.makedirs(save_path, exist_ok=True)

        def countdown(count=3):
            if count > 0:
                status_text.set(f"Taking pictures in {count}...")
                root.after(1000, lambda: countdown(count - 1))
            else:
                status_text.set("Taking pictures!")
                root.after(500, lambda: capture_next())

        captured_count = [0]

        def capture_next():
            count = captured_count[0]
            if count >= 15:
                status_text.set("✅ User added. Reloading recognition model...")
                load_embeddings()
                root.after(2000, back_to_main)
                return

            cam.get_image(xi_img)
            frame = xi_img.get_image_data_numpy()
            filename = os.path.join(save_path, f'{user_name}_{count+1:02d}.jpg')
            cv2.imwrite(filename, frame)
            progress['value'] = count + 1
            status_text.set(f"Captured {count+1}/15")
            captured_count[0] += 1
            root.after(300, capture_next)

        countdown()

    ttk.Button(left_panel, text="Start taking pictures", command=start_capture, width=25)\
        .pack(pady=10, anchor='center', fill='x')
    ttk.Button(left_panel, text="Back", command=back_to_main, width=25)\
        .pack(pady=10, anchor='center', fill='x')
    error_label.pack(pady=5, anchor='center', fill='x')


def update_frame():
    global confirmed_name, confirmed_dist
    cam.get_image(xi_img)
    frame = xi_img.get_image_data_numpy()
    if len(frame.shape) == 2 or frame.shape[2] == 1:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    
    try:
        pil_img = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        face_color, bbox = detect_face_from_PIL(pil_img)
        if bbox is None:
            label_text.set("Face not detected")
            raise Exception("No face detected")
        
        (startX, startY, endX, endY) = bbox
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        face = face_color.resize((224, 224)).convert("L")
        face_tensor = transform(face).unsqueeze(0).cuda()
        live_embed, _ = model(face_tensor, face_tensor)
        
        min_dist = float('inf')
        predicted_class = "Unknown operator"
        for class_name, emb in class_embeddings.items():
            dist = F.pairwise_distance(live_embed, emb)
            if dist < min_dist:
                min_dist = dist
                predicted_class = class_name

        if min_dist.item() > 0.2:
            predicted_class = "Unknown operator"

        predictions.append((predicted_class, min_dist.item()))
        
        if len(predictions) == PREDICTION_BUFFER_SIZE:
            classes = [p[0] for p in predictions if p[0] != "Unknown operator"]
            if len(classes) == 0:
                label_text.set("Recognizing...")
            else:
                class_scores = {}
                for name, dist in predictions:
                    if name != "Unknown operator":
                        if name not in class_scores:
                            class_scores[name] = {"count": 0, "total_dist": 0}
                        class_scores[name]["count"] += 1
                        class_scores[name]["total_dist"] += dist

                best_class = max(
                    class_scores.items(),
                    key=lambda x: (x[1]["count"], -x[1]["total_dist"] / x[1]["count"])
                )
                avg_dist = best_class[1]["total_dist"] / best_class[1]["count"]

                if best_class[1]["count"] >= 3 and avg_dist < 0.2:
                    confirmed_name = best_class[0]
                    publish = True
                    confirmed_dist = avg_dist
                    label_text.set(f"Operator recognized successfully ✅({confirmed_dist:.2f})")
                else:
                    publish = False
                    label_text.set("Recognizing...")
        else:
            label_text.set("Recognizing...")
            
    except Exception as e:
        predictions.clear()
        confirmed_name = "Recognizing..."
        confirmed_dist = None
        label_text.set("Face not detected")


    # if publish == True:
    #     pub.publish(1)
    # else:
    #     pub.publish(0)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_frame = cv2.resize(frame_rgb, (CAM_WIDTH, CAM_HEIGHT))
    img_pil = PILImage.fromarray(resized_frame)
    img_tk = ImageTk.PhotoImage(image=img_pil)
    canvas.img_tk = img_tk
    canvas.create_image(0, 0, anchor='nw', image=img_tk)
    root.after(30, update_frame)

# === Start ===
setup_main_ui()
update_frame()
root.mainloop()

cam.stop_acquisition()
cam.close_device()

