import torch
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import pickle

class CourtLineDetector:
    def __init__(self, model_path):
        self.model = models.resnet50(pretrained=False)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14*2)
       
        
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.49, 0.45, 0.4], std=[0.23, 0.225,0.224])

        ])
    
    def predict(self, image):

        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = self.transform(img_rgb).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(image_tensor)

        keypoints = outputs.squeeze().cpu().numpy()
        original_h, original_w = img_rgb.shape[:2]

        keypoints[::2] *= original_w/224.0
        keypoints[1::2] *= original_h/224.0



        return keypoints
    
     
    def predict_frames(self, frames, read_from_stub=False, stub_path = None):
        keypoints_detections = []
        
        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                keypoints_detections = pickle.load(f)
            return keypoints_detections
        
        
        
        for i, frame in enumerate(frames):
            keypoints_dict = self.predict(frame)
            keypoints_detections.append(keypoints_dict)

            print(f"Frame {i + 1}: Detected Keypoints: {keypoints_dict}")
        
        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(keypoints_detections, f) #saves data in pickle format, this transforms list to binary
        
        return keypoints_detections


    def draw_keypoints(self, image, keypoints):
        for i in range(0, len(keypoints), 2):
            x = int(keypoints[i])
            y = int(keypoints[i+1])

            cv2.putText(image, str(i//2), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255))
            cv2.circle(image,(x,y), 5, (0, 0, 255), -1) #5 pixels , -1 meaning its filled
        return image

    def draw_keypoints_on_video(self, video_frames, keypoints_detections):
        output_video_frames = []
        for frame, keypoints in zip(video_frames, keypoints_detections):
            frame = self.draw_keypoints(frame, keypoints)
            output_video_frames.append(frame)
        return output_video_frames

