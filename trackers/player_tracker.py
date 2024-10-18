from ultralytics import YOLO
import cv2
import pickle
import sys
sys.path.append('../')
from utils import measure_distance, get_center_of_bbox

class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def choose_and_filter_players(self, court_keypoints, player_detections):
        player_detections_first_frame = player_detections[100]
        chosen_players = self.choose_players(court_keypoints, player_detections_first_frame)
        filtered_player_detections = []
        for player_dict in player_detections:
            filtered_player_dict = {track_id: bbox for track_id, bbox in player_dict.items() if track_id in chosen_players}
            filtered_player_detections.append(filtered_player_dict)
        
        return filtered_player_detections

    def choose_players(self, court_keypoints, player_dict):
        distances = []
        for track_id, bbox in player_dict.items():
            player_center = get_center_of_bbox(bbox)
            first_frame_keypoints = court_keypoints[0]

            
            distance = []
            for i in range(0, len(first_frame_keypoints), 2):
                court_keypoint = (first_frame_keypoints[i], first_frame_keypoints[i+1])
                distance.append(measure_distance(player_center, court_keypoint))

            distance.sort() 
            sum_distance = sum(distance[:10]) #take sum of 7 keypoints with least distance

            distances.append((track_id, sum_distance))

        #sort the distances
        distances.sort(key = lambda x: x[1]) #lambda x: x[1] defines function returning second element of x
        chosen_players = [distances[0][0], distances[1][0]]
        return chosen_players

        


    
    def detect_frames(self, frames, read_from_stub=False, stub_path = None):
        player_detections = []
        
        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections
        
        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)
        
    
        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(player_detections, f) #saves data in pickle format, this transforms list to binary

        return player_detections

    def detect_frame(self, frame):
        results = self.model.track(frame, persist=True)[0] #persist for tracking, even tough we give individual frame at once
        id_name_dict = results.names

        player_dict = {}
        for box in results.boxes:
            if box.id is not None:
                track_id = int(box.id.tolist()[0])
            else:
                continue  # Skip if there's no track ID
                
            result = box.xyxy.tolist()[0]
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]
            if object_cls_name == "person":
                player_dict[track_id] = result
        return player_dict

    def draw_bboxes(self, video_frames, player_detections):
        output_video_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            # draw bounding boxes
            for track_id, bbox in player_dict.items():
                x1, y1, x2 ,y2 = bbox
                cv2.putText(frame,  f"Player ID: {track_id}", (int(bbox[0]), int(bbox[1] - 10)),  cv2.FONT_HERSHEY_SIMPLEX,  0.8,  (0, 0, 255),  2)
                cv2.rectangle(frame, (int(x1),int(y1)), (int(x2), int(y2)), (0, 0, 255), 2) #2 for outside borders
            output_video_frames.append(frame)
        return(output_video_frames)




    