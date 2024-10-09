from utils import (read_video,
                   save_video)
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector

def main():
   # read video
   input_video_path = "input_videos/vid.mp4"
   video_frames = read_video(input_video_path)

   # Detect players and ball
   player_tracker = PlayerTracker(model_path='yolov8x')
   ball_tracker = BallTracker(model_path='models/best.pt')

   player_detections = player_tracker.detect_frames(video_frames, 
                                                    read_from_stub=True, 
                                                    stub_path="tracker_stubs/player_detections.pkl")
   
   ball_detections = ball_tracker.detect_frames(video_frames, 
                                                    read_from_stub=True, 
                                                    stub_path="tracker_stubs/ball_detections.pkl")
   
   # Court line detector model
   
   court_model_path = 'models/keypoints_model.pth'
   court_line_detector = CourtLineDetector(court_model_path)
   court_keypoints = court_line_detector.predict(video_frames[0])

   

   # Draw ball and player bounding boxes
   output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
   output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)
   output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)
   
   # Draw court keypoints
   


   

   save_video(output_video_frames, "output_videos/output_video.avi")

if __name__ == "__main__":
    main()   

   

