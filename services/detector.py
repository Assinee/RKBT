import torch
from utils import Detection, filter_detections_by_class, generate_frames
from tqdm.notebook import tqdm


import torch
from utils import Detection, filter_detections_by_class, generate_frames
from tqdm.notebook import tqdm

class Detector:
    def __init__(self, weights_path, device=0):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', weights_path, device=device)
    
    def apply_to_video(self, video_file, total_frames=750):
        frame_iterator = iter(generate_frames(video_file))
        all_detections = []
        ball_detections = []
        referee_detections = []
        goalkeeper_detections = []
        player_detections = []

        for frame in tqdm(frame_iterator, total=total_frames):
            results = self.model(frame, size=1280)
            detections = Detection.from_results(
                pred=results.pred[0].cpu().numpy(),
                names=self.model.names
            )
            all_detections.append(detections)

            ball_detections.append(filter_detections_by_class(detections=detections, class_name="ball"))
            referee_detections.append(filter_detections_by_class(detections=detections, class_name="referee"))
            goalkeeper_detections.append(filter_detections_by_class(detections=detections, class_name="goalkeeper"))
            player_detections.append(filter_detections_by_class(detections=detections, class_name="player"))

        player_goalkeeper_detections = [player + goalkeeper for player, goalkeeper in zip(player_detections, goalkeeper_detections)]
        tracked_detections = [player + goalkeeper + referee for player, goalkeeper, referee in zip(player_detections, goalkeeper_detections, referee_detections)]

        return all_detections, ball_detections, referee_detections, goalkeeper_detections, player_detections, player_goalkeeper_detections, tracked_detections