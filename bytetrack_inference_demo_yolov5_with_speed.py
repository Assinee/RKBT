from annotate import BALL_COLOR, PLAYER_COLOR, PLAYER_IN_POSSESSION_PROXIMITY, REFEREE_COLOR, THICKNESS, BaseAnnotator, annotate_frame, filter_detections_by_class
from detect import loadyolo
from moviepy.editor import VideoFileClip
from processing import VideoConfig,get_video_writer
from statitstique import add_speed, get_player_in_possession
from track import BYTETrackerArgs, Detection, detections2boxes, match_detections_with_tracks
from yolox.tracker.byte_tracker import BYTETracker
from tqdm.notebook import tqdm


TARGET_VIDEO_PATH = "./final/08fd33_4.mp4"
SOURCE_VIDEO_PATH = "./clips/08fd33_4.mp4"
WEIGHTS_PATH = "../best.pt"

video = VideoFileClip(SOURCE_VIDEO_PATH)
video_length = video.duration

video_config = VideoConfig(
    fps=30,
    width=1920,
    height=1080)
video_writer = get_video_writer(
    target_video_path=TARGET_VIDEO_PATH,
    video_config=video_config)


base_annotator = BaseAnnotator(
    colors=[
        BALL_COLOR,
        PLAYER_COLOR,
        REFEREE_COLOR
    ],
    thickness=THICKNESS)

byte_tracker = BYTETracker(BYTETrackerArgs())

# initiate tracker
previous_tracked_detections=False
detections_dict = {}
model,frame_iterator = loadyolo(WEIGHTS_PATH,SOURCE_VIDEO_PATH)

# loop over frames
for frame in tqdm(frame_iterator, total=750):

    # run detector
    results = model(frame, size=1280)
    detections = Detection.from_results(
        pred=results.pred[0].cpu().numpy(),
        names=model.names)

    # filter detections by class
    ball_detections = filter_detections_by_class(detections=detections, class_name="ball")
    referee_detections = filter_detections_by_class(detections=detections, class_name="referee")
    goalkeeper_detections = filter_detections_by_class(detections=detections, class_name="goalkeeper")
    player_detections = filter_detections_by_class(detections=detections, class_name="player")

    player_goalkeeper_detections = player_detections + goalkeeper_detections
    tracked_detections = player_detections + goalkeeper_detections + referee_detections

    # calculate player in possession
    player_in_possession_detection = get_player_in_possession(
        player_detections=player_goalkeeper_detections,
        ball_detections=ball_detections,
        proximity=PLAYER_IN_POSSESSION_PROXIMITY)

    # track
    tracks = byte_tracker.update(
        output_results=detections2boxes(detections=tracked_detections),
        img_info=frame.shape,
        img_size=frame.shape
    )
    tracked_detections = match_detections_with_tracks(detections=tracked_detections, tracks=tracks)
    if(previous_tracked_detections):
        tracked_detections=add_speed(tracked_detections,previous_tracked_detections,video_length/750)

    previous_tracked_detections=tracked_detections
    tracked_referee_detections = filter_detections_by_class(detections=tracked_detections, class_name="referee")
    tracked_goalkeeper_detections = filter_detections_by_class(detections=tracked_detections, class_name="goalkeeper")
    tracked_player_detections = filter_detections_by_class(detections=tracked_detections, class_name="player")

    # annotate video frame
    annotated_image = annotate_frame(frame,tracked_detections,tracked_goalkeeper_detections,tracked_player_detections,tracked_referee_detections,ball_detections, player_in_possession_detection)
    
    # dict of info about each player
    for detection in detections:
        tracker_id = detection.tracker_id
        if tracker_id is not None:
            if tracker_id in detections_dict:
                detections_dict[tracker_id].append(detection)
            else:
                detections_dict[tracker_id] = [detection]
video_writer.release()
