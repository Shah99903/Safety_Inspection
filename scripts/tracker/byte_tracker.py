import numpy as np
from norfair import Detection, Tracker

class BYTETracker:
    def __init__(self, distance_threshold=30):
        self.tracker = Tracker(
            distance_function=self._euclidean_distance,
            distance_threshold=distance_threshold
        )
        self.next_id = 0
        self.id_map = {}

    def _euclidean_distance(self, det, trk):
        return np.linalg.norm(det.points - trk.estimate)

    def update_tracks(self, detections, frame):
        person_detections = []
        centers = []
    
        for det in detections:
            x1, y1, x2, y2, score, cls = det
            if int(cls) == 0:  # Only track persons
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                person_detections.append(
                    Detection(points=np.array([[center_x, center_y]]), scores=np.array([score]))
                )
                centers.append([x1, y1, x2, y2, center_x, center_y])  # <--- include center

        tracked_objects = self.tracker.update(detections=person_detections)
        results = []

        for trk, bbox_data in zip(tracked_objects, centers):
            x1, y1, x2, y2, center_x, center_y = bbox_data
    
            trk_id = self.id_map.get(id(trk))
            if trk_id is None:
                trk_id = self.next_id
                self.id_map[id(trk)] = trk_id
                self.next_id += 1
    
            results.append({
                "track_id": trk_id,
                "bbox": [x1, y1, x2, y2],
                "center": (center_x, center_y)
            })
    
        return results

