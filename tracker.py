"""
ByteTrack tracking implementation in pure Python.
Includes KalmanFilter, STrack, and BYTETracker classes, as well as IoU and assignment helpers.
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import linear_sum_assignment


class KalmanFilter:
    """
    A simple 8-state constant velocity Kalman Filter for multi-object tracking.
    State representation: [x, y, a, h, vx, vy, va, vh]
    Measurement representation: [x, y, a, h]
    """
    def __init__(self):
        self.dim_x = 8
        self.dim_z = 4
        
        self.F = np.eye(self.dim_x)
        for i in range(4):
            self.F[i, i + 4] = 1.0  # dt = 1
            
        self.H = np.zeros((self.dim_z, self.dim_x))
        for i in range(4):
            self.H[i, i] = 1.0
            
        self._std_weight_position = 1.0 / 20.0
        self._std_weight_velocity = 1.0 / 160.0
        
    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = np.zeros(self.dim_x)
        mean[:4] = measurement
        
        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance
        
    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]
        ]
        Q = np.diag(np.square(std_pos + std_vel))
        
        mean = np.dot(self.F, mean)
        covariance = np.dot(self.F, np.dot(covariance, self.F.T)) + Q
        return mean, covariance
        
    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        std = [
            self._std_weight_position * measurement[3],
            self._std_weight_position * measurement[3],
            1e-2,
            self._std_weight_position * measurement[3]
        ]
        R = np.diag(np.square(std))
        
        y = measurement - np.dot(self.H, mean)
        S = np.dot(self.H, np.dot(covariance, self.H.T)) + R
        
        K = np.dot(covariance[:, :4], np.linalg.inv(S))
        
        new_mean = mean + np.dot(K, y)
        new_covariance = covariance - np.dot(K, np.dot(S, K.T))
        return new_mean, new_covariance


class STrack:
    """
    STrack represents a single tracked object (tracklet).
    """
    def __init__(self, tlbr: list[float] | np.ndarray, score: float):
        self.tlbr = np.asarray(tlbr, dtype=np.float32)
        self.score = float(score)
        
        self.is_activated = False
        self.track_id = 0
        
        self.mean = None
        self.covariance = None
        
        self.state = 1  # 1: Tracked, 2: Lost, 3: Removed
        self.age = 0
        self.time_since_update = 0
        self.history = []  # List of (cx, cy, frame_idx)
        
    @property
    def tlwh(self) -> np.ndarray:
        ret = self.tlbr.copy()
        ret[2] -= ret[0]
        ret[3] -= ret[1]
        return ret
        
    @staticmethod
    def tlbr_to_xyah(tlbr: np.ndarray) -> np.ndarray:
        w = tlbr[2] - tlbr[0]
        h = tlbr[3] - tlbr[1]
        x = tlbr[0] + w / 2.0
        y = tlbr[1] + h / 2.0
        a = w / h if h > 0 else 0.0
        return np.array([x, y, a, h], dtype=np.float32)

    @staticmethod
    def xyah_to_tlbr(xyah: np.ndarray) -> np.ndarray:
        x, y, a, h = xyah
        w = a * h
        return np.array([x - w/2.0, y - h/2.0, x + w/2.0, y + h/2.0], dtype=np.float32)

    def predict(self, kf: KalmanFilter):
        if self.state != 3:
            mean, covariance = kf.predict(self.mean, self.covariance)
            self.mean = mean
            self.covariance = covariance
            self.age += 1
            if self.state == 1:
                self.time_since_update += 1
            self.tlbr = self.xyah_to_tlbr(self.mean[:4])

    def activate(self, kf: KalmanFilter, frame_id: int, track_id: int):
        self.track_id = track_id
        self.mean, self.covariance = kf.initiate(self.tlbr_to_xyah(self.tlbr))
        self.is_activated = True
        self.state = 1
        self.time_since_update = 0
        self.age = 1
        cx = (self.tlbr[0] + self.tlbr[2]) / 2.0
        cy = (self.tlbr[1] + self.tlbr[3]) / 2.0
        self.history.append((cx, cy, frame_id))
        
    def update(self, kf: KalmanFilter, frame_id: int, new_track: STrack):
        self.tlbr = new_track.tlbr
        self.score = new_track.score
        self.mean, self.covariance = kf.update(self.mean, self.covariance, self.tlbr_to_xyah(self.tlbr))
        self.state = 1
        self.is_activated = True
        self.time_since_update = 0
        cx = (self.tlbr[0] + self.tlbr[2]) / 2.0
        cy = (self.tlbr[1] + self.tlbr[3]) / 2.0
        self.history.append((cx, cy, frame_id))
        
    def mark_lost(self):
        self.state = 2
        
    def mark_removed(self):
        self.state = 3


def iou_distance(atracks: list[STrack], btracks: list[STrack]) -> np.ndarray:
    if len(atracks) == 0 or len(btracks) == 0:
        return np.empty((len(atracks), len(btracks)), dtype=np.float32)
    a_boxes = np.array([t.tlbr for t in atracks], dtype=np.float32)
    b_boxes = np.array([t.tlbr for t in btracks], dtype=np.float32)
    x1 = np.maximum(a_boxes[:, None, 0], b_boxes[None, :, 0])
    y1 = np.maximum(a_boxes[:, None, 1], b_boxes[None, :, 1])
    x2 = np.minimum(a_boxes[:, None, 2], b_boxes[None, :, 2])
    y2 = np.minimum(a_boxes[:, None, 3], b_boxes[None, :, 3])
    w = np.maximum(0.0, x2 - x1)
    h = np.maximum(0.0, y2 - y1)
    intersection = w * h
    area_a = (a_boxes[:, 2] - a_boxes[:, 0]) * (a_boxes[:, 3] - a_boxes[:, 1])
    area_b = (b_boxes[:, 2] - b_boxes[:, 0]) * (b_boxes[:, 3] - b_boxes[:, 1])
    union = area_a[:, None] + area_b[None, :] - intersection
    iou = intersection / np.maximum(union, 1e-6)
    return 1.0 - iou


def linear_assignment(cost_matrix: np.ndarray, thresh: float) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches = []
    unmatched_rows = set(range(cost_matrix.shape[0]))
    unmatched_cols = set(range(cost_matrix.shape[1]))
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] <= thresh:
            matches.append((r, c))
            unmatched_rows.discard(r)
            unmatched_cols.discard(c)
    return matches, list(unmatched_rows), list(unmatched_cols)


class BYTETracker:
    """
    BYTETracker implements the two-stage association of ByteTrack.
    """
    def __init__(self, track_thresh: float = 0.25, track_buffer: int = 30, match_thresh: float = 0.8, min_box_score: float = 0.1):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.min_box_score = min(min_box_score, 0.5 * track_thresh)
        self.kf = KalmanFilter()
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0
        self.track_id_counter = 1
        
    def update(self, output_results: list[list[float]] | np.ndarray) -> list[STrack]:
        """
        Runs tracking step on new detections.
        output_results: list of detections, where each is [x1, y1, x2, y2, score]
        """
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        
        detections = []
        detections_second = []
        
        for res in output_results:
            tlbr = res[:4]
            score = res[4]
            tracklet = STrack(tlbr, score)
            if score >= self.track_thresh:
                detections.append(tracklet)
            elif score >= self.min_box_score:
                detections_second.append(tracklet)
                
        # Predict active and lost track states
        for track in self.tracked_stracks:
            track.predict(self.kf)
        for track in self.lost_stracks:
            track.predict(self.kf)
            
        strack_pool = [t for t in self.tracked_stracks if t.state == 1]
        strack_pool = strack_pool + self.lost_stracks
        
        # First association: high confidence detections
        dists = iou_distance(strack_pool, detections)
        matches, u_track, u_detection = linear_assignment(dists, thresh=self.match_thresh)
        
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == 1:
                track.update(self.kf, self.frame_id, det)
                activated_stracks.append(track)
            else:
                track.update(self.kf, self.frame_id, det)
                track.state = 1
                refind_stracks.append(track)
                
        # Second association: low confidence detections
        r_tracked = [strack_pool[i] for i in u_track if strack_pool[i].state == 1]
        dists_second = iou_distance(r_tracked, detections_second)
        matches_second, u_track_second, u_detection_second = linear_assignment(dists_second, thresh=0.5)
        
        for itracked, idet in matches_second:
            track = r_tracked[itracked]
            det = detections_second[idet]
            track.update(self.kf, self.frame_id, det)
            activated_stracks.append(track)
            
        for it in u_track_second:
            track = r_tracked[it]
            if not track.state == 2:
                track.mark_lost()
                lost_stracks.append(track)
                
        # Initiate new tracks from unmatched high-confidence detections
        for idet in u_detection:
            det = detections[idet]
            det.activate(self.kf, self.frame_id, self.track_id_counter)
            self.track_id_counter += 1
            activated_stracks.append(det)
            
        # Clean up lost tracks
        for it in u_track:
            track = strack_pool[it]
            if track.state == 2:
                pass
                
        # Update trackers
        self.tracked_stracks = []
        for track in activated_stracks:
            if track.state == 1:
                self.tracked_stracks.append(track)
        for track in refind_stracks:
            if track.state == 1:
                self.tracked_stracks.append(track)
                
        for track in self.lost_stracks:
            if track.state == 2:
                if track.time_since_update <= self.track_buffer:
                    lost_stracks.append(track)
                else:
                    track.mark_removed()
                    removed_stracks.append(track)
                    
        self.lost_stracks = []
        for track in lost_stracks:
            if track not in self.lost_stracks:
                self.lost_stracks.append(track)
                
        self.removed_stracks.extend(removed_stracks)
        return self.tracked_stracks


def get_motion_vector(track: STrack, current_frame_idx: int, fps: float) -> tuple[np.ndarray, float, float] | None:
    """
    Computes a 1-second window motion vector (pixels per second) for a track.
    Looks back in history to find the centroid from approximately 1 second ago.
    Requires at least fps/2 frames of history to avoid division by very small intervals.
    """
    if len(track.history) < 2:
        return None
    target_frame = current_frame_idx - fps
    best_item = None
    best_diff = float('inf')
    for cx, cy, f_id in track.history:
        diff = abs(f_id - target_frame)
        if diff < best_diff:
            best_diff = diff
            best_item = (cx, cy, f_id)
    if best_item is None:
        return None
    cx_prev, cy_prev, f_id_prev = best_item
    delta_f = current_frame_idx - f_id_prev
    if delta_f < fps / 2:
        return None
    dt = delta_f / fps
    vx = (track.history[-1][0] - cx_prev) / dt
    vy = (track.history[-1][1] - cy_prev) / dt
    return np.array([vx, vy], dtype=np.float32), cx_prev, cy_prev


def reject_outliers_iqr(vectors: np.ndarray | list[np.ndarray]) -> np.ndarray:
    """
    Prunes vector outliers based on speed (magnitude of vectors) using the 1.5 * IQR rule.
    """
    arr = np.asarray(vectors, dtype=np.float32)
    if len(arr) < 4:
        return arr
    magnitudes = np.linalg.norm(arr, axis=1)
    q25, q75 = np.percentile(magnitudes, [25, 75])
    iqr = q75 - q25
    lower_bound = q25 - 1.5 * iqr
    upper_bound = q75 + 1.5 * iqr
    mask = (magnitudes >= lower_bound) & (magnitudes <= upper_bound)
    return arr[mask]
