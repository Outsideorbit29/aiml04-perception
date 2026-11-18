# perception.types.PerceptionFrame without the lanes (those go to /perception/lanes).
# header.stamp = stamp of the /carla/ego/rgb image it was computed from (sim time).
std_msgs/Header header
int64 frame_id                 # PerceptionFrame.frame_id
TrackedObject[] objects
int32 lead_track_id            # PerceptionFrame.lead_object() track id, -1 if none
float32 total_ms               # pipeline time of this frame (timings_ms.total_ms)
