# One fused object: perception.types.ObjectState (+ its Track and Detection).
# Field-for-field the dict of perception.pipeline.object_to_dict().
int32 track_id
int32 cls                      # class index (configs/data_bdd.yaml order)
string cls_name
float32 score
float32[4] bbox_xyxy           # pixels in the RGB image: x1, y1, x2, y2
int32 age                      # frames since the track started
float32 distance_m             # filtered longitudinal distance to the ground-contact point
float32 distance_std           # 1-sigma, metres
float32 closing_speed_mps      # positive = getting closer
bool has_ttc                   # false: ObjectState.ttc_s is None (not closing significantly)
float32 ttc_s                  # time to collision, NaN when has_ttc is false
bool in_ego_lane
