# perception.types.LaneResult. Boundaries are x = a*y^2 + b*y + c in image pixels.
# The ego mask itself is not sent (has_ego_mask only); the debug image shows it.
std_msgs/Header header
bool has_left
float32[3] left_coeffs         # a, b, c (zeros when has_left is false)
bool has_right
float32[3] right_coeffs
float32 confidence
bool has_ego_mask
float32 y_top                  # image rows the fit measured (NaN if unknown)
float32 y_bottom
