# Longitudinal command from control_node to carla_bridge_node.
# header.stamp = stamp of the /perception/objects message it reacts to; the bridge ticks the
# simulator once the control for the current frame has arrived (or after a timeout).
std_msgs/Header header
bool override_autopilot        # false: the Traffic Manager autopilot drives (throttle/brake)
float32 throttle               # [0, 1], used only when override_autopilot is true
float32 brake                  # [0, 1], used only when override_autopilot is true
string mode                    # AUTOPILOT | ACC | GAP | AEB | HOLD (perception.control.MODES)
string reason
int32 lead_track_id            # -1 if no confirmed lead
float32 lead_distance_m        # NaN if no confirmed lead
float32 lead_ttc_s             # NaN if no lead or no TTC
float32 ego_speed_mps
