"""
PID Controller

components:
    follow attitude commands
    gps commands and yaw
    waypoint following
"""
import getopt
import numpy as np

DRONE_MASS_KG = 0.5
GRAVITY = -9.81
MOI = np.array([0.005, 0.005, 0.01])
MAX_THRUST = 10.0
MAX_TORQUE = 1.0

X_K_P = 5.0
X_K_D = 2.0
Y_K_P = 0.4
Y_K_D = 6.0
ROLL_K_P = 4.0
PITCH_K_P = 5.0
YAW_K_P = 1.9

Z_K_P = 24.0
Z_K_D = 6.0
Z_OFFSET = 0.2

try:
    opts, _ = getopt.getopt(sys.argv[1:], 'a:b:c:d:e:f:g:h:i:',
                            [ 'x_k_p=', 'x_k_d=',
                              'y_k_p=', 'y_k_d=',
                              'z_k_p=', 'z_k_d=',
                              'roll_k_p=', 'pitch_k_p=',
                              'yaw_k_p=',])
    for opt, arg in opts:
        if opt == '--x_k_p':
            X_K_P = float(arg)
        elif opt == '--x_k_d':
            X_K_D = float(arg)
        elif opt == '--y_k_p':
            Y_K_P = float(arg)
        elif opt == '--y_k_d':
            Y_K_D = float(arg)
        elif opt == '--roll_k_p':
            ROLL_K_P = float(arg)
        elif opt == '--pitch_k_p':
            PITCH_K_P = float(arg)
        elif opt == '--yaw_k_p':
            YAW_K_P = float(arg)
        elif opt == '--z_k_p':
            Z_K_P = float(arg)
        elif opt == '--z_k_d':
            Z_K_D = float(arg)
except getopt.GetoptError as e:
    print(e)
    exit(1)

def rotational_matrix(phi, theta, psi):
    c_phi = np.cos(phi)
    s_phi = np.sin(phi)
    r_x = np.matrix([[1, 0, 0],
                     [0, c_phi, -s_phi],
                     [0, s_phi, c_phi]])

    c_theta = np.cos(theta)
    s_theta = np.sin(theta)
    r_y = np.matrix([[c_theta, 0, s_theta],
                     [0, 1, 0],
                     [-s_theta, 0, c_theta]])

    #psi = np.pi - psi
    c_psi = np.cos(psi)
    s_psi = np.sin(psi)
    r_z = np.matrix([[c_psi, -s_psi, 0],
                     [s_psi, c_psi, 0],
                     [0, 0, 1]])
    # TODO replace with your own implementation
    #   according to the math above
    #
    # return rotation_matrix

    return np.asarray(np.matmul(r_z, np.matmul(r_y, r_x)))

class NonlinearController(object):

    def __init__(self,
                 x_k_p=X_K_P,
                 x_k_d=X_K_D,
                 y_k_p=Y_K_P,
                 y_k_d=Y_K_D,
                 z_k_p=Z_K_P,
                 z_k_d=Z_K_D,
                 k_p_roll=ROLL_K_P,
                 k_p_pitch=PITCH_K_P,
                 k_p_yaw=YAW_K_P,
                 k_p_p=0.1075,
                 k_p_q=0.105,
                 k_p_r=0.1):
        """Initialize the controller object and control gains"""
        self.k_p_p = k_p_p
        self.k_p_q = k_p_q
        self.k_p_r = k_p_r

        self.x_k_p = x_k_p
        self.x_k_d = x_k_d
        self.y_k_p = y_k_p
        self.y_k_d = y_k_d
        self.z_k_p = z_k_p
        self.z_k_d = z_k_d

        self.k_p_roll = k_p_roll
        self.k_p_pitch = k_p_pitch
        self.k_p_yaw = k_p_yaw

        self.last_lateral_call = None
        self.accumulated_lateral_error = np.array([0.0, 0.0])

        self.last_altitude_call = None
        self.accumulated_altitude_error = np.array([0.0, 0.0])
        return

    def trajectory_control(self, position_trajectory, yaw_trajectory, time_trajectory, current_time):
        """Generate a commanded position, velocity and yaw based on the trajectory

        Args:
            position_trajectory: list of 3-element numpy arrays, NED positions
            yaw_trajectory: list yaw commands in radians
            time_trajectory: list of times (in seconds) that correspond to the position and yaw commands
            current_time: float corresponding to the current time in seconds

        Returns: tuple (commanded position, commanded velocity, commanded yaw)

        """

        ind_min = np.argmin(np.abs(np.array(time_trajectory) - current_time))
        time_ref = time_trajectory[ind_min]


        if current_time < time_ref:
            position0 = position_trajectory[ind_min - 1]
            position1 = position_trajectory[ind_min]

            time0 = time_trajectory[ind_min - 1]
            time1 = time_trajectory[ind_min]
            yaw_cmd = yaw_trajectory[ind_min - 1]

        else:
            yaw_cmd = yaw_trajectory[ind_min]
            if ind_min >= len(position_trajectory) - 1:
                position0 = position_trajectory[ind_min]
                position1 = position_trajectory[ind_min]

                time0 = 0.0
                time1 = 1.0
            else:

                position0 = position_trajectory[ind_min]
                position1 = position_trajectory[ind_min + 1]
                time0 = time_trajectory[ind_min]
                time1 = time_trajectory[ind_min + 1]

        position_cmd = (position1 - position0) * \
                        (current_time - time0) / (time1 - time0) + position0
        velocity_cmd = (position1 - position0) / (time1 - time0)

        result = position_cmd, velocity_cmd, yaw_cmd
        #print(yaw_cmd)
        return result

    def lateral_position_control(self, local_position_cmd, local_velocity_cmd, local_position, local_velocity,
                               acceleration_ff = np.array([0.0, 0.0])):
        """Generate horizontal acceleration commands for the vehicle in the local frame

        Args:
            local_position_cmd: desired 2D position in local frame [north, east]
            local_velocity_cmd: desired 2D velocity in local frame [north_velocity, east_velocity]
            local_position: vehicle position in the local frame [north, east]
            local_velocity: vehicle velocity in the local frame [north_velocity, east_velocity]
            acceleration_cmd: feedforward acceleration command

        Returns: desired vehicle 2D acceleration in the local frame [north, east]
        """
        #return np.array([0.0, 0.0])

        xy_error = local_position_cmd - local_position
        xy_dot_error = local_velocity_cmd - local_velocity

        xy_k_p = np.array([self.x_k_p, self.y_k_p])
        xy_k_d = np.array([self.y_k_d, self.y_k_d])

        u_bar_1 = xy_k_p * xy_error + xy_k_d * xy_dot_error + acceleration_ff
        return u_bar_1

    def altitude_control(self, altitude_cmd, vertical_velocity_cmd,
                         altitude, vertical_velocity, attitude, acceleration_ff=0.0):
        """Generate vertical acceleration (thrust) command

        Args:
            altitude_cmd: desired vertical position (+up)
            vertical_velocity_cmd: desired vertical velocity (+up)
            altitude: vehicle vertical position (+up)
            vertical_velocity: vehicle vertical velocity (+up)
            attitude: the vehicle's current attitude, 3 element numpy array (roll, pitch, yaw) in radians
            acceleration_ff: feedforward acceleration command (+up)

        Returns: thrust command for the vehicle (+up)
        """
        z_error = altitude_cmd - altitude + Z_OFFSET
        z_dot_error = vertical_velocity_cmd - vertical_velocity

        u_bar_1 = self.z_k_p * z_error + self.z_k_d * z_dot_error + acceleration_ff

        result = (u_bar_1 + GRAVITY) / rotational_matrix(*attitude)[2, 2]
        return result


    def roll_pitch_controller(self, acceleration_cmd, attitude, thrust_cmd):
        """ Generate the rollrate and pitchrate commands in the body frame

        Args:
            target_acceleration: 2-element numpy array (north_acceleration_cmd,east_acceleration_cmd) in m/s^2
            attitude: 3-element numpy array (roll, pitch, yaw) in radians
            thrust_cmd: vehicle thrust command in Newton

        Returns: 2-element numpy array, desired rollrate (p) and pitchrate (q) commands in radians/s
        """
        #return np.array([0.0, 0.0])

        collective_accel = -thrust_cmd / DRONE_MASS_KG

        xy_accel = np.linalg.norm(acceleration_cmd)
        abs_col_accel = np.abs(collective_accel)
        if xy_accel > abs_col_accel:
            print("collective_accel = ", collective_accel, "xy_accel = ", xy_accel)
            print("desired_accel = ", acceleration_cmd)
            acceleration_cmd = acceleration_cmd * abs_col_accel / xy_accel

        b_target = acceleration_cmd / collective_accel

        np.clip(b_target, -0.99, 0.99)

        rot_mat = rotational_matrix(*attitude)
        b_actual = np.array([rot_mat[0, 2], rot_mat[1, 2]])

        k_p = np.array([self.k_p_roll, self.k_p_pitch])
        b_dot = ((b_target - b_actual) * k_p)[np.newaxis].T  # Turn b_dot into a column vector.

        r22 = np.matrix([rot_mat[1, 0:2], -rot_mat[0, 0:2]]).T
        pq = np.matmul(r22, b_dot) / rot_mat[2, 2]

        result = np.ravel(pq)
        return result


    def body_rate_control(self, body_rate_cmd, body_rate):
        """ Generate the roll, pitch, yaw moment commands in the body frame

        Args:
            body_rate_cmd: 3-element numpy array (p_cmd,q_cmd,r_cmd) in radians/second^2
            body_rate: 3-element numpy array (p,q,r) in radians/second^2

        Returns: 3-element numpy array, desired roll moment, pitch moment, and yaw moment commands in Newtons*meters
        """
        cmd = (body_rate_cmd - body_rate) * np.array([self.k_p_p, self.k_p_q, self.k_p_r]) # * MOI
        return cmd

    def yaw_control(self, yaw_cmd, yaw):
        """ Generate the target yawrate

        Args:
            yaw_cmd: desired vehicle yaw in radians
            yaw: vehicle yaw in radians

        Returns: target yawrate in radians/sec
        """
        yaw_error = yaw_cmd - yaw
        if yaw_error > np.pi: yaw_error -= 2 * np.pi
        elif yaw_error < -np.pi: yaw_error += 2 * np.pi

        yaw_rate = self.k_p_yaw * yaw_error
        return yaw_rate
