import numpy as np
from helperfunctions import add_pose_from_global, add_landmark_measurement_from_global
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_pose(graph, initial_estimate, pose_5):
    # Adding the initial estimate for the 5th pose using our helper function `add_pose_from_global` which also adds the odometry factor between X(4) and X(5).
    pose_4 = initial_estimate.atPose2(X(4))
    graph, initial_estimate = add_pose_from_global(
        graph=graph,
        initial_estimate=initial_estimate,
        prev_key=X(4),
        new_key=X(5),
        prev_pose=pose_4,
        new_pose_global=pose_5,
        odom_noise=ODOMETRY_NOISE
    )
    return graph, initial_estimate

def add_landmark_measurement(graph, result, pose_5, landmark):
    # Adding the measurement from X(5) to the chosen landmark using our helper function `add_landmark_measurement_from_global` which calculates the correct bearing and range from the global poses.``
    landmark_point = result.atPoint2(L(landmark))
    graph = add_landmark_measurement_from_global(
        graph=graph,
        pose_key=X(5),
        pose=pose_5,
        landmark_key=L(landmark),
        landmark_point=landmark_point,
        measurement_noise=MEASUREMENT_NOISE
    )
    return graph

def optimize(graph, initial_estimate):
    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
    result = optimizer.optimize()
    print(result)
    return result

def minimize_marginals(graph, initial_estimate, pose_options):
    best_pose = "d"
    best_landmark = 1
    pose_5 = pose_options[best_pose]
    graph, initial_estimate = add_pose(graph, initial_estimate, pose_5)
    result = optimize(graph, initial_estimate)
    graph = add_landmark_measurement(graph, result, pose_5, best_landmark)
    result = optimize(graph, initial_estimate)

    marginals = gtsam.Marginals(graph, result)
    sum_of_marginals = (
        marginals.marginalCovariance(L(1)).sum()
        + marginals.marginalCovariance(L(2)).sum()
    )
    return best_pose, best_landmark, sum_of_marginals

def minimize_errors(graph, initial_estimate, pose_options):
    best_pose = "b"
    best_landmark = 2
    pose_5 = pose_options[best_pose]
    graph, initial_estimate = add_pose(graph, initial_estimate, pose_5)
    result = optimize(graph, initial_estimate)
    graph = add_landmark_measurement(graph, result, pose_5, best_landmark)
    result = optimize(graph, initial_estimate)

    list_of_errors = [graph.error(result)]
    sum_of_errors = sum(list_of_errors)
    return best_pose, best_landmark, sum_of_errors