
import cv2
import numpy as np


def region_of_interest(image):
    """
    Keep the lower road area and mask the rest.
    Works with grayscale or color images.
    """

    height, width = image.shape[:2]

    mask = np.zeros_like(image)

    polygon = np.array([[
        (int(width * 0.05), height),
        (int(width * 0.38), int(height * 0.48)),
        (int(width * 0.62), int(height * 0.48)),
        (int(width * 0.95), height)
    ]], dtype=np.int32)

    if len(image.shape) == 3:
        mask_color = (255, 255, 255)
    else:
        mask_color = 255

    cv2.fillPoly(mask, polygon, mask_color)

    masked_image = cv2.bitwise_and(image, mask)

    return masked_image


def average_lane_lines(image, lines):
    """
    Separate detected Hough lines into left and right lanes.
    """

    height, width = image.shape[:2]

    left_lines = []
    right_lines = []

    if lines is None:
        return None, None

    for line in lines.reshape(-1, 4):

        x1, y1, x2, y2 = line

        if x2 == x1:
            continue

        slope = (y2 - y1) / (x2 - x1)

        # Ignore nearly horizontal lines.
        if abs(slope) < 0.35:
            continue

        # Ignore nearly vertical noise.
        if abs(slope) > 3.0:
            continue

        length = np.sqrt(
            (x2 - x1) ** 2 + (y2 - y1) ** 2
        )

        if slope < 0:
            left_lines.append(
                (slope, x1, y1, x2, y2, length)
            )
        else:
            right_lines.append(
                (slope, x1, y1, x2, y2, length)
            )

    def fit_lane(lane_lines):

        if not lane_lines:
            return None

        slopes = np.array([
            line[0] for line in lane_lines
        ])

        x_values = np.array([
            line[1] for line in lane_lines
        ])

        y_values = np.array([
            line[2] for line in lane_lines
        ])

        weights = np.array([
            line[5] for line in lane_lines
        ])

        slope = np.average(
            slopes,
            weights=weights
        )

        x_mean = np.average(
            x_values,
            weights=weights
        )

        y_mean = np.average(
            y_values,
            weights=weights
        )

        intercept = y_mean - slope * x_mean

        y_bottom = height
        y_top = int(height * 0.52)

        x_bottom = int(
            (y_bottom - intercept) / slope
        )

        x_top = int(
            (y_top - intercept) / slope
        )

        return (
            x_bottom,
            y_bottom,
            x_top,
            y_top
        )

    left_lane = fit_lane(left_lines)
    right_lane = fit_lane(right_lines)

    return left_lane, right_lane


def draw_lane_line(image, lane, color, thickness=8):

    if lane is None:
        return

    x1, y1, x2, y2 = lane

    cv2.line(
        image,
        (x1, y1),
        (x2, y2),
        color,
        thickness
    )


def process_frame(frame):
    """
    Process one frame and return:
        output,
        stages dictionary,
        lane status,
        lane offset
    """

    height, width = frame.shape[:2]

    # -------------------------------------------------
    # 1. GRAYSCALE
    # -------------------------------------------------

    grayscale = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # -------------------------------------------------
    # 2. GAUSSIAN BLUR
    # -------------------------------------------------

    blurred = cv2.GaussianBlur(
        grayscale,
        (5, 5),
        0
    )

    # -------------------------------------------------
    # 3. CANNY EDGE DETECTION
    # -------------------------------------------------

    edges = cv2.Canny(
        blurred,
        50,
        150
    )

    # -------------------------------------------------
    # 4. REGION OF INTEREST
    # -------------------------------------------------

    roi = region_of_interest(edges)

    # -------------------------------------------------
    # 5. HOUGH LINE DETECTION
    # -------------------------------------------------

    hough_lines = cv2.HoughLinesP(
        roi,
        rho=1,
        theta=np.pi / 180,
        threshold=25,
        minLineLength=25,
        maxLineGap=100
    )

    hough_visualization = cv2.cvtColor(
        roi,
        cv2.COLOR_GRAY2BGR
    )

    if hough_lines is not None:

        for line in hough_lines.reshape(-1, 4):

            x1, y1, x2, y2 = line

            cv2.line(
                hough_visualization,
                (x1, y1),
                (x2, y2),
                (0, 255, 255),
                2
            )

    # -------------------------------------------------
    # 6. FIT LEFT AND RIGHT LANE BOUNDARIES
    # -------------------------------------------------

    left_lane, right_lane = average_lane_lines(
        frame,
        hough_lines
    )

    # -------------------------------------------------
    # 7. FINAL LANE VISUALIZATION
    # -------------------------------------------------

    output = frame.copy()

    lane_overlay = np.zeros_like(frame)

    draw_lane_line(
        lane_overlay,
        left_lane,
        (0, 255, 0),
        8
    )

    draw_lane_line(
        lane_overlay,
        right_lane,
        (0, 255, 0),
        8
    )

    # Fill lane area only when both boundaries exist.
    if left_lane is not None and right_lane is not None:

        lane_polygon = np.array([[
            (left_lane[0], left_lane[1]),
            (left_lane[2], left_lane[3]),
            (right_lane[2], right_lane[3]),
            (right_lane[0], right_lane[1])
        ]], dtype=np.int32)

        cv2.fillPoly(
            lane_overlay,
            lane_polygon,
            (0, 100, 0)
        )

    output = cv2.addWeighted(
        output,
        0.85,
        lane_overlay,
        0.65,
        0
    )

    # -------------------------------------------------
    # 8. LANE CENTER AND OFFSET
    # -------------------------------------------------

    lane_offset = None

    if left_lane is not None and right_lane is not None:

        left_bottom_x = left_lane[0]
        right_bottom_x = right_lane[0]

        lane_center = int(
            (left_bottom_x + right_bottom_x) / 2
        )

        vehicle_center = width // 2

        lane_offset = float(
            vehicle_center - lane_center
        )

        # Lane center marker
        cv2.circle(
            output,
            (lane_center, height - 25),
            8,
            (255, 0, 0),
            -1
        )

        # Vehicle center reference
        cv2.line(
            output,
            (vehicle_center, height - 50),
            (vehicle_center, height),
            (0, 255, 255),
            3
        )

        lane_status = "BOTH LANE BOUNDARIES DETECTED"

    elif left_lane is not None:

        lane_status = "LEFT LANE BOUNDARY DETECTED"

    elif right_lane is not None:

        lane_status = "RIGHT LANE BOUNDARY DETECTED"

    else:

        lane_status = "LANE NOT DETECTED"

    # -------------------------------------------------
    # 9. DISPLAY STATUS ON FINAL OUTPUT
    # -------------------------------------------------

    cv2.rectangle(
        output,
        (0, 0),
        (width, 65),
        (20, 20, 20),
        -1
    )

    cv2.putText(
        output,
        "LANE DETECTION SYSTEM",
        (15, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        output,
        lane_status,
        (15, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2
    )

    # -------------------------------------------------
    # RETURN ALL STAGES
    # -------------------------------------------------

    stages = {
        "grayscale": grayscale,
        "blurred": blurred,
        "edges": edges,
        "roi": roi,
        "hough": hough_visualization,
        "final": output
    }

    return output, stages, lane_status, lane_offset
