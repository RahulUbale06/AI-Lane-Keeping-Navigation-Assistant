
import os
import sys
import cv2
import pygame
import numpy as np

from lane_detection import process_frame
from yolo_detector import ObjectDetector


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VIDEO_SOURCE = os.path.join(
    BASE_DIR,
    "videos",
    "mixkit-cars-driving-by-on-road-2022-hd-ready.mp4"
)

SCREEN_W = 1600
SCREEN_H = 900
FPS = 30

PANEL_W = 850
GAME_X = 870

ROAD_LEFT = 1010
ROAD_RIGHT = 1540
ROAD_TOP = 125
ROAD_BOTTOM = 790

LANE_WIDTH = (ROAD_RIGHT - ROAD_LEFT) / 3

LANE_CENTERS = [
    ROAD_LEFT + LANE_WIDTH * 0.5,
    ROAD_LEFT + LANE_WIDTH * 1.5,
    ROAD_LEFT + LANE_WIDTH * 2.5
]

CAR_Y = 650
CAR_W = 55
CAR_H = 90

MAX_SPEED = 12

NEAR_BOTTOM_RATIO = 0.82
MIN_BOX_AREA_RATIO = 0.025


# =========================================================
# INITIALIZE PYGAME
# =========================================================

pygame.init()

screen = pygame.display.set_mode(
    (SCREEN_W, SCREEN_H)
)

pygame.display.set_caption(
    "Lane Keeping and Navigation Assistant | V7.2"
)

clock = pygame.time.Clock()

font = pygame.font.SysFont("Arial", 18)
small_font = pygame.font.SysFont("Arial", 14)
title_font = pygame.font.SysFont("Arial", 23, bold=True)


# =========================================================
# OPEN VIDEO
# =========================================================

print("Video path:", VIDEO_SOURCE)
print("File exists:", os.path.exists(VIDEO_SOURCE))

camera = cv2.VideoCapture(VIDEO_SOURCE)

if not camera.isOpened():

    print("ERROR: Could not open video.")
    pygame.quit()
    sys.exit()


# =========================================================
# YOLO
# =========================================================

print("Loading YOLO model...")

detector = ObjectDetector()

print("YOLO loaded.")


# =========================================================
# STATE
# =========================================================

running = True

player_lane = 1
car_x = float(LANE_CENTERS[player_lane])

game_speed = 5.0
simulation_speed = 0
road_offset = 0

autopilot = True

virtual_obstacle = False
obstacle_lane = None

avoidance_lane = None
avoidance_active = False

lane_status = "WAITING"
navigation_decision = "INITIALIZING"
current_offset = None

obstacle_warning = False

frame_count = 0
cached_detections = []

current_frame = None
stages = {}

game_message = "AUTOPILOT ACTIVE"
game_message_color = (100, 255, 150)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def draw_text(
    text,
    x,
    y,
    color=(255, 255, 255),
    selected_font=font
):

    surface = selected_font.render(
        str(text),
        True,
        color
    )

    screen.blit(surface, (x, y))


def convert_frame(frame, size):

    if len(frame.shape) == 2:

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_GRAY2BGR
        )

    resized = cv2.resize(
        frame,
        size
    )

    rgb = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2RGB
    )

    rgb = np.transpose(
        rgb,
        (1, 0, 2)
    )

    return pygame.surfarray.make_surface(rgb)


def detect_near_object(objects, frame_shape):

    height, width = frame_shape[:2]
    frame_area = height * width

    for obj in objects:

        x1, y1, x2, y2 = obj["box"]

        box_area = max(
            0,
            x2 - x1
        ) * max(
            0,
            y2 - y1
        )

        area_ratio = box_area / frame_area
        bottom_ratio = y2 / height

        center_x = (x1 + x2) / 2

        center_region = (
            width * 0.25
            <= center_x
            <= width * 0.75
        )

        if (
            center_region
            and bottom_ratio >= NEAR_BOTTOM_RATIO
            and area_ratio >= MIN_BOX_AREA_RATIO
        ):

            return True

    return False


# =========================================================
# RESET
# =========================================================

def reset_game():

    global player_lane
    global car_x
    global game_speed
    global road_offset
    global virtual_obstacle
    global obstacle_lane
    global avoidance_lane
    global avoidance_active
    global obstacle_warning
    global simulation_speed
    global navigation_decision
    global game_message
    global game_message_color

    player_lane = 1
    car_x = float(LANE_CENTERS[player_lane])

    game_speed = 5
    road_offset = 0

    virtual_obstacle = False
    obstacle_lane = None

    avoidance_lane = None
    avoidance_active = False

    obstacle_warning = False
    simulation_speed = 0

    navigation_decision = "RESET"

    game_message = "GAME RESET"
    game_message_color = (100, 255, 150)


# =========================================================
# SELECT SAFE ADJACENT LANE
# =========================================================

def choose_avoidance_lane():

    global avoidance_lane
    global avoidance_active
    global game_message
    global game_message_color

    # Try right first, then left.
    candidates = [
        player_lane + 1,
        player_lane - 1
    ]

    for lane in candidates:

        if lane < 0 or lane > 2:
            continue

        if lane == obstacle_lane:
            continue

        avoidance_lane = lane
        avoidance_active = True

        game_message = (
            f"AVOIDING OBSTACLE - LANE {lane + 1}"
        )

        game_message_color = (100, 255, 150)

        return True

    avoidance_lane = None
    avoidance_active = False

    game_message = "NO SAFE LANE - STOP"
    game_message_color = (255, 80, 80)

    return False


# =========================================================
# NAVIGATION
# =========================================================

def update_navigation():

    global navigation_decision
    global simulation_speed
    global obstacle_warning
    global game_speed
    global game_message
    global game_message_color
    global car_x
    global player_lane
    global avoidance_lane
    global avoidance_active

    yolo_warning = False

    if current_frame is not None:

        yolo_warning = detect_near_object(
            cached_detections,
            current_frame.shape
        )

    # YOLO warning: brake.
    # It does not select a simulated lane in this version.
    if yolo_warning:

        obstacle_warning = True

        navigation_decision = "YOLO OBSTACLE WARNING"

        simulation_speed = 0
        game_speed = 0

        game_message = "YOLO WARNING - BRAKING"
        game_message_color = (255, 90, 90)

        return

    # -----------------------------------------------------
    # VIRTUAL GAME OBSTACLE
    # -----------------------------------------------------

    if virtual_obstacle:

        obstacle_warning = True

        if not autopilot:

            navigation_decision = "OBSTACLE - MANUAL MODE"

            simulation_speed = 0
            game_speed = 0

            game_message = "SWITCH TO AUTOPILOT"
            game_message_color = (255, 90, 90)

            return

        if not avoidance_active:

            if not choose_avoidance_lane():

                navigation_decision = "NO SAFE LANE"

                simulation_speed = 0
                game_speed = 0

                return

        target_x = LANE_CENTERS[avoidance_lane]

        car_x += (
            target_x - car_x
        ) * 0.08

        navigation_decision = "AUTOMATIC LANE CHANGE"

        game_speed = min(
            game_speed + 0.1,
            MAX_SPEED
        )

        simulation_speed = int(
            game_speed * 5
        )

        if abs(car_x - target_x) < 4:

            car_x = target_x
            player_lane = avoidance_lane

            navigation_decision = "OBSTACLE AVOIDED"

            game_message = "OBSTACLE AVOIDED"
            game_message_color = (100, 255, 150)

        return

    # -----------------------------------------------------
    # NO OBSTACLE
    # -----------------------------------------------------

    obstacle_warning = False

    avoidance_active = False
    avoidance_lane = None

    # -----------------------------------------------------
    # AUTOPILOT LANE KEEPING
    # -----------------------------------------------------

    if autopilot:

        if current_offset is None:

            navigation_decision = "LANE NOT DETECTED"

            simulation_speed = 0
            game_speed = 0

            game_message = "LANE NOT DETECTED"
            game_message_color = (255, 180, 80)

            return

        threshold = 960 * 0.04

        if current_offset > threshold:

            navigation_decision = "STEER LEFT"

        elif current_offset < -threshold:

            navigation_decision = "STEER RIGHT"

        else:

            navigation_decision = "MAINTAIN LANE"

        target_x = (
            LANE_CENTERS[player_lane]
            - np.clip(
                current_offset * 0.25,
                -LANE_WIDTH * 0.35,
                LANE_WIDTH * 0.35
            )
        )

        target_x = np.clip(
            target_x,
            ROAD_LEFT + CAR_W,
            ROAD_RIGHT - CAR_W
        )

        car_x += (
            target_x - car_x
        ) * 0.05

        game_speed = min(
            game_speed + 0.15,
            MAX_SPEED
        )

        simulation_speed = int(
            game_speed * 5
        )

        game_message = "AUTOPILOT ACTIVE"
        game_message_color = (100, 255, 150)

    else:

        navigation_decision = "MANUAL DRIVING"

        game_speed = min(
            game_speed + 0.1,
            MAX_SPEED
        )

        simulation_speed = int(
            game_speed * 5
        )

        game_message = "MANUAL MODE"
        game_message_color = (100, 200, 255)


# =========================================================
# COMPUTER VISION PIPELINE DISPLAY
# =========================================================

def draw_stage_panel(
    title,
    image,
    x,
    y,
    width,
    height
):

    pygame.draw.rect(
        screen,
        (32, 40, 58),
        (x, y, width, height),
        border_radius=8
    )

    draw_text(
        title,
        x + 8,
        y + 5,
        (255, 220, 0),
        small_font
    )

    if image is not None:

        surface = convert_frame(
            image,
            (width - 12, height - 32)
        )

        screen.blit(
            surface,
            (x + 6, y + 26)
        )


def draw_cv_pipeline(frame, stages):

    # Pipeline title
    draw_text(
        "COMPUTER VISION PIPELINE",
        20,
        12,
        (255, 220, 0),
        title_font
    )

    panel_w = 202
    panel_h = 145

    x_positions = [
        20,
        230,
        440,
        650
    ]

    y_top = 50
    y_bottom = 205

    draw_stage_panel(
        "1. ORIGINAL FRAME",
        frame,
        x_positions[0],
        y_top,
        panel_w,
        panel_h
    )

    draw_stage_panel(
        "2. GRAYSCALE",
        stages.get("grayscale"),
        x_positions[1],
        y_top,
        panel_w,
        panel_h
    )

    draw_stage_panel(
        "3. GAUSSIAN BLUR",
        stages.get("blurred"),
        x_positions[2],
        y_top,
        panel_w,
        panel_h
    )

    draw_stage_panel(
        "4. CANNY EDGES",
        stages.get("edges"),
        x_positions[3],
        y_top,
        panel_w,
        panel_h
    )

    draw_stage_panel(
        "5. ROI MASK",
        stages.get("roi"),
        x_positions[0],
        y_bottom,
        panel_w,
        panel_h
    )

    draw_stage_panel(
        "6. HOUGH LINES",
        stages.get("hough"),
        x_positions[1],
        y_bottom,
        panel_w,
        panel_h
    )

    draw_stage_panel(
        "7. FINAL LANE DETECTION",
        stages.get("final"),
        x_positions[2],
        y_bottom,
        panel_w,
        panel_h
    )

    # YOLO result is the eighth panel.
    yolo_image = stages.get("yolo")

    draw_stage_panel(
        "8. YOLO OBJECT DETECTION",
        yolo_image,
        x_positions[3],
        y_bottom,
        panel_w,
        panel_h
    )


# =========================================================
# NAVIGATION DASHBOARD
# =========================================================

def draw_dashboard():

    pygame.draw.rect(
        screen,
        (30, 38, 55),
        (20, 370, 830, 500),
        border_radius=12
    )

    draw_text(
        "NAVIGATION ANALYSIS",
        40,
        390,
        (255, 220, 0),
        title_font
    )

    offset_text = (
        f"{current_offset:.2f} px"
        if current_offset is not None
        else "Unavailable"
    )

    draw_text(
        f"Lane Status: {lane_status}",
        40,
        445
    )

    draw_text(
        f"Navigation: {navigation_decision}",
        40,
        485,
        (255, 220, 0)
    )

    draw_text(
        f"Lane Offset: {offset_text}",
        40,
        525
    )

    draw_text(
        f"Simulation Speed: {simulation_speed} km/h",
        40,
        565
    )

    draw_text(
        f"YOLO Objects: {len(cached_detections)}",
        40,
        605
    )

    draw_text(
        f"Virtual Obstacle: {'ACTIVE' if virtual_obstacle else 'INACTIVE'}",
        40,
        645
    )

    draw_text(
        "CONTROLS",
        440,
        445,
        (0, 255, 130)
    )

    draw_text(
        "O - Toggle obstacle",
        440,
        485,
        (200, 200, 200),
        small_font
    )

    draw_text(
        "P - Autopilot/manual",
        440,
        515,
        (200, 200, 200),
        small_font
    )

    draw_text(
        "A/D or arrows - Change lane",
        440,
        545,
        (200, 200, 200),
        small_font
    )

    draw_text(
        "R - Reset simulation",
        440,
        575,
        (200, 200, 200),
        small_font
    )

    draw_text(
        "ESC - Exit",
        440,
        605,
        (200, 200, 200),
        small_font
    )


# =========================================================
# DRAW SIMULATED CAR
# =========================================================

def draw_car(x, y):

    x = int(x)

    # Wheels
    for wheel_x in (x - 29, x + 20):

        pygame.draw.rect(
            screen,
            (12, 12, 12),
            (wheel_x, y + 15, 9, 25),
            border_radius=3
        )

        pygame.draw.rect(
            screen,
            (12, 12, 12),
            (wheel_x, y + 55, 9, 25),
            border_radius=3
        )

    # Car body
    pygame.draw.rect(
        screen,
        (25, 125, 255),
        (
            x - CAR_W // 2,
            y,
            CAR_W,
            CAR_H
        ),
        border_radius=12
    )

    # Windows
    pygame.draw.rect(
        screen,
        (150, 220, 255),
        (x - 19, y + 15, 38, 25),
        border_radius=5
    )

    pygame.draw.rect(
        screen,
        (150, 220, 255),
        (x - 19, y + 55, 38, 20),
        border_radius=5
    )

    # Headlights
    pygame.draw.circle(
        screen,
        (255, 230, 50),
        (x - 20, y + 5),
        5
    )

    pygame.draw.circle(
        screen,
        (255, 230, 50),
        (x + 20, y + 5),
        5
    )


# =========================================================
# DRAW RIGHT-SIDE GAME
# =========================================================

def draw_game():

    global road_offset

    pygame.draw.rect(
        screen,
        (18, 22, 32),
        (GAME_X, 0, SCREEN_W - GAME_X, SCREEN_H)
    )

    draw_text(
        "LIVE VEHICLE SIMULATION",
        GAME_X + 25,
        25,
        (255, 220, 0),
        title_font
    )

    # Road
    pygame.draw.rect(
        screen,
        (55, 60, 70),
        (
            ROAD_LEFT,
            ROAD_TOP,
            ROAD_RIGHT - ROAD_LEFT,
            ROAD_BOTTOM - ROAD_TOP
        )
    )

    # Road boundaries
    for edge_x in (ROAD_LEFT, ROAD_RIGHT):

        pygame.draw.line(
            screen,
            (255, 255, 255),
            (edge_x, ROAD_TOP),
            (edge_x, ROAD_BOTTOM),
            5
        )

    # Animated lane markings
    if simulation_speed > 0:

        road_offset += simulation_speed

        if road_offset >= 80:
            road_offset = 0

    for lane in range(1, 3):

        lane_x = int(
            ROAD_LEFT + lane * LANE_WIDTH
        )

        for y in range(
            ROAD_TOP - 80,
            ROAD_BOTTOM + 80,
            80
        ):

            line_y = int(
                y + road_offset
            )

            if ROAD_TOP <= line_y <= ROAD_BOTTOM:

                pygame.draw.line(
                    screen,
                    (235, 235, 235),
                    (lane_x, line_y),
                    (lane_x, line_y + 40),
                    4
                )

    # Obstacle
    if virtual_obstacle and obstacle_lane is not None:

        obstacle_x = int(
            LANE_CENTERS[obstacle_lane]
        )

        pygame.draw.rect(
            screen,
            (230, 45, 45),
            (
                obstacle_x - 30,
                330,
                60,
                70
            ),
            border_radius=8
        )

        draw_text(
            "OBSTACLE",
            obstacle_x - 40,
            300,
            (255, 100, 100),
            small_font
        )

    # Car
    draw_car(
        car_x,
        CAR_Y
    )

    draw_text(
        game_message,
        GAME_X + 25,
        815,
        game_message_color,
        small_font
    )

    draw_text(
        f"Speed: {simulation_speed} km/h",
        GAME_X + 25,
        840,
        (220, 220, 220),
        small_font
    )


# =========================================================
# MAIN LOOP
# =========================================================

try:

    while running:

        clock.tick(FPS)

        # ---------------------------------------------
        # EVENTS
        # ---------------------------------------------

        for event in pygame.event.get():

            if event.type == pygame.QUIT:

                running = False

            if event.type == pygame.KEYDOWN:

                if event.key == pygame.K_ESCAPE:

                    running = False

                elif event.key == pygame.K_o:

                    virtual_obstacle = not virtual_obstacle

                    if virtual_obstacle:

                        obstacle_lane = player_lane

                        avoidance_lane = None
                        avoidance_active = False

                    else:

                        obstacle_lane = None

                        avoidance_lane = None
                        avoidance_active = False

                elif event.key == pygame.K_r:

                    reset_game()

                elif event.key == pygame.K_p:

                    autopilot = not autopilot

                elif event.key in (
                    pygame.K_LEFT,
                    pygame.K_a
                ):

                    if not autopilot and not virtual_obstacle:

                        player_lane = max(
                            0,
                            player_lane - 1
                        )

                        car_x = float(
                            LANE_CENTERS[player_lane]
                        )

                elif event.key in (
                    pygame.K_RIGHT,
                    pygame.K_d
                ):

                    if not autopilot and not virtual_obstacle:

                        player_lane = min(
                            2,
                            player_lane + 1
                        )

                        car_x = float(
                            LANE_CENTERS[player_lane]
                        )

        # ---------------------------------------------
        # READ VIDEO
        # ---------------------------------------------

        success, frame = camera.read()

        if not success:

            camera.set(
                cv2.CAP_PROP_POS_FRAMES,
                0
            )

            continue

        current_frame = frame

        # ---------------------------------------------
        # COMPUTER VISION PIPELINE
        # ---------------------------------------------

        (
            processed,
            stages,
            lane_status,
            current_offset
        ) = process_frame(frame)

        # ---------------------------------------------
        # YOLO DETECTION
        # ---------------------------------------------

        frame_count += 1

        if frame_count % 2 == 0 or not cached_detections:

            yolo_annotated, cached_detections = (
                detector.detect(frame)
            )

        else:

            yolo_annotated = frame.copy()

        # Draw YOLO detections
        yolo_annotated = frame.copy()

        for obj in cached_detections:

            x1, y1, x2, y2 = obj["box"]

            name = obj["name"]
            confidence = obj["confidence"]

            cv2.rectangle(
                yolo_annotated,
                (x1, y1),
                (x2, y2),
                (0, 165, 255),
                2
            )

            cv2.putText(
                yolo_annotated,
                f"{name}: {confidence:.2f}",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 165, 255),
                2
            )

        stages["yolo"] = yolo_annotated
        stages["final"] = processed

        # ---------------------------------------------
        # NAVIGATION
        # ---------------------------------------------

        update_navigation()

        # ---------------------------------------------
        # DRAW UI
        # ---------------------------------------------

        screen.fill(
            (15, 18, 28)
        )

        draw_cv_pipeline(
            frame,
            stages
        )

        draw_dashboard()

        draw_game()

        pygame.display.flip()

finally:

    camera.release()

    pygame.quit()

    cv2.destroyAllWindows()

    print("Application closed.")
