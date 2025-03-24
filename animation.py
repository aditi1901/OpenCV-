import cv2
import numpy as np
import random
import time
import mediapipe as mp

# Load images for the game
cat_img1 = cv2.imread(r"C:\Users\aditi\OneDrive\Desktop\project-hopefully\cat.jpeg", cv2.IMREAD_UNCHANGED)
cat_img2 = cv2.imread(r"C:\Users\aditi\OneDrive\Desktop\project-hopefully\cat.jpeg", cv2.IMREAD_UNCHANGED)
rat_img = cv2.imread(r"C:\Users\aditi\OneDrive\Desktop\project-hopefully\rat.webp", cv2.IMREAD_UNCHANGED)

powerup_rat_img = cv2.imread(r"C:\Users\aditi\OneDrive\Desktop\project-hopefully\apple.png", cv2.IMREAD_UNCHANGED)
powerup_rat_img = cv2.resize(powerup_rat_img, (40, 40))  # Slightly larger for visibility

# Variables for the power-up rat
powerup_rat_position = None  # Position for the power-up rat
powerup_rat_active = False   # Whether the power-up rat is active
powerup_collected = False    # Whether the power-up rat has been collected

# Function to apply a glow effect to the power-up rat
def apply_glow_effect(image):
    # Create a larger blurred version of the rat image for the glow
    glow = cv2.GaussianBlur(image, (21, 21), 10)
    # Overlay the glow on the main image
    overlay = cv2.addWeighted(glow, 0.8, image, 0.5, 0)
    return overlay

# Apply glow effect to power-up rat
powerup_rat_img_glow = apply_glow_effect(powerup_rat_img)


# Resize images for better gameplay
cat_img1 = cv2.resize(cat_img1, (100, 100))
cat_img2 = cv2.resize(cat_img2, (100, 100))
rat_img = cv2.resize(rat_img, (30, 30))

# Initialize video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

# Initialize MediaPipe hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)

# Initialize variables
cat_positions = [[160, 240], [480, 240]]  # Initial positions for both cats
rat_positions = [[random.randint(0, 640), random.randint(50, 400)] for _ in range(10)]
rat_directions = [np.random.randint(-2, 3, size=2) for _ in range(10)]
scores = [0, 0]  # Player 1 and Player 2 scores
start_time = time.time()

# Game settings
TOP_MARGIN = 50
RIGHT_MARGIN = 50
FRAME_WIDTH, FRAME_HEIGHT = 640, 480

# Full-screen setup
cv2.namedWindow("Two-Player Game", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Two-Player Game", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Variables for Optical Flow
prev_gray = None
optical_flow_enabled = True  # Flag to toggle optical flow for testing
flow_smoothing_factor = 0.1  # Controls how much smoothing is applied

# Constants for text formatting
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.8
FONT_THICKNESS = 2
COLOR_PLAYER1 = (0, 255, 0)  # Green
COLOR_PLAYER2 = (0, 255, 0)  # Green
COLOR_TIMER = (255, 255, 0)  # Yellow
COLOR_GAME_OVER = (0, 0, 255)  # Red
TEXT_MARGIN = 20

# Function to overlay image
# def overlay_image(background, overlay, position):
#     x, y = position
#     h, w, _ = overlay.shape
#     if y + h > background.shape[0]:
#         h = background.shape[0] - y
#     if x + w > background.shape[1]:
#         w = background.shape[1] - x
#     overlay_resized = cv2.resize(overlay, (w, h))
#     for c in range(3):
#         background[y:y+h, x:x+w, c] = overlay_resized[:h, :w, c]
#     return background
def overlay_image(background, overlay, position):
    x, y = position
    h, w = overlay.shape[:2]

    # Ensure the overlay fits within the background dimensions
    if y + h > background.shape[0]:
        h = background.shape[0] - y
    if x + w > background.shape[1]:
        w = background.shape[1] - x
    overlay = overlay[:h, :w]

    # Split overlay into BGR and alpha channels
    if overlay.shape[2] == 4:  # Check if overlay has an alpha channel
        b, g, r, alpha = cv2.split(overlay)
        alpha = alpha / 255.0  # Normalize alpha to range 0-1

        # Extract the region of interest from the background
        bg_region = background[y:y+h, x:x+w]

        # Blend overlay with background using alpha channel
        for c in range(3):  # Loop over B, G, R channels
            bg_region[:, :, c] = bg_region[:, :, c] * (1 - alpha) + overlay[:, :, c] * alpha

        # Replace the region in the background with the blended result
        background[y:y+h, x:x+w] = bg_region
    else:
        # If no alpha channel, just overlay as is
        for c in range(3):
            background[y:y+h, x:x+w, c] = overlay[:, :, c]

    return background


# Game state variable
game_over = False

# Function to reset the game
def reset_game():
    global cat_positions, rat_positions, rat_directions, scores, start_time, game_over
    cat_positions = [[160, 240], [480, 240]]  # Reset positions for both cats
    rat_positions = [[random.randint(0, 640), random.randint(50, 400)] for _ in range(10)]
    rat_directions = [np.random.randint(-2, 3, size=2) for _ in range(10)]
    scores = [0, 0]  # Reset scores
    start_time = time.time()  # Reset timer
    game_over = False  # Reset game state



# Function for fade-out effect during game over
def fade_out(frame, fade_duration=3):
    for i in range(0, 255, int(255 / (fade_duration * 30))):
        faded_frame = cv2.addWeighted(frame, 1, np.zeros_like(frame), 0, i)
        cv2.imshow("Two-Player Game", faded_frame)
        cv2.waitKey(1)
    return np.zeros_like(frame)  # Return a black screen after fade-out

# Function for animated text
def animate_text(frame, text, position, color, font, font_scale, font_thickness, speed=50):
    for i in range(1, len(text) + 1):
        frame_copy = frame.copy()
        cv2.putText(frame_copy, text[:i], position, font, font_scale, color, font_thickness)
        cv2.imshow("Two-Player Game", frame_copy)
        cv2.waitKey(speed)
# Game loop
game_over = False
fade_out_done = False
while True:
    if game_over and not fade_out_done:
        # Display Game Over message and prompt to restart
        frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

        # Perform fade-out
        frame = fade_out(frame)
       
        # Determine the winner
        winner = "Player 1" if scores[0] > scores[1] else ("Player 2" if scores[1] > scores[0] else "Tied")
        game_over_text = f"Game Over! Winner: {winner}"

        # Display animated game-over message
        animate_text(frame, game_over_text, (FRAME_WIDTH // 2 - 200, FRAME_HEIGHT // 2 - 20), COLOR_GAME_OVER, FONT, 1, FONT_THICKNESS)
        restart_text = "Press 'R' to Restart or 'Q' to Quit"
        animate_text(frame, restart_text, (FRAME_WIDTH // 2 - 250, FRAME_HEIGHT // 2 + 40), COLOR_TIMER, FONT, FONT_SCALE, FONT_THICKNESS)
       
        # Mark fade-out as complete
        fade_out_done = True

    if game_over:
        # Display static "Game Over" frame after animations
        cv2.imshow("Two-Player Game", frame)

        # Wait for user input to restart or quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):  # Restart game
            reset_game()
            fade_out_done = False  # Reset animation state for next game
        elif key == ord('q'):  # Quit game
            break
        continue

    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Flip the frame for a mirror-like experience
    frame = cv2.flip(frame, 1)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Optical flow computation with smoothing
    if prev_gray is not None and optical_flow_enabled:
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow_magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)

        # Smooth the flow to avoid jerky movement
        flow_smoothed = flow * (1 - flow_smoothing_factor) + np.array(flow) * flow_smoothing_factor

        # Calculate the flow magnitude and direction, and use that for moving the cats
        for i, cat_pos in enumerate(cat_positions):
            y, x = int(cat_pos[1]), int(cat_pos[0])
            if 0 <= x < FRAME_WIDTH and 0 <= y < FRAME_HEIGHT:
                dx = int(flow_smoothed[y, x, 0])
                dy = int(flow_smoothed[y, x, 1])
                if abs(dx) > 2 or abs(dy) > 2:
                    cat_positions[i][0] = np.clip(cat_pos[0] + dx, 0, FRAME_WIDTH - 100)
                    cat_positions[i][1] = np.clip(cat_pos[1] + dy, 0, FRAME_HEIGHT - 100)

    prev_gray = gray_frame

    # MediaPipe hand tracking for fallback
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    hand_positions = [None, None]

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            hand_label = results.multi_handedness[idx].classification[0].label
            x = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * FRAME_WIDTH)
            y = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * FRAME_HEIGHT)

            if hand_label == "Left":
                hand_positions[0] = [x, y]
            elif hand_label == "Right":
                hand_positions[1] = [x, y]

    # Update player (cat) positions based on detected hand positions
    for i, hand_pos in enumerate(hand_positions):
        if hand_pos:
            cat_positions[i][0] = np.clip(hand_pos[0] - 50, 0, FRAME_WIDTH - 100)
            cat_positions[i][1] = np.clip(hand_pos[1] - 50, 0, FRAME_HEIGHT - 100)

    # Update rat positions
    for i, rat in enumerate(rat_positions):
        rat[0] += rat_directions[i][0]
        rat[1] += rat_directions[i][1]
        if rat[1] <= TOP_MARGIN or rat[1] >= FRAME_HEIGHT - 30:
            rat_directions[i][1] *= -1
        rat[0] = np.clip(rat[0], 0, FRAME_WIDTH - 30 - RIGHT_MARGIN)

        # Check if a cat eats a rat
        for j, cat in enumerate(cat_positions):
            if abs(cat[0] - rat[0]) < 50 and abs(cat[1] - rat[1]) < 50:
                rat_positions.pop(i)
                rat_directions.pop(i)
                scores[j] += 1
                break

    # Add new rats if necessary
    while len(rat_positions) < 10:
        rat_positions.append([random.randint(0, FRAME_WIDTH), random.randint(50, FRAME_HEIGHT)])
        rat_directions.append(np.random.randint(-2, 3, size=2))

    # Overlay cats and rats
    frame = overlay_image(frame, cat_img1, cat_positions[0])
    frame = overlay_image(frame, cat_img2, cat_positions[1])
    for rat in rat_positions:
        frame = overlay_image(frame, rat_img, rat)

    # Display scores and timer
    elapsed_time = time.time() - start_time
    remaining_time = max(0, 30 - int(elapsed_time))


    elapsed_time = time.time() - start_time
    remaining_time = max(0, 30 - int(elapsed_time))

    if remaining_time <= 10 and not powerup_rat_active and not powerup_collected:
        # Activate power-up rat in the last 10 seconds if it hasn't been collected
        powerup_rat_position = [random.randint(0, FRAME_WIDTH - 40), random.randint(TOP_MARGIN, FRAME_HEIGHT - 40)]
        powerup_rat_active = True

    # Check if a cat collects the power-up rat
    if powerup_rat_active and powerup_rat_position:
        for j, cat in enumerate(cat_positions):
            if abs(cat[0] - powerup_rat_position[0]) < 50 and abs(cat[1] - powerup_rat_position[1]) < 50:
                # Double the score for the player who collected the power-up rat
                scores[j] *= 2
                powerup_rat_active = False
                powerup_collected = True  # Mark power-up as collected
                break

    # Overlay the power-up rat if active
    if powerup_rat_active and powerup_rat_position:
        frame = overlay_image(frame, powerup_rat_img_glow, powerup_rat_position)


    # Show the timer and scores
    cv2.putText(frame, f"Player 1 Score: {scores[0]}", (20, 40), FONT, FONT_SCALE, COLOR_PLAYER1, FONT_THICKNESS)
    cv2.putText(frame, f"Player 2 Score: {scores[1]}", (20, 80), FONT, FONT_SCALE, COLOR_PLAYER2, FONT_THICKNESS)
    cv2.putText(frame, f"Time Left: {remaining_time}", (FRAME_WIDTH - 180, 40), FONT, FONT_SCALE, COLOR_TIMER, FONT_THICKNESS)

    if remaining_time == 0:
        game_over = True
        continue

    cv2.imshow("Two-Player Game", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

