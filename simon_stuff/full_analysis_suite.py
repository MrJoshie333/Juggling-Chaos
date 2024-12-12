import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

# ===== INPUT VARIABLES (CAN BE CHANGED) =====

AUTO_FIND_VIDEO_PATHS: bool = False  # True to automatically process videos in the "videos" directory
VIDEO_EXTENSIONS: list[str] = [".mp4", ".mov"]  # Video extensions to search for

# List of video paths (if AUTO_FIND_VIDEO_PATHS is False)
VIDEO_PATHS: list[str] = ["videos/simon_normal.mp4", "videos/simon_fast.mp4", "videos/josh_normal.mp4",
                          "videos/josh_fast.mp4", "videos/josh_slow.mp4"]

OUTPUT_DIR: str = "output"  # Main output directory

SCALE_FACTOR: int = 2866  # pixels per meter
WHITE_RANGE: tuple = ((0, 0, 200), (180, 50, 255))  # Color range for detecting white balls

MIN_VALLEY_DISTANCE = 0.6  # Minimum delay between valleys/catches (in seconds)

PROCESS_VIDEOS = True  # True to process videos, False to load data from text files
OVERWRITE_GRAPHS = False  # True to overwrite existing final graph


# ===========================


def main():
    # Create output directories (if they don't exist)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR + "/text_data", exist_ok=True)
    os.makedirs(OUTPUT_DIR + "/peak_plots", exist_ok=True)
    os.makedirs(OUTPUT_DIR + "/y_plots", exist_ok=True)
    os.makedirs(OUTPUT_DIR + "/fourier_plots", exist_ok=True)
    os.makedirs(OUTPUT_DIR + "/ratio_plots", exist_ok=True)
    os.makedirs(OUTPUT_DIR + "/phase_space_plots", exist_ok=True)

    if len(VIDEO_PATHS) == 0 and not AUTO_FIND_VIDEO_PATHS:
        print("No video paths provided.")
        return

    if AUTO_FIND_VIDEO_PATHS:
        paths_to_process = find_video_paths(os.path.join(os.getcwd(), "videos"))
    else:
        paths_to_process = VIDEO_PATHS

    # Initialize lists to store results
    all_l_f = []
    all_l_u = []
    point_appearance = []

    print("============================================")
    for i, path in enumerate(paths_to_process):

        if PROCESS_VIDEOS:
            # Process video
            path = path.replace("\\", "/")
            print(f"Processing video {i + 1}/{len(paths_to_process)}: {path.split('/')[-1]}")
            time_steps, y_positions, velocity_data, fps = process_video(path)

            # Detect useful peaks and valleys in the velocity data
            good_peaks, valleys = find_velocity_extrema(velocity_data, time_steps, fps)
            mean_data = analyze_peaks_and_valleys(good_peaks, valleys)
        else:
            # Load data from text files
            print(f"Loading data from text files for video {i + 1}/{len(paths_to_process)}: {path.split('/')[-1]}")
            base_name = path.split("/")[-1].split(".")[0]
            text_path = OUTPUT_DIR + "/text_data/" + base_name + "_peaks_valleys.txt"
            good_peaks, valleys, mean_data = parse_text_file(text_path)

        # Update ratios
        l = mean_data[1][1]
        f = mean_data[0][1]
        u = (mean_data[2][1] + mean_data[3][1]) / 2 / 10
        try:
            all_l_f.append(l / f)
            all_l_u.append(l / u)
        except ZeroDivisionError:
            print(f"Invalid Data: Skipping video {i + 1}/{len(paths_to_process)}: {path.split('/')[-1]}")
            print("============================================")
            continue
        point_appearance.append(("red" if "josh" in path else "blue", "J" if "josh" in path else "S"))

        # Display results
        print("\nPeaks (Throwing Events):")
        for t, v in good_peaks:
            print(f"Time: {t:.2f}s, Velocity: {v:.2f} m/s")
        print("\nValleys (Catching Events):")
        for t, v in valleys:
            print(f"Time: {t:.2f}s, Velocity: {v:.2f} m/s")

        # Display calculated mean data
        print()
        for key, value in mean_data:
            print(f"{key} {value:.2f}")

        # Get clean strings for plot title
        output_name = path.split('/')[-1].split('.')[0]

        if PROCESS_VIDEOS:
            # Save peaks and valleys to a text file
            output_file = os.path.join(OUTPUT_DIR, "text_data", f"{output_name}_peaks_valleys.txt")
            with open(output_file, "w") as f:
                f.write("Peaks (Throwing Events):\n")
                for t, v in good_peaks:
                    f.write(f"Time: {t:.2f}s, Velocity: {v:.2f} m/s\n")
                f.write("\nValleys (Catching Events):\n")
                for t, v in valleys:
                    f.write(f"Time: {t:.2f}s, Velocity: {v:.2f} m/s\n")
                f.write("\nMean Data:\n")
                for key, value in mean_data:
                    f.write(f"{key}: {value:.2f}\n")

            # ========= PLOTTING SECTION =========
            print("\nPlotting results...\n")

            # Save plot of y-position
            y_pos_path = os.path.join(OUTPUT_DIR, "y_plots",
                                      f"{path.split('/')[-1].split('.')[0]}_video_y_positions.png")
            save_y_position_plot(time_steps, y_positions, y_pos_path)
            print(f"y-position plot saved as '{y_pos_path}'.")

            # Save plot of velocity with marked peaks and valleys
            peak_path = os.path.join(OUTPUT_DIR, "peak_plots", f"{output_name}_velocity_peaks_valleys.png")
            save_peak_plot(time_steps, velocity_data, good_peaks, valleys, peak_path)
            print(f"Peak plot saved as '{peak_path}'.")

            # Save plot of fourier transform
            fourier_path = os.path.join(OUTPUT_DIR, "fourier_plots",
                                        f"{output_name}_fourier_transform.png")
            save_fourier_plot(y_positions, fps, fourier_path)
            print(f"Fourier Transform plot saved as '{fourier_path}'.")

            # Save plot of phase space
            phase_space_path = os.path.join(OUTPUT_DIR, "phase_space_plots",
                                            f"{path.split('/')[-1].split('.')[0]}_phase_space_plot.png")
            save_phase_space_plot(y_positions, velocity_data, phase_space_path)
            print(f"Phase-Space plot saved as '{phase_space_path}'.")

        print("============================================")

    # Save ratio plots
    save_ratio_plots(all_l_f, all_l_u, point_appearance, OVERWRITE_GRAPHS)


def find_video_paths(search_directory):
    video_paths = []

    for root, dirs, files in os.walk(search_directory):
        for file in files:
            for ext in VIDEO_EXTENSIONS:
                if file.endswith(ext):
                    video_paths.append(os.path.join(root, file))

    print(f"Found {len(video_paths)} videos in {search_directory}")

    return video_paths


def parse_text_file(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

    peaks = []
    valleys = []
    mean_data = []
    data_type = None
    for i, line in enumerate(lines):
        if line.strip() == "":
            continue
        elif "Peak" in line.split()[0]:
            data_type = "peak"
            continue
        elif "Valley" in line.split()[0]:
            data_type = "valley"
            continue
        elif "Mean Data" in line:
            data_type = "mean"
            continue

        if data_type == "mean":
            raw = line.strip().split(": ")
            mean_data.append((raw[0], float(raw[1])))
        else:
            raw = line.strip().split(", ")
            time = raw[0].split(": ")[1][:-1]
            velocity = raw[1].split(": ")[1].split(" ")[0]

            if data_type == "peak":
                peaks.append((float(time), float(velocity)))
            elif data_type == "valley":
                valleys.append((float(time), float(velocity)))

    return peaks, valleys, mean_data


def process_video(video_path: str):
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video loaded: {video_path}")
    print(f"Resolution: {frame_width}x{frame_height}, FPS: {fps}, Total Frames: {total_frames}")

    # Variables for tracking/calculating velocity
    y_positions = []  # Store y-values of the ball's position
    time_steps = []
    velocity_data = []  # Store velocity data
    prev_y = None

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        current_time = frame_idx / fps

        # Convert the frame to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create a mask for white objects
        mask = cv2.inRange(hsv, WHITE_RANGE[0], WHITE_RANGE[1])
        mask = cv2.GaussianBlur(mask, (15, 15), 0)

        # Find contours (balls)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]

        # Get pixel position of detected ball
        current_position = None
        for contour in contours:
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                current_position = (cx, cy)

        if current_position:
            cx, cy = current_position
            y_positions.append((frame_height - cy) / SCALE_FACTOR)
            time_steps.append(current_time)

            if prev_y is not None:
                vel = ((cy - prev_y) * fps) / SCALE_FACTOR  # Compute velocity in meters per second
                velocity_data.append(vel)

            prev_y = cy

        frame_idx += 1

    # Close video capture
    cap.release()

    return time_steps, y_positions, velocity_data, fps


# Peak finding for velocity extrema
def find_velocity_extrema(velocity_data: list[float], time_data: list[float], fps: float) -> tuple[
    list[tuple[float, float]], list[tuple[float, float]]]:
    if len(velocity_data) < 2:
        print("Not enough data for peak detection.")
        return

    # Find peaks and valleys
    peak_indices, _ = find_peaks(velocity_data, prominence=0.05)
    valley_indices, _ = find_peaks(-np.array(velocity_data), prominence=0.5, distance=MIN_VALLEY_DISTANCE * fps)

    # Gather data for peaks and valleys
    peaks: list[tuple[float, float]] = [(time_data[peak], velocity_data[peak]) for peak in peak_indices]
    valleys: list[tuple[float, float]] = [(time_data[valley], velocity_data[valley]) for valley in valley_indices if
                                          velocity_data[valley] < -1]

    # Extract only the peaks that immediately precede a valley
    good_peaks = []
    for i, valley in enumerate(valleys):
        peaks_to_check = [peak for peak in peaks if peak[0] < valley[0]]
        if not peaks_to_check:
            # Ignore valleys with no preceding peak
            valleys[i] = None
            continue
        closest_peak = min(peaks_to_check, key=lambda peak: abs(peak[0] - valley[0]))
        good_peaks.append(closest_peak)

    # Remove valleys with no preceding peak
    good_valleys = [valley for valley in valleys if valley is not None]

    return good_peaks, good_valleys


def analyze_peaks_and_valleys(peaks, valleys) -> list[tuple[str, float]]:
    """
    Analyzes the differences between peaks and valleys in various scenarios:
    1. Differences between nth peak and (n+1)th valley.
    2. Differences between nth peak and nth valley.
    3. Differences between first peak, skipping every other peak, and (n+1)th valley.
    4. Differences between skipped peaks and their corresponding (n+1)th valleys.
    """
    # Scenario 1: nth peak - nth valley
    scenario_1_differences = []
    for i in range(len(peaks) - 1):  # Stop at len(peaks) - 1 to avoid out-of-range
        if i + 1 < len(valleys):
            scenario_1_differences.append(abs(peaks[i][0] - valleys[i][0]))

    # Scenario 2: (n+1)th peak - nth valley
    scenario_2_differences = []
    for i in range(min(len(peaks), len(valleys))):  # Pair only valid peaks and valleys
        if i + 1 < len(valleys):
            scenario_2_differences.append(abs(peaks[i + 1][0] - valleys[i][0]))

    # Scenario 3: First peak, skipping every other peak, and (n+1)th valley
    scenario_3_differences = []
    for i in range(0, len(peaks), 2):  # Skip every other peak
        if i + 1 < len(valleys):
            scenario_3_differences.append(abs(peaks[i][0] - valleys[i + 1][0]))

    # Scenario 4: Skipped peaks and their corresponding (n+1)th valleys
    scenario_4_differences = []
    for i in range(1, len(peaks), 2):  # Use skipped peaks
        if i + 1 < len(valleys):
            scenario_4_differences.append(abs(peaks[i][0] - valleys[i + 1][0]))

    # Calculate means
    mean_scenario_1 = sum(scenario_1_differences) / len(scenario_1_differences) if scenario_1_differences else 0
    mean_scenario_2 = sum(scenario_2_differences) / len(scenario_2_differences) if scenario_2_differences else 0
    mean_scenario_3 = sum(scenario_3_differences) / len(scenario_3_differences) if scenario_3_differences else 0
    mean_scenario_4 = sum(scenario_4_differences) / len(scenario_4_differences) if scenario_4_differences else 0

    return [
        ("Mean of differences for nth peak and (n+1)th valley", mean_scenario_1),
        ("Mean of differences for nth peak and nth valley", mean_scenario_2),
        ("Mean of differences for first, skipping peaks, and (n+1)th valley", mean_scenario_3),
        ("Mean of differences for skipped peaks and their corresponding (n+1)th valley", mean_scenario_4),
    ]


def save_peak_plot(time_steps, velocity_data, good_peaks, valleys, output_path):
    # Save plot of velocity with marked peaks and valleys
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_steps[:len(velocity_data)], velocity_data, label="Velocity", color='blue')
    ax.scatter(*zip(*good_peaks), color='red', label="Good Peaks (Throwing)", zorder=5)
    ax.scatter(*zip(*valleys), color='green', label="Valleys (Catching)", zorder=5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (m/s)')
    juggle_speed = output_path.split('/')[-1].split('.')[0].split('_')[-1].capitalize()
    ax.set_title(f"{juggle_speed} Juggling - Velocity Over Time with Peaks and Valleys")
    ax.grid(True)
    ax.legend()
    fig.savefig(output_path)
    plt.close(fig)


def save_phase_space_plot(y_positions, velocity, output_path):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.plot(y_positions[:len(velocity)], velocity, color='blue', markersize=5, label="Phase Trajectory")
    ax.set_xlabel('y-Displacement (m)')
    ax.set_ylabel('y-Velocity (m/s)')
    ax.set_title(
        f"{output_path.split('/')[-1].split('.')[0].split('_')[-1].capitalize()}" + ' Phase-Space Plot: y-Displacement vs. y-Velocity')
    ax.grid(True)
    ax.legend()
    fig.savefig(output_path)
    plt.close(fig)


def save_y_position_plot(time_steps, y_positions, output_path):
    # Save y-position plot
    if len(y_positions) > 0:
        fig1 = plt.figure(figsize=(10, 6))
        ax1 = fig1.add_subplot(111)
        ax1.plot(time_steps, y_positions, label="y-position of Ball", color='green')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('y-Position (m)')
        ax1.set_title(
            f"{output_path.split('/')[-1].split('.')[0].split('_')[-1].capitalize()}" + ' y-Position of Ball Over Time')
        ax1.grid(True)
        ax1.legend()
        fig1.savefig(output_path)
        plt.close(fig1)


def save_fourier_plot(y_positions, fps, output_path):
    # Perform Fourier transform
    if len(y_positions) > 1:
        y_data = np.array(y_positions)
        fft_result = np.fft.fft(y_data)
        freqs = np.fft.fftfreq(len(y_data), 1 / fps)
        fft_amplitude = np.abs(fft_result)

        fig2 = plt.figure(figsize=(10, 6))
        ax2 = fig2.add_subplot(111)
        ax2.plot(np.fft.fftshift(freqs), np.fft.fftshift(fft_amplitude), label="Fourier Transform (Amplitude)",
                 color='blue')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Amplitude')
        ax2.set_title(
            f"{output_path.split('/')[-1].split('.')[0].split('_')[-1].capitalize()}" + ' Fourier Transform of y-Position Over Time')
        ax2.grid(True)
        ax2.set_xlim(-5, 5)
        ax2.legend()
        fig2.savefig(output_path)
        plt.close(fig2)


def save_ratio_plots(all_l_f, all_l_u, point_appearance, overwrite_path=False):
    # Create a figure with two subplots
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Scatter plot for ratio points on the first subplot
    prev_name = ""
    for l_f, l_u, info in zip(all_l_f, all_l_u, point_appearance):
        if info[1] != prev_name:
            ax[0].scatter(l_f, l_u, color=info[0], label=info[1])
            prev_name = info[1]
        else:
            ax[0].scatter(l_f, l_u, color=info[0])

    ax[0].set_xlabel("l/f")
    ax[0].set_ylabel("l/u")
    ax[0].legend()
    ax[0].set_title("Scatter Plot")

    # Quadratic trendline for combined data (quadratic regression)
    coefficients_all = np.polyfit(all_l_f, all_l_u, 2)  # degree 2 for quadratic fit
    trendline_all = np.poly1d(coefficients_all)

    # Second subplot with both scatter points and quadratic trendline
    x_vals_all = np.linspace(min(all_l_f), max(all_l_f), 100)  # X values for the combined trendline

    prev_name = ""
    for l_f, l_u, info in zip(all_l_f, all_l_u, point_appearance):
        if info[1] != prev_name:
            ax[1].scatter(l_f, l_u, color=info[0], label=info[1])
            prev_name = info[1]
        else:
            ax[1].scatter(l_f, l_u, color=info[0])
    ax[1].plot(x_vals_all, trendline_all(x_vals_all), color="green", label="Combined Quadratic Trendline")

    ax[1].set_xlabel("l/f")
    ax[1].set_ylabel("l/u")
    ax[1].legend()
    ax[1].set_title("Combined Quadratic Trendline")

    # Display the plots
    plt.tight_layout()
    plt.show()
    filename_count = 0
    save_path = os.path.join(OUTPUT_DIR, "ratio_plots", f"combined_ratio_plot.png")
    if not overwrite_path:
        while os.path.exists(save_path):
            filename_count += 1
            save_path = os.path.join(OUTPUT_DIR, f"combined_ratio_plot_{filename_count}.png")
    fig.savefig(save_path)

    # Output the quadratic trendline equation for the combined data
    print(
        f"Combined quadratic trendline equation: y = {coefficients_all[0]}x^2 + {coefficients_all[1]}x + {coefficients_all[2]}")


if __name__ == '__main__':
    main()
