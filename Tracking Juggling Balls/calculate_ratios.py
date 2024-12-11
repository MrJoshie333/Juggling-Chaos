def analyze_peaks_and_valleys(peaks, valleys):
    """
    Analyzes the differences between peaks and valleys in various scenarios:
    1. Differences between nth peak and (n+1)th valley.
    2. Differences between nth peak and nth valley.
    3. Differences between first peak, skipping every other peak, and (n+1)th valley.
    4. Differences between skipped peaks and their corresponding (n+1)th valleys.

    Parameters:
        peaks (list of float): List of peak values.
        valleys (list of float): List of valley values.

    Returns:
        dict: Means of differences for all scenarios.
    """
    # Scenario 1: nth peak - (n+1)th valley
    scenario_1_differences = []
    for i in range(len(peaks) - 1):  # Stop at len(peaks) - 1 to avoid out-of-range
        if i + 1 < len(valleys):
            scenario_1_differences.append(abs(peaks[i] - valleys[i + 1]))

    # Scenario 2: nth peak - nth valley
    scenario_2_differences = []
    for i in range(min(len(peaks), len(valleys))):  # Pair only valid peaks and valleys
        scenario_2_differences.append(abs(peaks[i] - valleys[i]))

    # Scenario 3: First peak, skipping every other peak, and (n+1)th valley
    scenario_3_differences = []
    for i in range(0, len(peaks), 2):  # Skip every other peak
        if i + 1 < len(valleys):
            scenario_3_differences.append(abs(peaks[i] - valleys[i + 1]))

    # Scenario 4: Skipped peaks and their corresponding (n+1)th valleys
    scenario_4_differences = []
    for i in range(1, len(peaks), 2):  # Use skipped peaks
        if i + 1 < len(valleys):
            scenario_4_differences.append(abs(peaks[i] - valleys[i + 1]))

    # Calculate means
    mean_scenario_1 = sum(scenario_1_differences) / len(scenario_1_differences) if scenario_1_differences else 0
    mean_scenario_2 = sum(scenario_2_differences) / len(scenario_2_differences) if scenario_2_differences else 0
    mean_scenario_3 = sum(scenario_3_differences) / len(scenario_3_differences) if scenario_3_differences else 0
    mean_scenario_4 = sum(scenario_4_differences) / len(scenario_4_differences) if scenario_4_differences else 0

    return {
        "mean_scenario_1": mean_scenario_1,
        "mean_scenario_2": mean_scenario_2,
        "mean_scenario_3": mean_scenario_3,
        "mean_scenario_4": mean_scenario_4,
    }

# Usage
peaks = [1.20, 2.63, 3.87]
valleys = [1.58, 2.98, 4.38]


 #if the first point is a valley, delete it ONLY for the last two mean differences


results = analyze_peaks_and_valleys(peaks, valleys)

print(f"Mean of differences for nth peak and (n+1)th valley: {results['mean_scenario_1']:.3f}")
print(f"Mean of differences for nth peak and nth valley: {results['mean_scenario_2']:.3f}")
print(f"Mean of differences for first, skipping peaks, and (n+1)th valley: {results['mean_scenario_3']:.3f}")
print(f"Mean of differences for skipped peaks and their corresponding (n+1)th valley: {results['mean_scenario_4']:.3f}")
