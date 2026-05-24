"""
Entry point: launches the settings window, then the simulation.
"""
import sys

from settings_window import SettingsWindow
from simulation import Simulation


def main():
    settings = SettingsWindow()
    result = settings.run()

    if result is None:
        # User closed the window without confirming
        sys.exit(0)

    parking_config, car_config = result
    sim = Simulation(parking_config, car_config)
    sim.run()


if __name__ == "__main__":
    main()
