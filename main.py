"""
Entry point: launches the settings window, then the simulation.
Pressing S in the simulation returns to the settings window.
"""
import sys

from settings_window import SettingsWindow
from simulation import Simulation


def main():
    last_parking, last_car = None, None

    while True:
        settings = SettingsWindow(last_parking, last_car)
        result = settings.run()

        if result is None:
            # User closed settings without confirming
            sys.exit(0)

        parking_config, car_config = result
        last_parking, last_car = parking_config, car_config

        sim = Simulation(parking_config, car_config)
        go_back = sim.run()

        if not go_back:
            # User pressed ESC/Q or closed the window — exit
            sys.exit(0)
        # go_back == True → loop back to settings with last values


if __name__ == "__main__":
    main()
