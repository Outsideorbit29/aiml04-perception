"""Night / rain: the stopped-car layout under a ``carla.WeatherParameters`` preset (default
HardRainNight), with the ego and obstacle low beams on. The original weather is restored."""

from __future__ import annotations

from sim.carla_utils import weather_preset
from sim.scenarios.stopped_car import StoppedCar


class NightRain(StoppedCar):
    name = "night_rain"

    def setup(self) -> None:
        import carla

        self._weather = self.world.get_weather()
        self.world.set_weather(weather_preset(str(self.cfg["weather"])))
        super().setup()
        if self.cfg.get("lights"):
            beams = carla.VehicleLightState(
                carla.VehicleLightState.LowBeam | carla.VehicleLightState.Position
            )
            for v in (self.ego, self.hazard):
                v.set_light_state(beams)

    def cleanup(self) -> None:
        if getattr(self, "_weather", None) is not None:
            self.world.set_weather(self._weather)
