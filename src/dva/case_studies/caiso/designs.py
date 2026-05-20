from __future__ import annotations

from dataclasses import dataclass

from dva.analysis.caiso_shap import ParameterPlayerSpec
from dva.model.storage_dispatch import StorageDispatchParameters


CAISO_INFO_PLAYERS = (
    "min_temp_c",
    "max_temp_c",
    "mean_temp_c",
    "mean_humidity",
    "mean_wind_speed",
    "mean_solar_irradiance",
    "max_solar_irradiance",
    "day_of_week",
)


@dataclass(frozen=True, slots=True)
class CaisoStorageDesign:
    name: str
    throughput_penalty: float
    energy_capacity: float
    efficiency: float
    power_limit: float = 1.0
    initial_state_of_charge: float | None = None
    terminal_state_of_charge: float | None = None

    def parameters(self) -> StorageDispatchParameters:
        initial_soc = (
            self.energy_capacity / 2.0
            if self.initial_state_of_charge is None
            else self.initial_state_of_charge
        )
        terminal_soc = (
            initial_soc
            if self.terminal_state_of_charge is None
            else self.terminal_state_of_charge
        )
        return StorageDispatchParameters(
            energy_capacity=self.energy_capacity,
            power_limit=self.power_limit,
            charge_efficiency=self.efficiency,
            discharge_efficiency=self.efficiency,
            throughput_penalty=self.throughput_penalty,
            initial_state_of_charge=initial_soc,
            terminal_state_of_charge=terminal_soc,
        )


CAISO_ACTUAL_DESIGN = CaisoStorageDesign(
    name="default",
    throughput_penalty=5.0,
    energy_capacity=4.0,
    efficiency=0.95,
    initial_state_of_charge=2.0,
    terminal_state_of_charge=2.0,
)
CAISO_BASELINE_DESIGNS = {
    "conservative": CaisoStorageDesign(
        name="conservative",
        throughput_penalty=10.0,
        energy_capacity=2.0,
        efficiency=0.8,
        initial_state_of_charge=1.0,
        terminal_state_of_charge=1.0,
    ),
    "optimistic": CaisoStorageDesign(
        name="optimistic",
        throughput_penalty=0.0,
        energy_capacity=24.0,
        efficiency=1.0,
        initial_state_of_charge=12.0,
        terminal_state_of_charge=12.0,
    ),
}


def parameter_player_spec_for_baseline(
    baseline: CaisoStorageDesign,
    *,
    include_state_of_charge: bool = False,
) -> ParameterPlayerSpec:
    baseline_parameters = baseline.parameters()
    return ParameterPlayerSpec(
        throughput_penalty_baseline=baseline.throughput_penalty,
        efficiency_is_player=True,
        charge_efficiency_baseline=baseline.efficiency,
        discharge_efficiency_baseline=baseline.efficiency,
        energy_capacity_is_player=True,
        energy_capacity_baseline=baseline.energy_capacity,
        initial_state_of_charge_baseline=(
            baseline_parameters.initial_state_of_charge
            if include_state_of_charge
            else None
        ),
        terminal_state_of_charge_baseline=(
            baseline_parameters.terminal_state_of_charge
            if include_state_of_charge
            else None
        ),
    )


__all__ = [
    "CAISO_ACTUAL_DESIGN",
    "CAISO_BASELINE_DESIGNS",
    "CAISO_INFO_PLAYERS",
    "CaisoStorageDesign",
    "parameter_player_spec_for_baseline",
]
