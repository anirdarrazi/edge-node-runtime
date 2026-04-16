from node_agent.config import NodeAgentSettings


def test_blank_optional_environment_values_are_treated_as_unset(monkeypatch) -> None:
    for name in (
        "NODE_ID",
        "NODE_KEY",
        "BURST_PROVIDER",
        "BURST_LEASE_PHASE",
        "BURST_COST_CEILING_USD",
        "ROOM_TEMP_C",
        "TARGET_TEMP_C",
        "GPU_TEMP_C",
        "POWER_WATTS",
        "ESTIMATED_HEAT_OUTPUT_WATTS",
        "ENERGY_PRICE_KWH",
    ):
        monkeypatch.setenv(name, "")

    settings = NodeAgentSettings()

    assert settings.node_id is None
    assert settings.node_key is None
    assert settings.burst_provider is None
    assert settings.burst_lease_phase is None
    assert settings.burst_cost_ceiling_usd is None
    assert settings.room_temp_c is None
    assert settings.target_temp_c is None
    assert settings.gpu_temp_c is None
    assert settings.power_watts is None
    assert settings.estimated_heat_output_watts is None
    assert settings.energy_price_kwh is None
