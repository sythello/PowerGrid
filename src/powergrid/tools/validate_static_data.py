from __future__ import annotations

from powergrid.rules_data import load_map, load_power_plants, load_rule_tables, validate_static_data


def main() -> None:
    report = validate_static_data()
    rules = load_rule_tables()
    germany = load_map("germany")
    usa = load_map("usa")
    plants = load_power_plants()

    print("Static data validation: PASS")
    print(f"Maps loaded: {', '.join(report.maps_loaded)}")
    print(f"Power plants loaded: {report.power_plant_count}")
    print(f"Player counts loaded: {', '.join(str(count) for count in report.player_counts)}")
    print(
        "Germany sample city: "
        f"{germany.cities[0].name} in region {germany.cities[0].region}"
    )
    print(
        "USA sample connection: "
        f"{usa.connections[0].city_1} -> {usa.connections[0].city_2} "
        f"(cost {usa.connections[0].cost})"
    )
    print(
        "Sample power plants: "
        f"{plants[0].price}, {plants[1].price}, {plants[2].price}, "
        f"{plants[-1].price}"
    )
    print(
        "Starting market counts: "
        f"coal={len(rules.resource_market_tracks['coal']['starting_prices']) * 3}, "
        f"oil={len(rules.resource_market_tracks['oil']['starting_prices']) * 3}, "
        f"garbage={len(rules.resource_market_tracks['garbage']['starting_prices']) * 3}, "
        f"uranium={len(rules.resource_market_tracks['uranium']['starting_prices'])}"
    )


if __name__ == "__main__":
    main()
