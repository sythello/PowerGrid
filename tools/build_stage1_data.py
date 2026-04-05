from __future__ import annotations

import json
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "src" / "powergrid" / "data"
MAPS_ROOT = DATA_ROOT / "maps"
RULES_ROOT = DATA_ROOT / "rules"

REFERENCE_ROOT = Path("/tmp/powergrid_reference/Resources/config")

GERMANY_XML = REFERENCE_ROOT / "map" / "Germanymap.xml"
USA_XML = REFERENCE_ROOT / "map" / "USAMap.xml"
CONFIG_XML = REFERENCE_ROOT / "Config.xml"

GERMANY_REGION_ORDER = [
    ("#0055ff", "blue", "Blue"),
    ("#ff55ff", "magenta", "Magenta"),
    ("#000000", "black", "Black"),
    ("#ffff00", "yellow", "Yellow"),
    ("#55aa00", "green", "Green"),
    ("#55aaff", "cyan", "Cyan"),
]

USA_REGION_ORDER = [
    ("#aa55ff", "purple", "Purple"),
    ("#00ffff", "cyan", "Cyan"),
    ("#f9f900", "yellow", "Yellow"),
    ("#aa5500", "brown", "Brown"),
    ("#ff55ff", "magenta", "Magenta"),
    ("#00aa00", "green", "Green"),
]

GERMANY_NAME_FIXES = {
    "Augsbug": "Augsburg",
    "Dusseldorf": "Duesseldorf",
    "Flensberg": "Flensburg",
    "Fluida": "Fulda",
    "Frankfurt-M": "Frankfurt-Main",
    "Frankfurt-O": "Frankfurt-Oder",
    "Freibrurg": "Freiburg",
    "Hamberg": "Hamburg",
    "Koln": "Koeln",
    "Lubeck": "Luebeck",
    "Magdeberg": "Magdeburg",
    "Muchen": "Muenchen",
    "Munster": "Muenster",
    "Nurnburg": "Nuernberg",
    "Osnabruck": "Osnabrueck",
    "Regensberg": "Regensburg",
    "Saarbrucken": "Saarbruecken",
    "Shwerin": "Schwerin",
    "Wurzburg": "Wuerzburg",
}

USA_NAME_FIXES = {
    "Protland": "Portland",
    "Washington D.C.": "Washington D.C.",
    "St. Louis": "St. Louis",
}

GERMANY_CITY_SPECS = [
    "Flensburg 1 Kiel 4",
    "Kiel 1 Hamburg 8 Luebeck 4",
    "Cuxhaven 1 Bremen 8 Hamburg 11",
    "Wilhelmshaven 1 Osnabrueck 14 Bremen 11",
    "Hamburg 1 Bremen 11 Hannover 17 Schwerin 8",
    "Bremen 1 Osnabrueck 11 Hannover 10",
    "Hannover 1 Kassel 15 Erfurt 19 Magdeburg 15",
    "Luebeck 2 Hamburg 6 Schwerin 6",
    "Rostock 2 Schwerin 6 Torgelow 19",
    "Schwerin 2 Hannover 19 Magdeburg 16 Berlin 18 Torgelow 19",
    "Torgelow 2 Berlin 15",
    "Berlin 2 Magdeburg 10 Halle 17 Frankfurt-Oder 6",
    "Magdeburg 2 Halle 11",
    "Frankfurt-Oder 2 Leipzig 21 Dresden 16",
    "Osnabrueck 3 Muenster 7 Hannover 16 Kassel 20",
    "Muenster 3 Essen 6 Dortmund 2",
    "Duisburg 3 Essen 0",
    "Essen 3 Duesseldorf 2 Dortmund 4",
    "Dortmund 3 Koeln 10 Frankfurt-Main 20 Kassel 18",
    "Kassel 3 Frankfurt-Main 13 Fulda 8 Erfurt 15",
    "Duesseldorf 3 Aachen 9 Koeln 4",
    "Halle 4 Erfurt 6 Leipzig 0",
    "Leipzig 4 Dresden 13",
    "Erfurt 4 Fulda 13 Nuernberg 21 Dresden 19",
    "Dresden 4",
    "Fulda 4 Frankfurt-Main 8 Wuerzburg 11",
    "Wuerzburg 4 Mannheim 10 Stuttgart 12 Augsburg 19 Nuernberg 8",
    "Nuernberg 4 Augsburg 18 Regensburg 12",
    "Koeln 5 Aachen 7 Trier 20 Wiesbaden 21",
    "Aachen 5 Trier 19",
    "Frankfurt-Main 5 Wiesbaden 0 Wuerzburg 13",
    "Wiesbaden 5 Trier 18 Saarbruecken 10 Mannheim 11",
    "Trier 5 Saarbruecken 11",
    "Mannheim 5 Saarbruecken 11 Stuttgart 6",
    "Saarbruecken 5 Stuttgart 17",
    "Regensburg 6 Augsburg 13 Muenchen 10 Passau 12",
    "Stuttgart 6 Freiburg 16 Konstanz 16 Augsburg 15",
    "Augsburg 6 Konstanz 17 Muenchen 6",
    "Passau 6 Muenchen 14",
    "Freiburg 6 Konstanz 14",
    "Muenchen 6",
    "Konstanz 6",
]


def slugify(name: str) -> str:
    ascii_name = (
        unicodedata.normalize("NFKD", name)
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
    )
    for old, new in (
        (" ", "_"),
        ("-", "_"),
        (".", ""),
        (",", ""),
        ("'", ""),
        ("/", "_"),
    ):
        ascii_name = ascii_name.replace(old, new)
    while "__" in ascii_name:
        ascii_name = ascii_name.replace("__", "_")
    return ascii_name.strip("_")


def ensure_reference_files() -> None:
    missing = [path for path in (GERMANY_XML, USA_XML, CONFIG_XML) if not path.exists()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise SystemExit(
            "Missing reference files. Expected these local paths from the earlier "
            f"research step: {missing_text}"
        )


def build_rules_payload() -> dict[str, object]:
    return {
        "starting_money": 50,
        "houses_per_player": 22,
        "resource_supply": {
            "coal": 24,
            "oil": 24,
            "garbage": 24,
            "uranium": 12,
        },
        "resource_market_tracks": {
            "coal": {
                "capacity_by_price": {str(price): 3 for price in range(1, 9)},
                "starting_prices": [1, 2, 3, 4, 5, 6, 7, 8],
            },
            "oil": {
                "capacity_by_price": {str(price): 3 for price in range(1, 9)},
                "starting_prices": [3, 4, 5, 6, 7, 8],
            },
            "garbage": {
                "capacity_by_price": {str(price): 3 for price in range(1, 9)},
                "starting_prices": [6, 7, 8],
            },
            "uranium": {
                "capacity_by_price": {
                    "1": 1,
                    "2": 1,
                    "3": 1,
                    "4": 1,
                    "5": 1,
                    "6": 1,
                    "7": 1,
                    "8": 1,
                    "10": 1,
                    "12": 1,
                    "14": 1,
                    "16": 1,
                },
                "starting_prices": [14, 16],
            },
        },
        "payment_schedule": {
            "0": 10,
            "1": 22,
            "2": 33,
            "3": 44,
            "4": 54,
            "5": 64,
            "6": 73,
            "7": 82,
            "8": 90,
            "9": 98,
            "10": 105,
            "11": 112,
            "12": 118,
            "13": 124,
            "14": 129,
            "15": 134,
            "16": 138,
            "17": 142,
            "18": 145,
            "19": 148,
            "20": 150,
        },
        "player_count_rules": {
            "3": {
                "areas": 3,
                "remove_plug_plants": 2,
                "remove_socket_plants": 6,
                "max_power_plants": 3,
                "step_2_cities": 7,
                "end_game_cities": 17,
                "resource_refill": {
                    "step_1": {"coal": 4, "oil": 2, "garbage": 1, "uranium": 1},
                    "step_2": {"coal": 5, "oil": 3, "garbage": 2, "uranium": 1},
                    "step_3": {"coal": 3, "oil": 4, "garbage": 3, "uranium": 1},
                },
            },
            "4": {
                "areas": 4,
                "remove_plug_plants": 1,
                "remove_socket_plants": 3,
                "max_power_plants": 3,
                "step_2_cities": 7,
                "end_game_cities": 17,
                "resource_refill": {
                    "step_1": {"coal": 5, "oil": 3, "garbage": 2, "uranium": 1},
                    "step_2": {"coal": 6, "oil": 4, "garbage": 3, "uranium": 2},
                    "step_3": {"coal": 4, "oil": 5, "garbage": 4, "uranium": 2},
                },
            },
            "5": {
                "areas": 5,
                "remove_plug_plants": 0,
                "remove_socket_plants": 0,
                "max_power_plants": 3,
                "step_2_cities": 7,
                "end_game_cities": 15,
                "resource_refill": {
                    "step_1": {"coal": 5, "oil": 4, "garbage": 3, "uranium": 2},
                    "step_2": {"coal": 7, "oil": 5, "garbage": 3, "uranium": 3},
                    "step_3": {"coal": 5, "oil": 6, "garbage": 5, "uranium": 2},
                },
            },
            "6": {
                "areas": 5,
                "remove_plug_plants": 0,
                "remove_socket_plants": 0,
                "max_power_plants": 3,
                "step_2_cities": 6,
                "end_game_cities": 14,
                "resource_refill": {
                    "step_1": {"coal": 7, "oil": 5, "garbage": 3, "uranium": 2},
                    "step_2": {"coal": 9, "oil": 6, "garbage": 5, "uranium": 3},
                    "step_3": {"coal": 6, "oil": 7, "garbage": 6, "uranium": 3},
                },
            },
        },
        "setup": {
            "current_market_size": 4,
            "future_market_size": 4,
            "starting_plug_market_count": 8,
            "set_aside_plug_after_market": 1,
            "step_3_card_count": 1,
        },
    }


def build_power_plants_payload() -> list[dict[str, object]]:
    root = ET.parse(CONFIG_XML).getroot()
    payload: list[dict[str, object]] = []
    for plant in root.find("cards").findall("powerPlantCard"):
        price = int(plant.attrib["price"])
        resource_types = [resource.attrib["name"] for resource in plant.findall("resource")]
        payload.append(
            {
                "price": price,
                "resource_types": resource_types,
                "resource_cost": int(plant.attrib["resources"]),
                "output_cities": int(plant.attrib["power"]),
                "deck_back": "plug" if price <= 15 else "socket",
                "is_hybrid": len(resource_types) > 1,
                "is_ecological": len(resource_types) == 0,
            }
        )
    payload.sort(key=lambda plant: int(plant["price"]))
    return payload


def build_regions(region_order: list[tuple[str, str, str]]) -> list[dict[str, str]]:
    return [
        {"id": region_id, "label": label, "color": color}
        for color, region_id, label in region_order
    ]


def build_map_from_xml(
    *,
    map_id: str,
    name: str,
    xml_path: Path,
    region_order: list[tuple[str, str, str]],
    name_fixes: dict[str, str],
    special_rules: list[str],
) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    root = ET.parse(xml_path).getroot()
    region_lookup = {color: region_id for color, region_id, _ in region_order}
    cities: list[dict[str, str]] = []
    city_lookup: dict[str, dict[str, str]] = {}
    for city in root.find("cities").findall("city"):
        original_name = city.attrib["name"]
        fixed_name = name_fixes.get(original_name, original_name)
        entry = {
            "id": slugify(fixed_name),
            "name": fixed_name,
            "region": region_lookup[city.attrib["region"]],
        }
        cities.append(entry)
        city_lookup[fixed_name] = entry
    cities.sort(key=lambda item: item["id"])
    return cities, city_lookup


def build_germany_map_payload() -> dict[str, object]:
    cities, city_lookup = build_map_from_xml(
        map_id="germany",
        name="Germany",
        xml_path=GERMANY_XML,
        region_order=GERMANY_REGION_ORDER,
        name_fixes=GERMANY_NAME_FIXES,
        special_rules=["germany_no_uranium_resupply_after_plant_39"],
    )
    if "Mannheim" not in city_lookup:
        mannheim = {"id": "mannheim", "name": "Mannheim", "region": "green"}
        cities.append(mannheim)
        city_lookup["Mannheim"] = mannheim
        cities.sort(key=lambda item: item["id"])

    connections: set[tuple[str, str, int]] = set()
    for spec in GERMANY_CITY_SPECS:
        tokens = spec.split()
        city_name = tokens[0]
        if tokens[1] not in {"1", "2", "3", "4", "5", "6"}:
            city_name = f"{tokens[0]} {tokens[1]}"
            start_index = 3
        else:
            start_index = 2
        city_id = city_lookup[city_name]["id"]
        remaining = tokens[start_index:]
        index = 0
        while index < len(remaining):
            neighbor_name = remaining[index]
            if index + 2 < len(remaining) and not remaining[index + 1].isdigit():
                neighbor_name = f"{remaining[index]} {remaining[index + 1]}"
                cost = int(remaining[index + 2])
                index += 3
            else:
                cost = int(remaining[index + 1])
                index += 2
            neighbor_id = city_lookup[neighbor_name]["id"]
            ordered = tuple(sorted((city_id, neighbor_id)))
            connections.add((ordered[0], ordered[1], cost))

    return {
        "id": "germany",
        "name": "Germany",
        "regions": build_regions(GERMANY_REGION_ORDER),
        "cities": cities,
        "connections": [
            {"city_1": city_1, "city_2": city_2, "cost": cost}
            for city_1, city_2, cost in sorted(connections)
        ],
        "special_rules": ["germany_no_uranium_resupply_after_plant_39"],
    }


def build_usa_map_payload() -> dict[str, object]:
    root = ET.parse(USA_XML).getroot()
    region_lookup = {color: region_id for color, region_id, _ in USA_REGION_ORDER}
    cities: list[dict[str, str]] = []
    for city in root.find("cities").findall("city"):
        fixed_name = USA_NAME_FIXES.get(city.attrib["name"], city.attrib["name"])
        cities.append(
            {
                "id": slugify(fixed_name),
                "name": fixed_name,
                "region": region_lookup[city.attrib["region"]],
            }
        )
    cities.sort(key=lambda item: item["id"])

    city_lookup = {city["name"]: city["id"] for city in cities}
    connections: set[tuple[str, str, int]] = set()
    for connection in root.find("connections").findall("connection"):
        city_1_name = USA_NAME_FIXES.get(connection.attrib["first"], connection.attrib["first"])
        city_2_name = USA_NAME_FIXES.get(connection.attrib["second"], connection.attrib["second"])
        city_1 = city_lookup[city_1_name]
        city_2 = city_lookup[city_2_name]
        ordered = tuple(sorted((city_1, city_2)))
        connections.add((ordered[0], ordered[1], int(connection.attrib["cost"])))

    return {
        "id": "usa",
        "name": "USA",
        "regions": build_regions(USA_REGION_ORDER),
        "cities": cities,
        "connections": [
            {"city_1": city_1, "city_2": city_2, "cost": cost}
            for city_1, city_2, cost in sorted(connections)
        ],
        "special_rules": ["usa_coal_storage_space"],
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")


def main() -> None:
    ensure_reference_files()
    write_json(RULES_ROOT / "rule_tables.json", build_rules_payload())
    write_json(RULES_ROOT / "power_plants.json", build_power_plants_payload())
    write_json(MAPS_ROOT / "germany.json", build_germany_map_payload())
    write_json(MAPS_ROOT / "usa.json", build_usa_map_payload())
    print("Wrote Stage 1 data files.")


if __name__ == "__main__":
    main()
