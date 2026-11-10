import json

from mechcal.eval.tool_sources import build_tool_source_packet
from test_tool_sources import execution, save_execution


def test_coefficient_projection_is_explicit_with_sources_and_no_state_loss(tmp_path):
    record = execution(status="partial")
    values = record["structured_results"]["parsed_observables"]
    rows = [{"from_orbital": i + 1, "to_orbital": i + 3000, "coefficient": -i / 100000}
            for i in range(2000)]
    values["dominant_transitions"] = [{"state_index": 1, "components": rows}]
    values["atom_contributions"] = list(range(120))
    values["spatial_descriptors"] = {"D_angstrom": 2.089, "Sr": .77322}
    path = save_execution(tmp_path, record)
    before = path.read_bytes()
    packet = build_tool_source_packet(tmp_path, ["R001:scan:e"])
    result = packet.tool_records[0]["structured_results"]["parsed_observables"]
    assert result["route_records"] == values["route_records"]
    assert result["missing_descriptors"] == values["missing_descriptors"]
    assert result["atom_contributions"] == values["atom_contributions"]
    assert result["spatial_descriptors"] == values["spatial_descriptors"]
    table = result["dominant_transitions"][0]
    assert table["component_table_view"]["total_rows"] == 2000
    assert table["component_table_view"]["omitted_rows"] == 1992
    assert table["components"] == list(reversed(rows[-8:]))
    assert path.read_bytes() == before
    assert json.loads(before)["structured_results"]["parsed_observables"]["dominant_transitions"][0]["components"] == rows
    assert packet.tool_records[0]["source_record"] == {"packet_ref": "/execution_index/0"}
    assert packet.execution_index[0]["source_file"] and packet.execution_index[0]["source_sha256"]
