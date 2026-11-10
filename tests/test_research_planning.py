import json
import pytest
from research_planning import planning_prompt, research_packet, parse_plan


def plan():
    return dict(alternatives=[dict(id=str(i), mechanism="m", implementation="i", representation="r",
        learning_objective="o", solver="s", evidence_ids=["failure"], prediction="p", falsifier="f",
        structural_difference="d", analogy=dict(source="s",mapping="m",non_correspondence="n")) for i in range(3)],
        selected_id="1", selection_reason="relevant")


def test_failed_nonincumbent_feedback_and_private_exclusion():
    records=[dict(id="failure",data_role="public",status="error",failure="timeout"),
             dict(id="private",data_role="private",status="ok",score=-99)]
    packet=research_packet(records)
    assert packet["record_ids"]==["failure"]
    assert packet["records"][0]["failure"]=="timeout"
    assert "private" not in json.dumps(packet)


def test_parent_implementation_absent_only_from_mechanism_arm():
    args=dict(contract="contract",packet=research_packet([]),parent_source="SECRET_PARENT_BODY",round_index=0)
    assert "SECRET_PARENT_BODY" not in planning_prompt(arm="mechanism_first",**args)
    assert "SECRET_PARENT_BODY" in planning_prompt(arm="parent_visible",**args)


def test_unknown_evidence_cannot_be_promoted_to_plan():
    p=plan(); p["alternatives"][0]["evidence_ids"]=["invented"]
    with pytest.raises(ValueError,match="evidence"):
        parse_plan(json.dumps(p),{"record_ids":["failure"]})


def test_duplicate_selection_rejected():
    p=plan();p["alternatives"][1]["id"]="0"
    with pytest.raises(ValueError,match="ambiguous"):
        parse_plan(json.dumps(p),{"record_ids":["failure"]})


def test_round_two_packet_includes_new_outcome_changes_digest():
    records=[dict(id="seed",data_role="public",status="ok")]
    before=research_packet(records)
    records.append(dict(id="failure",data_role="public",status="error",failure="no fit"))
    after=research_packet(records)
    assert after["record_ids"]==["seed","failure"]
    assert after["source_records_digest"]!=before["source_records_digest"]
    assert parse_plan(json.dumps(plan()),after)["selected_id"]=="1"
