import awkward as ak
import numpy as np

# The decay mode definition lives in ml-tau-data so that the label derived here
# and the `gen_jet_tau_decaymode` stored in the ntuples cannot drift apart.  It
# classifies daughters by particle property rather than by an enumerated PDG
# list, ignores photons (a radiative decay is classed with its parent mode, as
# PDG treats them), and ignores neutrinos.
#
# ml-tau-data has to be importable for this: as the submodule, or pip installed,
# or on PYTHONPATH.
from ntupelizer.tools.tau_decaymode import RARE_DECAY_MODE_EXT, classify_decay_modes

from mltau.tools.general import reinitialize_p4
from mltau.tools.meson_classes import get_meson_classes

# The set model predicts meson CLASSES, not species. For the decay mode a class
# is represented by one hadron that stands for it: any charged hadron for a
# charged class, any neutral hadron for a neutral one. These are the same
# representative ids the ntupelizer writes into gen_jet_tau_vis_daughter_pdgs
# (map_pdgid_to_candid), and ml-tau-data classifies them by property, so a
# class and a stored daughter go through the identical code path.
CHARGED_HADRON_PDG = 211
NEUTRAL_HADRON_PDG = 130


def meson_class_representative_pdgs(tau_daughter_pdg_ids) -> list[int]:
    """One representative |PDG| per configured meson class, in class order."""
    representatives = []
    for meson_class in get_meson_classes(tau_daughter_pdg_ids):
        charges = set(meson_class.charges)
        if charges == {0}:
            representatives.append(NEUTRAL_HADRON_PDG)
        elif 0 not in charges:
            representatives.append(CHARGED_HADRON_PDG)
        else:
            raise ValueError(
                f"Meson class '{meson_class.name}' mixes neutral and charged particles."
            )
    return representatives


def meson_class_to_pdg(meson_class, tau_daughter_pdg_ids):
    """Jagged meson-class indices -> jagged representative PDG ids."""
    representatives = np.asarray(
        meson_class_representative_pdgs(tau_daughter_pdg_ids), dtype=np.int64
    )
    counts = ak.num(meson_class, axis=1)
    flat = ak.to_numpy(ak.flatten(meson_class, axis=1)).astype(np.int64)
    return ak.unflatten(representatives[flat], counts)


def get_decay_mode(pdg):
    """Decay mode per jet from the daughter PDG ids.

    Uses the extended rare id (30): 15, the value the ntuples store, is also a
    reachable point on the 5*(n_charged-1)+n_neutral grid (four prongs, no
    neutrals), so under the normal convention a rare decay and a four-prong
    reconstruction are the same number.  A set model can predict four prongs --
    no tau decays that way, so it is a failure worth seeing -- and 30 keeps it
    visible.  Compare derived with derived: a mode from here is not directly
    comparable with the ntuple column, which says 15.
    """
    return ak.Array(classify_decay_modes(pdg, rare=RARE_DECAY_MODE_EXT))


def construct_jet_level_predictions(
    pred_daughters, true_daughters, tau_daughter_pdg_ids
):
    pred_pdg = meson_class_to_pdg(pred_daughters.meson_class, tau_daughter_pdg_ids)
    true_pdg = meson_class_to_pdg(true_daughters.meson_class, tau_daughter_pdg_ids)

    pred_tau_decay_mode = get_decay_mode(pred_pdg)
    pred_tau_p4 = reinitialize_p4(ak.sum(pred_daughters.p4, axis=1))
    pred_tau_charge = ak.sum(pred_daughters.charge, axis=1)

    true_tau_decay_mode_exp = get_decay_mode(true_pdg)
    return ak.Array(
        {
            "tau_decaymode": pred_tau_decay_mode,
            "tau_p4": pred_tau_p4,
            "tau_charge": pred_tau_charge,
            "gen_jet_tau_decaymode_exp": true_tau_decay_mode_exp,
        }
    )


def construct_prediction_file_content(
    data,
    pred_daughters,
    true_daughters,
    tau_daughter_pdg_ids,
    tau_tagging_score=None,
    debug=False,
):
    fields_of_interest = [
        "reco_jet_p4",
        "gen_jet_p4",
        "reco_cand_p4s",
        "reco_cand_pdgs",
        "reco_cand_charges",
        "gen_jet_tau_vis_energy",
        "gen_jet_tau_decaymode",
        "gen_jet_tau_charge",
        "gen_jet_tau_full_p4",
        "gen_jet_tau_vis_daughter_p4s",
        "gen_jet_tau_vis_daughter_pdgs",
        "gen_jet_tau_vis_daughter_charges",
        "gen_jet_tau_p4",
    ]
    if debug:
        fields_of_interest.extend([
            "file_id",
            "event_id",
        ])
    data_of_interest = ak.Array(data[fields_of_interest])
    pred_tau_daughter_data = ak.Array(
        {
            "pred_tau_daughter_meson_classes": pred_daughters.meson_class,
            "pred_tau_daughter_p4s": pred_daughters.p4,
            "pred_tau_daughter_charges": pred_daughters.charge,
        }
    )
    pred_tau_jet_level_data = construct_jet_level_predictions(
        pred_daughters, true_daughters, tau_daughter_pdg_ids
    )
    output_fields = {
        **{f: data_of_interest[f] for f in ak.fields(data_of_interest)},
        **{f: pred_tau_daughter_data[f] for f in ak.fields(pred_tau_daughter_data)},
        **{
            f: pred_tau_jet_level_data[f]
            for f in ak.fields(pred_tau_jet_level_data)
        },
    }
    if tau_tagging_score is not None:
        output_fields["tau_tagging_score"] = tau_tagging_score
    combined_data = ak.zip(output_fields, depth_limit=1)
    return combined_data


def construct_background_prediction_file_content(data, tau_tagging_score):
    """Build the minimal jet-level output for a background sample."""
    fields_of_interest = ["reco_jet_p4", "gen_jet_p4"]
    fields_of_interest.extend(
        field for field in ("file_id", "event_id") if field in ak.fields(data)
    )
    return ak.zip(
        {
            **{field: data[field] for field in fields_of_interest},
            "tau_tagging_score": tau_tagging_score,
        },
        depth_limit=1,
    )
