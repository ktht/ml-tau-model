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
from mltau.tools.meson_classes import CHARGED_HADRON_PDG, NEUTRAL_HADRON_PDG

# The set model predicts meson CLASSES, not species. For the decay mode a class
# is represented by one hadron that stands for it: any charged hadron for a
# charged class, any neutral hadron for a neutral one. These are the same
# representative ids the ntupelizer writes into gen_jet_tau_vis_daughter_pdgs
# (map_pdgid_to_candid), and ml-tau-data classifies them by property, so a
# class and a stored daughter go through the identical code path.
def charge_to_representative_pdg(charge):
    """Map neutral daughters to 130 and charged daughters to 211."""
    return ak.where(charge == 0, NEUTRAL_HADRON_PDG, CHARGED_HADRON_PDG)


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
    pred_pdg = charge_to_representative_pdg(pred_daughters.charge)
    true_pdg = charge_to_representative_pdg(true_daughters.charge)

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
        pred_daughters,
        true_daughters,
        tau_daughter_pdg_ids,
    )
    combined_data = ak.zip(
        {
            **{f: data_of_interest[f] for f in ak.fields(data_of_interest)},
            **{f: pred_tau_daughter_data[f] for f in ak.fields(pred_tau_daughter_data)},
            **{
                f: pred_tau_jet_level_data[f]
                for f in ak.fields(pred_tau_jet_level_data)
            },
        },
        depth_limit=1,
    )
    return combined_data
