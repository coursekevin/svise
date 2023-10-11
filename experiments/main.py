import argparse
import os
import warnings
from pathlib import Path
from enum import Enum

parser = argparse.ArgumentParser(description="Main experiment launch utility.")

class ExpList(Enum):
    PURE_SE = ("pure-se", "1_pure_state_estimation")
    CORRUPTED_SE = ("corrupted-se", "2_probabilistic_corruptions")
    CYLINDER_FLOW = ("cylinder-flow", "3_cylinder_flow")
    SYMB_GOVEQ_DISC = ("symb-goveq-disc", "4_symbolic_goveq_discovery")
    EXT_GOVEQ_DISC = ("ext-goveq-disc", "5_extended_goveq_discovery")
    SECOND_ORDER_DISC = ("second-order-disc", "6_second_order_system")
    BINARY_BLACK_HOLE = ("binary-black-hole", "7_binary_black_hole")

exp_list = [e.value[0] for e in ExpList]
parser.add_argument("experiment", type=str, choices=exp_list, help="experiment choice")
act_list = ["generate-data", "run-svise", "run-pf", "run-sindy", "post-process"]
parser.add_argument(
    "action", type=str, choices=act_list, help="what to do in experiment"
)
parser.add_argument(
    "-dp", "--dpath", type=str, help="path to dataset to perform action", nargs="?"
)
parser.add_argument("-rs", "--random_seed", type=int, help="random seed", nargs="?")

rs_warning_str = "random seed is ignored for chosen experiment + action."
dpath_warning_str = "data path argument ignored for chosen experiment + action."

def get_dpath_rs(call, args, exp_path):
    if args.random_seed is not None:
        call += ["--rs", f"{args.random_seed}"]
    if args.dpath:
        dpath_err = f"{args.dpath} not in folder {exp_path}"
        assert Path(args.dpath).parts[0] == str(exp_path), dpath_err
        call += ["--dpath", f"{args.dpath}"]
    else:
        raise ValueError(f"--dpath arg required for {args.action}")
    return call



def main(args):
    call = ["python"]
    if args.experiment == ExpList.PURE_SE.value[0]:
        path = Path(ExpList.PURE_SE.value[1])
        (path / "results" / "models").mkdir(parents=True, exist_ok=True)
        (path / "results" / "pngs").mkdir(parents=True, exist_ok=True)
        if args.action == "generate-data":  # generate data
            if args.random_seed is not None:
                warnings.warn(rs_warning_str)
            if args.dpath is not None:
                warnings.warn(dpath_warning_str)
            call.append(str(path / "generate_data.py"))
        elif args.action == "run-svise":  # state est w/o governing equations
            call.append(str(path / "experiment_util.py"))
            call = get_dpath_rs(call, args, path)
        elif args.action == "run-pf":  # state estimation benchmarking
            call.append(str(path / "da_util.py"))
            call = get_dpath_rs(call, args, path)
        elif args.action == "post-process":  # post processing + plots
            if args.random_seed is not None:
                warnings.warn(rs_warning_str)
            if args.dpath is not None:
                warnings.warn(dpath_warning_str)
            call.append(str(path / "state_est_post_process.py"))
            os.system(" ".join(call))
            call = ["python", str(path / "state_est_plots.py")]
        else:
            raise NotImplementedError(
                f"{args.action} not implemented for {args.experiment}"
            )

    elif args.experiment == ExpList.CORRUPTED_SE.value[0]:
        path = Path(ExpList.CORRUPTED_SE.value[1])
        (path / "results" / "pngs").mkdir(parents=True, exist_ok=True)
        if args.action == "generate-data":  # generate data
            if args.random_seed is not None:
                warnings.warn(rs_warning_str)
            if args.dpath is not None:
                warnings.warn(dpath_warning_str)
            call.append(str(path / "generate_data.py"))
        elif args.action == "run-svise":  # state est w/o governing equations
            call.append(str(path / "experiment_util.py"))
            call = get_dpath_rs(call, args, path)
        elif args.action == "run-pf":  # state estimation benchmarking
            call.append(str(path / "da_util.py"))
            call = get_dpath_rs(call, args, path)
        elif args.action == "post-process":  # post processing + plots
            if args.random_seed is not None:
                warnings.warn(rs_warning_str)
            if args.dpath is not None:
                warnings.warn(dpath_warning_str)
            call.append(str(path / "state_est_post_process.py"))
            os.system(" ".join(call))
            call = ["python", str(path / "state_est_plots.py")]
        else:
            raise NotImplementedError(
                f"{args.action} not implemented for {args.experiment}"
            )

    elif args.experiment == ExpList.SYMB_GOVEQ_DISC.value[0]:
        path = Path(ExpList.SYMB_GOVEQ_DISC.value[1])
        (path / "results" / "pngs").mkdir(parents=True, exist_ok=True)
        if args.action == "generate-data":  # generate data
            if args.random_seed is not None:
                warnings.warn(rs_warning_str)
            if args.dpath is not None:
                warnings.warn(dpath_warning_str)
            call.append(str(path / "generate_data.py"))
        elif args.action == "run-svise":  # state est w/o governing equations
            call.append(str(path / "experiment_util.py"))
            call = get_dpath_rs(call, args, path)
        elif args.action == "run-sindy":  # sindy comparison bmarks
            call.append(str(path / "comparison_util.py"))
            call = get_dpath_rs(call, args, path)
        elif args.action == "post-process":  # post processing + plots
            if args.random_seed is not None:
                warnings.warn(rs_warning_str)
            if args.dpath is not None:
                warnings.warn(dpath_warning_str)
            call.append(str(path / "post_process.py"))
            os.system(" ".join(call))
            os.system(" ".join(["python", str(path / "tables.py")]))
            call = ["python", str(path / "box_plots_sns.py")]
        else:
            raise NotImplementedError(
                f"{args.action} not implemented for {args.experiment}"
            )
    elif args.experiment == ExpList.SECOND_ORDER_DISC.value[0]:
        path = Path(ExpList.SECOND_ORDER_DISC.value[1])
        (path / "results" / "pngs").mkdir(parents=True, exist_ok=True)
        if args.action == "generate-data":  # generate data
            if args.random_seed is not None:
                warnings.warn(rs_warning_str)
            if args.dpath is not None:
                warnings.warn(dpath_warning_str)
            call.append(str(path / "generate_data.py"))
        elif args.action == "run-svise":  # state est w/o governing equations
            call.append(str(path / "experiment_util.py"))
            call = get_dpath_rs(call, args, path)
        elif args.action == "post-process":  # post processing + plots
            if args.random_seed is not None:
                warnings.warn(rs_warning_str)
            if args.dpath is not None:
                warnings.warn(dpath_warning_str)
            call.append(str(path / "plot_trajectories.py"))
        else:
            raise NotImplementedError(
                f"{args.action} not implemented for {args.experiment}"
            )
    elif args.experiment == ExpList.EXT_GOVEQ_DISC.value[0]:
        path = Path(ExpList.EXT_GOVEQ_DISC.value[1])
        if args.action == "generate-data":  # generate data
            if args.random_seed is not None:
                warnings.warn(rs_warning_str)
            if args.dpath is not None:
                warnings.warn(dpath_warning_str)
            call.append(str(path / "generate_data.py"))
        elif args.action == "run-svise":  # state est w/o governing equations
            call.append(str(path / "experiment_util.py"))
            call = get_dpath_rs(call, args, path)
        elif args.action == "post-process":  # post processing + plots
            if args.random_seed is not None:
                warnings.warn(rs_warning_str)
            if args.dpath is not None:
                warnings.warn(dpath_warning_str)
            call.append(str(path / "post_process.py"))
            os.system(" ".join(call))
            call = ["python", str(path / "plots.py")]
        else:
            raise NotImplementedError(
                f"{args.action} not implemented for {args.experiment}"
            )
    elif args.experiment == ExpList.CYLINDER_FLOW.value[0]:
        path = Path(ExpList.CYLINDER_FLOW.value[1])
        if args.action == "generate-data":  # generate data
            if args.random_seed is not None:
                warnings.warn(rs_warning_str)
            if args.dpath is not None:
                warnings.warn(dpath_warning_str)
            call.append(str(path / "generate_data.py"))
        elif args.action == "run-svise":  # state est w/o governing equations
            call.append(str(path / "train_nsde.py"))
        elif args.action == "post-process":  # post processing + plots
            if args.random_seed is not None:
                warnings.warn(rs_warning_str)
            if args.dpath is not None:
                warnings.warn(dpath_warning_str)
            call.append(str(path / "plot_results.py"))
        else:
            raise NotImplementedError(
                f"{args.action} not implemented for {args.experiment}"
            )
    elif args.experiment == ExpList.BINARY_BLACK_HOLE.value[0]:
        path = Path(ExpList.BINARY_BLACK_HOLE.value[1])
        if args.action == "generate-data":  # generate data
            if args.random_seed is not None:
                warnings.warn(rs_warning_str)
            if args.dpath is not None:
                warnings.warn(dpath_warning_str)
            call.append(str(path / "get_data.py"))
        elif args.action == "run-svise":  # state est w/o governing equations
            call.append(str(path / "experiment_util.py"))
        elif args.action == "post-process":  # post processing + plots
            if args.random_seed is not None:
                warnings.warn(rs_warning_str)
            if args.dpath is not None:
                warnings.warn(dpath_warning_str)
            call.append(str(path / "post_process.py"))
        else:
            raise NotImplementedError(
                f"{args.action} not implemented for {args.experiment}"
            )

    else:  # argparse should handle this case automatically
        raise ValueError(f"Experiment: {args.experiment} not recognized.")

    os.system(" ".join(call))


if __name__ == "__main__":
    main(parser.parse_args())
