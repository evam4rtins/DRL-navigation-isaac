"""
Single entry point of the camera student.

python scripts/student.py <command> --config configs/student/<condition>.yaml [--set section.field=value ...]

Commands
    record    the teacher drives the vision task, its labels are written to disk
    distill   offline behaviour cloning on recorded data

Examples:
    scripts/student.py record  --config configs/student/actions.yaml --set "record.maps=[3,4,8,20]" \
        --set record.out=logs/datasets/teacher_v1/a --headless
    scripts/student.py distill --config configs/student/actions.yaml
    scripts/student.py distill --config configs/student/full.yaml --set distill.epochs=2 --set data.max_chunks=6
"""
 
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

PARTITION_KIT_ARG = "--/renderer/scenePartitioning/enabled=true"  


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("record", "distill"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        p.add_argument("--set", action="append", default=[], metavar="section.field=value")
        if name == "record" and len(sys.argv) > 1 and sys.argv[1] == "record":
            from isaaclab.app import AppLauncher

            AppLauncher.add_app_launcher_args(p)
    args = parser.parse_args()

    if args.command == "record":
        from isaaclab.app import AppLauncher

        args.enable_cameras = True
        args.kit_args = ((args.kit_args or "") + " " + PARTITION_KIT_ARG).strip()
        app = AppLauncher(args).app
        try:
            from student import recorder
            from student.config import load_config

            recorder.run(load_config(args.config, args.set))
        except Exception:
            import traceback

            traceback.print_exc()
        app.close()
        return

    from student.config import load_config

    cfg = load_config(args.config, args.set)
    if args.command == "distill":
        from student import distill

        distill.run(cfg)


if __name__ == "__main__":
    main()
