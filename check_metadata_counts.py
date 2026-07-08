#!/usr/bin/env python3
import argparse
import statistics
import shutil
import sys
from pathlib import Path

import yaml


REQUIRED_TOPICS = [
    "/camera_head/camera_head/color/image_raw",
    "/camera_head/camera_head/aligned_depth_to_color/image_raw",
    "/camera_left_wrist/camera_left_wrist/aligned_depth_to_color/image_raw",
    "/camera_left_wrist/camera_left_wrist/color/image_rect_raw",
    "/camera_right_wrist/camera_right_wrist/aligned_depth_to_color/image_raw",
    "/camera_right_wrist/camera_right_wrist/color/image_rect_raw",
]

SHORT_NAMES = {
    "/camera_head/camera_head/color/image_raw": "head_color",
    "/camera_head/camera_head/aligned_depth_to_color/image_raw": "head_depth",
    "/camera_left_wrist/camera_left_wrist/aligned_depth_to_color/image_raw": "left_depth",
    "/camera_left_wrist/camera_left_wrist/color/image_rect_raw": "left_color",
    "/camera_right_wrist/camera_right_wrist/aligned_depth_to_color/image_raw": "right_depth",
    "/camera_right_wrist/camera_right_wrist/color/image_rect_raw": "right_color",
}


def load_topic_counts(metadata_path):
    with metadata_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    info = data.get("rosbag2_bagfile_information", {}) if isinstance(data, dict) else {}
    topics = info.get("topics_with_message_count", [])

    counts = {}
    for item in topics:
        if not isinstance(item, dict):
            continue
        topic_metadata = item.get("topic_metadata", {})
        name = topic_metadata.get("name") if isinstance(topic_metadata, dict) else None
        if name in REQUIRED_TOPICS:
            counts[name] = item.get("message_count")

    return counts


def check_metadata(metadata_path, max_diff):
    counts = load_topic_counts(metadata_path)
    problems = []
    under_max_diff = False

    for topic in REQUIRED_TOPICS:
        if topic not in counts:
            problems.append(f"missing topic: {topic}")
            continue

        count = counts[topic]
        if count is None or count == "":
            problems.append(f"empty message_count: {topic}")
        elif not isinstance(count, int):
            problems.append(f"non-integer message_count={count!r}: {topic}")
        elif count <= 0:
            problems.append(f"zero message_count: {topic}")

    valid_counts = [
        count for count in counts.values() if isinstance(count, int) and count > 0
    ]
    if len(valid_counts) == len(REQUIRED_TOPICS):
        reference_count = statistics.median(valid_counts)
        low_counts = {
            SHORT_NAMES[topic]: count
            for topic, count in counts.items()
            if reference_count - count > max_diff
        }
        if low_counts:
            under_max_diff = True
            problems.append(
                f"message_count too low: reference={reference_count:g}, "
                f"allowed_under={max_diff}, low={low_counts}"
            )

    return counts, problems, under_max_diff


def format_counts(counts):
    return ", ".join(
        f"{SHORT_NAMES[topic]}:{counts.get(topic, 'missing')}"
        for topic in REQUIRED_TOPICS
    )


def main():
    parser = argparse.ArgumentParser(
        description="Check target camera topic message_count values in rosbag metadata.yaml files."
    )
    parser.add_argument(
        "--root",
        nargs="?",
        default=".",
        help="Directory to scan recursively. Default: current directory.",
    )
    parser.add_argument(
        "--max-diff",
        type=int,
        default=5,
        help="Maximum allowed frame shortage compared with the median message count. Default: 5.",
    )
    parser.add_argument(
        "--delete-over-diff",
        action="store_true",
        help="Delete the whole bag folder when metadata check fails.",
    )
    args = parser.parse_args()

    metadata_files = sorted(Path(args.root).rglob("metadata.yaml"))
    if not metadata_files:
        print(f"No metadata.yaml found under {args.root}", file=sys.stderr)
        return 2

    failed = []
    delete_candidates = []
    for metadata_path in metadata_files:
        try:
            counts, problems, _under_max_diff = check_metadata(metadata_path, args.max_diff)
        except Exception as exc:
            failed.append(metadata_path)
            if args.delete_over_diff:
                delete_candidates.append(metadata_path)
            print(f"[FAIL] {metadata_path}: failed to parse metadata: {exc}")
            continue

        if problems:
            failed.append(metadata_path)
            if args.delete_over_diff:
                delete_candidates.append(metadata_path)
            print(f"[FAIL] {metadata_path}")
            print(f"       counts: {format_counts(counts)}")
            for problem in problems:
                print(f"       - {problem}")
        else:
            print(f"[ OK ] {metadata_path}: {format_counts(counts)}")

    deleted = []
    if args.delete_over_diff and delete_candidates:
        print()
        for metadata_path in delete_candidates:
            bag_dir = metadata_path.parent
            shutil.rmtree(bag_dir)
            deleted.append(bag_dir)
            print(f"[DELETE] {bag_dir}")

    print()
    print(f"Checked {len(metadata_files)} metadata.yaml files, failed {len(failed)}.")
    if args.delete_over_diff:
        print(f"Deleted {len(deleted)} failed bag folders.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
