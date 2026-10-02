Any change to sandbox.py, Seatbelt profiles, or file permission logic requires Reviewer sign-off before merge.
The sandbox is write-confinement, not confidentiality — do not treat it as a full isolation boundary.
Never weaken read_regular_file validation, trusted_artifact_dir isolation, or symlink rejection.
