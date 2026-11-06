"""Do You UnderSTAND: JoJo pose recognition that summons Stands, and ASL word-sign recognition.

Module map (shared code; the command-line tools in scripts/ and app.py are thin wrappers around it):

    config       config.yaml loading and project-relative paths
    landmarks    MediaPipe wrapper (Detector): body + hands per frame, wrist-crop hand fallback, person mask
    features     JoJo poses: raw landmark row layout, pose features, mirroring, framing check
    clips        parse training clip names like jotaro_s1.mp4 -> ("jotaro", "s1")
    pose_model   JoJo poses: candidate classifiers and feature matrices, shared by training and evaluation
    smoothing    JoJo poses: hold-to-summon / dismiss state machine (PoseSmoother)
    live_eval    JoJo poses: replay recorded clips through the app's live logic and score it
    effects      JoJo poses: Stand rendering behind the user, ゴゴゴ text, sound
    signs        ASL: per-frame sign features, trimming to the signing span, resampling, mirroring
    sign_model   ASL: GRU sequence classifier (PyTorch, imported lazily)
    sign_live    ASL: live mode, cutting the stream into signs (SignSegmenter) and classifying them

Importing the package itself loads nothing heavy; MediaPipe and OpenCV are only imported by the
modules that need them, so the unit tests run without MediaPipe installed.
"""
