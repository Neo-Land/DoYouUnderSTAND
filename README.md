# Do You UnderSTAND?

Strike a JoJo pose and your Stand appears behind you. Switch modes and the same pipeline reads
American Sign Language.

Built with Python, MediaPipe (pose + hand landmarks, selfie segmentation), OpenCV and
scikit-learn.

## How it works
1. **Landmarks**: MediaPipe finds 33 body landmarks (pose mode) or 21 landmarks per hand (ASL mode).
2. **Features**: landmarks are made position- and scale-invariant (body: centered on the hips,
   scaled by torso length; hands: centered on the wrist, scaled by hand size).
3. **Classifier**: a Random Forest trained on samples I recorded myself.
4. **Smoothing**: a pose only triggers after it holds steady for several frames above a
   confidence threshold, so it doesn't flicker.
5. **Effects**: a segmentation mask puts the Stand *behind* you, with ゴゴゴ text and sound.

## Quick start
```powershell
./setup.ps1                         # creates .venv and installs dependencies
.\.venv\Scripts\Activate.ps1
python src/record.py --label neutral
python src/record.py --label jotaro
# ...repeat for each pose
python src/train.py --mode pose
python src/app.py --mode pose
```

No webcam? Record a short clip of each pose on another device and run
`python src/record.py --label jotaro --video path/to/clip.mov`.

## Poses
| Pose | Stand |
|---|---|
| Jotaro Kujo | Star Platinum |
| Giorno Giovanna | Gold Experience |
| Josuke Higashikata | Crazy Diamond |
| Bruno Bucciarati | Sticky Fingers |
| Diego Brando | Scary Monsters |
| Yoshikage Kira | Killer Queen |

## ASL mode
`--mode asl` uses hand landmarks to recognize fingerspelled letters. Planned: motion signs
(J, Z, and words) using a sequence model over landmark windows.

*JoJo's Bizarre Adventure is © Hirohiko Araki / Shueisha. This is a non-commercial fan project;
artwork and audio aren't included in the repo.*
