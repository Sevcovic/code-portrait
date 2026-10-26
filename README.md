# Code Portrait

Turn a photo into an animated portrait made of `. - > <` characters, then add up to three orange details like glasses, a hat, a watch or a dot on your t-shirt.

Everything runs in the browser. The photo is never uploaded anywhere.

## How it works

- [MediaPipe](https://ai.google.dev/edge/mediapipe) (on-device) removes the background and finds the face, eyes, mouth, shoulders and wrists.
- The photo is turned into a grid of characters: bright areas get dense characters (`<` `>`), mid-tones get `-`, shadows get `.`.
- Each orange detail is matched to a feature from your text ("add glasses", "orange dot on my t-shirt", "watch") and drawn in orange characters at the right spot.
- Export as PNG, SVG, plain text or an MP4 video of the animation (1080×1920, ready for Instagram stories).

## Supported details

glasses, sunglasses, hat, watch, heart or dot on the chest, earrings, mustache, bow tie, and highlighting hair, eyes, lips, face or shirt.

## Run locally

Serve the `site` folder with any static server, for example `npx serve site`.

## Credits

Made by Petr Ševcovic. MediaPipe Tasks Vision is © Google, Apache License 2.0 (runtime files in `site/mp`).

mp4-muxer by Vanilagy, MIT License (`site/lib/mp4-muxer.mjs`).
