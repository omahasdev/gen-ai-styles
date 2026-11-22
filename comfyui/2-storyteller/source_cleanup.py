"""Source cleanup: remove a fault of the photo at the frame edge before the image model sees it.

An out-of-focus object in front of the lens (a dark, soft blob that touches the edge of the frame) is not part of
the scene, but the image model copies its pixels into the redraw. The text rules of the writer cannot stop that,
so the source is cleaned: the blob is found and filled from its surroundings (OpenCV inpainting). The image model
redraws everything, so a smooth fill is enough.

A region counts as a fault only when all of these hold:
- it touches the frame edge and lies in the outer band of the frame,
- it is much darker than the ring of pixels around it on the inner side (a dark scene or a dark wall is not),
- its surface is soft (low detail, as an out-of-focus object is),
- it covers a small part of the frame,
- the frame is not low-key: in a dark scene (a cinema, a night street), dark edges are part of the look.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

WORK_SIDE = 400            # the analysis runs on a copy with this long side
EDGE_BAND = 0.15           # the outer band of the frame, as a part of the short side
DARK_RATIO = 0.6           # a fault pixel is darker than this part of the median of its inner surroundings
SOFT_MAX = 6.0             # mean absolute Laplacian of a soft (out-of-focus) region, at WORK_SIDE
MIN_AREA, MAX_AREA = 0.004, 0.08  # the part of the frame that a fault covers
RING_PX = 12               # the width of the ring around a region, at WORK_SIDE
GROW_PX = 6                # the mask grows by this much before the fill, so that the soft rim goes too
INPAINT_RADIUS = 9
MIN_SCENE_MEDIAN = 80       # of 255: the inner median brightness of a frame that is not low-key


def edge_fault_mask(image: np.ndarray) -> np.ndarray | None:
    """The mask (uint8, full size) of the faults at the frame edge, or None when the image has none."""
    h, w = image.shape[:2]
    scale = WORK_SIDE / max(h, w)
    small = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sh, sw = gray.shape
    band = max(1, round(min(sh, sw) * EDGE_BAND))
    inner = np.zeros_like(gray, dtype=bool)
    inner[band:-band, band:-band] = True
    scene = float(np.median(gray[inner]))
    if scene < MIN_SCENE_MEDIAN:
        return None  # a low-key scene: its dark edges belong to it
    dark = (gray < DARK_RATIO * scene).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    detail = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    mask = np.zeros_like(dark)
    for i in range(1, count):
        x, y, bw, bh, area = stats[i]
        part = area / (sh * sw)
        if not MIN_AREA <= part <= MAX_AREA:
            continue
        touches = x == 0 or y == 0 or x + bw == sw or y + bh == sh
        in_band = x + bw <= band * 2 or y + bh <= band * 2 or x >= sw - band * 2 or y >= sh - band * 2
        if not (touches and in_band):
            continue
        region = labels == i
        ring = cv2.dilate(region.astype(np.uint8), np.ones((RING_PX, RING_PX), np.uint8)).astype(bool) & ~region
        if not ring.any() or gray[region].mean() > DARK_RATIO * np.median(gray[ring]):
            continue  # as dark as its surroundings: a dark part of the scene, not an object in front of the lens
        if detail[region].mean() > SOFT_MAX:
            continue  # sharp: a part of the scene in focus
        mask[region] = 1
    if not mask.any():
        return None
    mask = cv2.dilate(mask, np.ones((GROW_PX, GROW_PX), np.uint8))
    return cv2.resize(mask * 255, (w, h), interpolation=cv2.INTER_NEAREST)


def clean(path: Path, out_dir: Path) -> tuple[Path, float]:
    """A cleaned copy of the image in out_dir and the part of the frame that was filled; the same path and 0 when
    the image has no fault."""
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return path, 0.0
    mask = edge_fault_mask(image)
    if mask is None:
        return path, 0.0
    filled = cv2.inpaint(image, mask, INPAINT_RADIUS, cv2.INPAINT_TELEA)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{path.stem}.clean.png"
    ok, data = cv2.imencode(".png", filled)
    if not ok:
        return path, 0.0
    data.tofile(str(out))
    return out, float((mask > 0).mean())


# Source softening (2026-11-20). The image model edits on top of the source and copies its contours: the face
# shapes, the eye size, and the pose stay photographic even when the prompt asks for a drawn style. A Gaussian blur
# of the copy that the image model gets keeps the layout, the colors, and the large shapes, but frees the drawing of
# the photo's edges. The writer still sees the sharp source. Checks on two ChatGPT sources (Anime, short prompts):
# 1 % of the long side changed nothing, 2.5 % gave a drawn face but moved the camera and replaced the props (the
# blur alone drops the core details, user 2026-11-20). Any visible photo edge locks the contours again: a 50/50
# mix of sharp and blurred, a sharp, posterized, or 192 px copy as a second reference all gave the semi-real face.
# A line drawing of the source as a second reference kept the props, but its face lines came back as false
# glasses (squint creases), and with a sharp scene it is not necessary, so it was dropped.
SOFTEN_MAX = 0.025            # the blur radius for a fully drawn style, as a part of the long side
SOFTEN_MIN_DISTANCE = 0.5     # a style nearer to a photograph than this gets no blur


def soften_fraction(photo_distance: float) -> float:
    """The blur radius (part of the long side) for a style at this distance from a photograph (0 photo, 1 drawn)."""
    return round(SOFTEN_MAX * photo_distance, 4) if photo_distance >= SOFTEN_MIN_DISTANCE else 0.0


# Only the people are blurred (user, 2026-11-20): a blur of the whole frame erased the core details of the scene
# (a cog-wheel decoration) and flattened its light (the bright slots of a tunnel); a light blur of the scene turned
# a painted forest into photo bokeh. The people carry the semi-real look (face shapes, skin and cloth shading), the
# scene carries the core details. The person mask comes from DeepLabV3 MobileNet (torchvision, the "person" class).
PERSON_THRESHOLD = 0.5
# The VOC classes of the mask. card28 (2026-11-20) added the animals for a cat source: the Anime cat lost its green
# eyes and its look, the Modern Happy and Arch cats got simpler and smaller than ChatGPT's; only the Poster body
# got more geometric. People only.
SUBJECT_CLASSES = ("person",)
PERSON_FEATHER = 0.01         # the soft rim of the mask, as a part of the long side
_SEGMENTER = None


def person_mask(image: np.ndarray) -> np.ndarray | None:
    """The person mask of a BGR image (float32, 0 to 1, full size, soft rim), or None without the segmentation model."""
    global _SEGMENTER
    try:
        import torch
        from torchvision.models.segmentation import (DeepLabV3_MobileNet_V3_Large_Weights,
                                                     deeplabv3_mobilenet_v3_large)
        if _SEGMENTER is None:
            weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
            _SEGMENTER = (deeplabv3_mobilenet_v3_large(weights=weights).eval(), weights)
        model, weights = _SEGMENTER
        rgb = torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
        with torch.no_grad():
            logits = model(weights.transforms()(rgb).unsqueeze(0))["out"][0]
        classes = [weights.meta["categories"].index(name) for name in SUBJECT_CLASSES]
        prob = torch.softmax(logits, 0)[classes].sum(0).numpy()
    except Exception as error:  # no torchvision, no weights, no network for the first download
        print(f"source soften: no person mask ({error.__class__.__name__}), the source stays sharp")
        return None
    h, w = image.shape[:2]
    hard = (cv2.resize(prob, (w, h)) > PERSON_THRESHOLD).astype(np.float32)
    return cv2.GaussianBlur(hard, (0, 0), max(h, w) * PERSON_FEATHER)


# The people crop (2026-11-20): the writer model reads the whole frame at a low resolution and misses small
# identity items (the thin glasses of a man in a tunnel). On a crop of the people, it names them.
PEOPLE_CROP_PAD = 0.05        # the margin around the people, as a part of the long side
# A small crop is enlarged to this long side: the model gets more vision tokens for it. On a 210 x 338 crop it
# missed the stubble, the chain, and the buckles of the hunter pair; on the same crop at 1024 it named them all.
PEOPLE_CROP_SIDE = 1024


def enlarged(path: Path, out_dir: Path, side: int) -> Path | None:
    """A copy of the image in out_dir enlarged to this long side (more vision tokens for a small source); None for a
    failed read or an image that is already as large."""
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or max(image.shape[:2]) >= side:
        return None
    scale = side / max(image.shape[:2])
    ok, data = cv2.imencode(".png", cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4))
    if not ok:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{path.stem}.big.png"
    data.tofile(str(out))
    return out


def people_crop(path: Path, out_dir: Path) -> Path | None:
    """A crop of the image around all people (with a margin) in out_dir, or None without people or a mask."""
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return None
    mask = person_mask(image)
    if mask is None or mask.max() < PERSON_THRESHOLD:
        return None
    ys, xs = np.where(mask >= PERSON_THRESHOLD)
    h, w = image.shape[:2]
    pad = round(max(h, w) * PEOPLE_CROP_PAD)
    crop = image[max(0, ys.min() - pad):min(h, ys.max() + pad + 1), max(0, xs.min() - pad):min(w, xs.max() + pad + 1)]
    scale = PEOPLE_CROP_SIDE / max(crop.shape[:2])
    if scale > 1:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
    ok, data = cv2.imencode(".png", crop)
    if not ok:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{path.stem}.people.png"
    data.tofile(str(out))
    return out


# The blur uses only the pixels of the people (normalized convolution, card35, 2026-11-20). A plain Gaussian blur
# mixed the people with the scene behind them: on the hunter pair, the dark coat and the red gown became a pale gray
# ghost of the wall, and the image model drew the coat as a flat mass without parts.
def person_blur(image: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """A blur of the image (float32) inside the people only: each pixel is the mean of the nearby person pixels."""
    hard = (mask >= PERSON_THRESHOLD).astype(np.float32)
    total = cv2.GaussianBlur(image.astype(np.float32) * hard[..., None], (0, 0), sigma)
    weight = cv2.GaussianBlur(hard, (0, 0), sigma)[..., None]
    plain = cv2.GaussianBlur(image, (0, 0), sigma).astype(np.float32)
    return np.where(weight > WEIGHT_FLOOR, total / np.maximum(weight, WEIGHT_FLOOR), plain)


WEIGHT_FLOOR = 1e-3           # under this person weight (the soft rim far from a person), use the plain blur

# Heads strong, garments light (card37, 2026-11-20). The strong blur on the whole person erased the garment detail:
# user, on the card36 hunter pair, "cloth features are blurry/unfinished"; the coat became a plain mass. It also
# erased a doll that the person mask counts as a person (the gnome on the rider's stick). The semi-real look comes
# from the faces, so only the heads get the strong blur, and the rest of a person gets a light one. The heads come
# from MediaPipe face detection (full range model, part of the mediapipe package); without it, the whole person
# gets the strong blur as before.
BODY_SHARE = 0.25             # the blur of the garments, hands, and props in the mask, as a part of the head blur
# The ellipse holds the face and the hair. The hair needs the strong blur too: user (card37), our hair has "one
# color and overdetailed contours", the ChatGPT hair "more simple contour, but interesting shape and more than one
# tone"; a sharp copy of the curls gives the image model a photo outline to trace. The tones come from the text.
HEAD_GROW = 1.8               # the head ellipse around a detected face, as a multiple of the face box
HEAD_LIFT = 0.1               # the head ellipse moves up by this part of the face box (hair, a hat brim)
HEAD_FEATHER = 0.01           # the soft rim of the head mask, as a part of the long side
FACE_CONFIDENCE = 0.5
FACE_ONLY_GROW = 1.0          # the gray face (gray_heads) covers the face box only, so the hair keeps its color
_FACES = None
# The hair (card67 A/B, 2026-11-21). With the same written prompt, a blur of 0.12 on the people turned the hunter
# woman's curls from many thin strokes (the image model traces each photo curl) into a few large wavy locks, but it
# also turned the creature into a human old man and broke the rider's composition. Only the hair gets that strong
# blur: a large ellipse around each face (long hair reaches the shoulders) without the face box itself.
HAIR_SHARE = 5.0              # the blur of the hair, as a multiple of the head blur
HAIR_GROW = 3.2               # the hair ellipse around a detected face, as a multiple of the face box
HAIR_LIFT = 0.0
HAIR_FACE_GROW = 1.1          # the face box that the hair blur leaves out
# The scene outside the people (card40, 2026-11-20): a sharp concrete wall reached the render with its photo cracks
# (hunter pair, card39, 3 of 3). A light blur of the scene removes that texture. An earlier light scene blur gave photo
# bokeh, but then the prompt also named the blur ('a blurred woodland backdrop'); the cleanup now drops those words.
SCENE_SHARE = 0.4             # the blur of the scene, as a part of the head blur (0.15 left the wall cracks, card40)
# A creature head (card68 A/B, 2026-11-21). The creature-desk monster came back as a human old man in 3 of 3 seeds, also
# with strong head words in the prompt. With the same prompt and no blur, it kept its own head (a hanging bulb nose,
# thick lips, jowls, frizz); any blur of 0.006 or more, and edge-preserving filters, lost the nose. The image model
# reads the build of a non-human head from the sharp photo. The face detectors do not find such a head, so the
# writer model gives its box (style_transfer.creature_head_box); inside the box the source stays sharp, and the
# people keep their blur (with the whole source sharp, the woman next to the creature got small photo eyes).
SHARP_FEATHER = 0.01          # the soft rim of a sharp box, as a part of the long side


def head_mask(image: np.ndarray, grow: float = HEAD_GROW, lift: float = HEAD_LIFT) -> np.ndarray | None:
    """The head mask of a BGR image (float32, 0 to 1, soft rim), zeros without faces, or None without MediaPipe.
    With grow 1 and lift 0, the mask holds the face box only."""
    global _FACES
    try:
        import mediapipe
        if _FACES is None:
            _FACES = mediapipe.solutions.face_detection.FaceDetection(model_selection=1,
                                                                      min_detection_confidence=FACE_CONFIDENCE)
        found = _FACES.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).detections or []
    except Exception as error:  # no mediapipe or no model file
        print(f"source soften: no face detection ({error.__class__.__name__}), the whole person gets one blur")
        return None
    h, w = image.shape[:2]
    mask = np.zeros((h, w), np.float32)
    for face in found:
        box = face.location_data.relative_bounding_box
        center = (round((box.xmin + box.width / 2) * w), round((box.ymin + box.height / 2 - lift * box.height) * h))
        axes = (round(box.width * w * grow / 2), round(box.height * h * grow / 2))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
    return cv2.GaussianBlur(mask, (0, 0), max(h, w) * HEAD_FEATHER)


def has_people(path: Path) -> bool:
    """True when the person mask finds people in the image file (then soften blurs them)."""
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return False
    mask = person_mask(image)
    return mask is not None and mask.max() >= PERSON_THRESHOLD


def box_mask(shape: tuple, boxes: tuple) -> np.ndarray:
    """A float32 mask (0 to 1, soft rim) of the ellipses inside the boxes (left, top, right, bottom as parts of the
    width and height)."""
    h, w = shape
    mask = np.zeros((h, w), np.float32)
    for left, top, right, bottom in boxes:
        center = (round((left + right) / 2 * w), round((top + bottom) / 2 * h))
        axes = (max(1, round((right - left) / 2 * w)), max(1, round((bottom - top) / 2 * h)))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
    return cv2.GaussianBlur(mask, (0, 0), max(h, w) * SHARP_FEATHER)


def soften(path: Path, out_dir: Path, fraction: float, gray_heads: bool = False,
           sharp_boxes: tuple = ()) -> Path:
    """A copy of the image in out_dir whose people are blurred, for the image model only; the same path for a
    fraction of 0, a failed read, or an image without people. With gray_heads, the heads lose their color too.
    sharp_boxes (left, top, right, bottom as parts of the width and height) keep the sharp source in an ellipse."""
    if fraction <= 0:
        return path
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return path
    mask = person_mask(image)
    if mask is None or mask.max() < PERSON_THRESHOLD:
        return path
    side = max(image.shape[:2])
    sharp = image.astype(np.float32)
    if SCENE_SHARE > 0:  # a light blur of the scene removes the fine photo texture that the image model copies
        image = cv2.GaussianBlur(image, (0, 0), side * fraction * SCENE_SHARE)
    blurred = person_blur(image, mask, side * fraction)
    heads = head_mask(image)
    if heads is not None:
        heads = heads[..., None]
        blurred = person_blur(image, mask, side * fraction * BODY_SHARE) * (1 - heads) + blurred * heads
        hair = head_mask(image, HAIR_GROW, HAIR_LIFT)
        if HAIR_SHARE > 0 and hair is not None:
            hair = np.clip(hair - head_mask(image, HAIR_FACE_GROW, 0.0), 0, 1)[..., None]
            blurred = blurred * (1 - hair) + person_blur(image, mask, side * fraction * HAIR_SHARE) * hair
        faces = head_mask(image, FACE_ONLY_GROW, 0.0) if gray_heads else None
        if faces is not None:  # card59: a style that prints the face in one or two inks; the hair keeps its color
            faces = faces[..., None]
            gray = cv2.cvtColor(cv2.cvtColor(np.clip(blurred, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY),
                                cv2.COLOR_GRAY2BGR).astype(np.float32)
            blurred = blurred * (1 - faces) + gray * faces
    mask = mask[..., None]
    mixed = image.astype(np.float32) * (1 - mask) + blurred * mask
    if sharp_boxes:
        keep = box_mask(sharp.shape[:2], sharp_boxes)[..., None]
        mixed = mixed * (1 - keep) + sharp * keep
    mixed = mixed.astype(np.uint8)
    ok, data = cv2.imencode(".png", mixed)
    if not ok:
        return path
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{path.stem}.soft{fraction:.4f}{'.hair' if HAIR_SHARE > 0 and heads is not None else ''}{'.gray' if gray_heads and heads is not None else ''}{'.sharp' if sharp_boxes else ''}.png"
    data.tofile(str(out))
    return out
