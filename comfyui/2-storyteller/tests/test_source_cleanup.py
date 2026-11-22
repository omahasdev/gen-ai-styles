"""Source cleanup: a soft dark object at the frame edge is filled; dark scenes and sharp edge parts stay."""
import sys
import unittest
import unittest.mock
from pathlib import Path

import cv2
import numpy as np

TOOL = Path(__file__).resolve().parents[1]
ROOT = TOOL.parents[1]
sys.path.insert(0, str(TOOL))
import source_cleanup  # noqa: E402

H, W = 600, 400
FLOOR = 150  # brightness of the bright floor of the test photo
SEED = 3


def floor_photo(level=FLOOR):
    rng = np.random.default_rng(SEED)
    noise = rng.normal(0, 6, (H, W)).astype(np.float32)
    gray = np.clip(level + noise, 0, 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def add_corner_blob(image, sharp=False):
    blob = np.zeros((H, W), np.float32)
    cv2.circle(blob, (0, H), 70, 1.0, -1)
    if not sharp:
        blob = cv2.GaussianBlur(blob, (0, 0), 12)
    return (image * (1 - 0.9 * blob[..., None])).astype(np.uint8)


class SourceCleanupTest(unittest.TestCase):
    def test_soft_dark_corner_object_is_found(self):
        mask = source_cleanup.edge_fault_mask(add_corner_blob(floor_photo()))
        self.assertIsNotNone(mask)
        self.assertGreater(mask[H - 5, 5], 0)  # the corner
        self.assertEqual(mask[H // 2, W // 2], 0)  # the middle of the frame stays

    def test_clean_photo_has_no_fault(self):
        self.assertIsNone(source_cleanup.edge_fault_mask(floor_photo()))

    def test_low_key_scene_is_left_alone(self):
        dark_scene = add_corner_blob(floor_photo(level=source_cleanup.MIN_SCENE_MEDIAN // 2))
        self.assertIsNone(source_cleanup.edge_fault_mask(dark_scene))

    def test_sharp_edge_part_is_left_alone(self):
        self.assertIsNone(source_cleanup.edge_fault_mask(add_corner_blob(floor_photo(), sharp=True)))

    def test_soften_strength_follows_the_distance_from_a_photograph(self):
        self.assertEqual(source_cleanup.soften_fraction(0.15), 0.0)  # a photographic style keeps the sharp source
        self.assertEqual(source_cleanup.soften_fraction(1.0), source_cleanup.SOFTEN_MAX)
        self.assertLess(source_cleanup.soften_fraction(0.8), source_cleanup.soften_fraction(0.95))

    def test_soften_blurs_a_copy_and_keeps_the_source(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src.png"
            image = floor_photo()
            cv2.rectangle(image, (100, 100), (300, 500), (0, 0, 0), -1)  # a hard edge
            cv2.imencode(".png", image)[1].tofile(str(src))
            self.assertEqual(source_cleanup.soften(src, Path(tmp) / "soft", 0.0), src)
            # Only the people are blurred: a person mask over the dark block blurs its edge.
            person = np.zeros((H, W), np.float32)
            person[:, :250] = 1.0
            with unittest.mock.patch.object(source_cleanup, "person_mask", return_value=person), \
                    unittest.mock.patch.object(source_cleanup, "head_mask", return_value=None):
                out = source_cleanup.soften(src, Path(tmp) / "soft", source_cleanup.SOFTEN_MAX)
            self.assertNotEqual(out, src)
            edge = lambda img, x: float(np.abs(np.diff(img[300, x - 10:x + 10, 0].astype(int))).max())
            self.assertLess(edge(cv2.imread(str(out)), 100), edge(cv2.imread(str(src)), 100) / 2)  # in the mask
            scene, people = edge(cv2.imread(str(out)), 300), edge(cv2.imread(str(out)), 100)
            self.assertLessEqual(scene, edge(cv2.imread(str(src)), 300))  # the scene gets a light blur at most
            self.assertGreater(scene, people)  # and stays sharper than the people
            self.assertTrue(np.array_equal(cv2.imread(str(src)), image))  # the source file is unchanged
            with unittest.mock.patch.object(source_cleanup, "person_mask", return_value=None):
                self.assertEqual(source_cleanup.soften(src, Path(tmp) / "soft", source_cleanup.SOFTEN_MAX), src)


    def test_soften_blurs_the_heads_more_than_the_garments(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src.png"
            image = floor_photo()
            for y in (150, 450):  # the same hard edge in the head part and in the garment part
                cv2.rectangle(image, (100, y - 50), (200, y + 50), (0, 0, 0), -1)
            cv2.imencode(".png", image)[1].tofile(str(src))
            person = np.ones((H, W), np.float32)
            heads = np.zeros((H, W), np.float32)
            heads[:H // 2] = 1.0
            with unittest.mock.patch.object(source_cleanup, "person_mask", return_value=person), \
                    unittest.mock.patch.object(source_cleanup, "head_mask", return_value=heads):
                out = cv2.imread(str(source_cleanup.soften(src, Path(tmp) / "soft", source_cleanup.SOFTEN_MAX)))
            edge = lambda img, y: float(np.abs(np.diff(img[y, 90:110, 0].astype(int))).max())
            self.assertLess(edge(out, 150), edge(out, 450))  # the head edge is softer than the garment edge
            self.assertLess(edge(out, 450), edge(image, 450))  # the garments are blurred too, a little

    def test_a_sharp_box_keeps_the_source(self):
        """A creature head box keeps the sharp source; the rest of the person is blurred."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src.png"
            image = floor_photo()
            for y in (150, 450):
                cv2.rectangle(image, (100, y - 50), (200, y + 50), (0, 0, 0), -1)
            cv2.imencode(".png", image)[1].tofile(str(src))
            person = np.ones((H, W), np.float32)
            box = (0.0, 0.0, 1.0, 0.5)  # the upper half
            with unittest.mock.patch.object(source_cleanup, "person_mask", return_value=person), \
                    unittest.mock.patch.object(source_cleanup, "head_mask", return_value=None):
                out_path = source_cleanup.soften(src, Path(tmp) / "soft", source_cleanup.SOFTEN_MAX,
                                                 sharp_boxes=(box,))
            out = cv2.imread(str(out_path))
            edge = lambda img, y: float(np.abs(np.diff(img[y, 90:110, 0].astype(int))).max())
            self.assertGreater(edge(out, 150), edge(image, 150) * 0.9)  # the box center is the sharp source
            self.assertLess(edge(out, 450), edge(image, 450) / 2)  # the rest of the person is blurred
            self.assertIn(".sharp", out_path.name)

    def test_person_blur_keeps_the_colors_of_the_person(self):
        side, dark, light = 200, 20, 230
        image = np.full((side, side, 3), light, np.uint8)
        mask = np.zeros((side, side), np.float32)
        image[50:150, 80:120] = dark
        mask[50:150, 80:120] = 1.0
        sigma = side * 0.025
        own = source_cleanup.person_blur(image, mask, sigma)
        plain = cv2.GaussianBlur(image, (0, 0), sigma).astype(np.float32)
        self.assertLess(abs(own[100, 82, 0] - dark), 2)  # the edge of the person keeps its dark color
        self.assertGreater(plain[100, 82, 0], dark + 40)  # a plain blur mixes in the light scene

    def test_people_crop_holds_the_people_with_a_margin(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src.png"
            cv2.imencode(".png", floor_photo())[1].tofile(str(src))
            person = np.zeros((H, W), np.float32)
            person[200:400, 100:200] = 1.0
            with unittest.mock.patch.object(source_cleanup, "person_mask", return_value=person):
                out = source_cleanup.people_crop(src, Path(tmp) / "crop")
            pad = round(max(H, W) * source_cleanup.PEOPLE_CROP_PAD)
            crop_h, crop_w = 200 + 2 * pad, 100 + 2 * pad
            scale = source_cleanup.PEOPLE_CROP_SIDE / crop_h  # a small crop is enlarged
            self.assertEqual(cv2.imread(str(out)).shape[:2], (round(crop_h * scale), round(crop_w * scale)))
            with unittest.mock.patch.object(source_cleanup, "person_mask", return_value=None):
                self.assertIsNone(source_cleanup.people_crop(src, Path(tmp) / "crop"))


if __name__ == "__main__":
    unittest.main()
