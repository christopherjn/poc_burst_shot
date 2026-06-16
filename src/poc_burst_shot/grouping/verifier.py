import uuid
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image

from poc_burst_shot.models.burst_group import BurstGroup
from poc_burst_shot.models.image_record import ImageRecord

MIN_MATCHES = 10
INLIER_RATIO_THRESHOLD = 0.4


def _pil_to_gray(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("L"))


def _same_scene(img_a: Image.Image, img_b: Image.Image) -> bool:
    orb = cv2.ORB_create(nfeatures=500)

    kp_a, des_a = orb.detectAndCompute(_pil_to_gray(img_a), None)
    kp_b, des_b = orb.detectAndCompute(_pil_to_gray(img_b), None)

    if des_a is None or des_b is None or len(kp_a) < MIN_MATCHES or len(kp_b) < MIN_MATCHES:
        return False

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des_a, des_b)

    if len(matches) < MIN_MATCHES:
        return False

    # list comprehension required — np.float32 does not accept a bare generator
    src_pts = np.float32([kp_a[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_b[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if mask is None:
        return False

    inlier_ratio = mask.sum() / len(matches)
    return inlier_ratio > INLIER_RATIO_THRESHOLD


def verify_and_split(group: BurstGroup) -> list[BurstGroup]:
    """Returns one or more BurstGroups — splits if false merges are detected."""
    members = group.members

    if len(members) <= 1:
        return [group]

    # each member starts as its own component
    parent = list(range(len(members)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        parent[find(x)] = find(y)

    images = [Image.open(m.path).convert("RGB").resize((224, 224)) for m in members]

    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            if _same_scene(images[i], images[j]):
                union(i, j)

    component_map: dict[int, list[ImageRecord]] = defaultdict(list)
    for i, member in enumerate(members):
        component_map[find(i)].append(member)

    if len(component_map) == 1:
        return [group]

    return [
        BurstGroup(
            group_id=str(uuid.uuid4()),
            members=component_members,
            grouping_method="visual",
        )
        for component_members in component_map.values()
    ]
