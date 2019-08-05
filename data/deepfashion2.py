import torchvision
import torch
import torch.distributed as dist
import sys
import random
import numpy as np
from PIL import Image

def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= 10:
        return True
    return False


class DeepFashion2Dataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
            self, ann_file, root, transforms=None, train=False
    ):
        super(DeepFashion2Dataset, self).__init__(root, ann_file)
        self.ids = sorted(self.ids)

        self.categories = {cat['id']: cat['name'] for cat in self.coco.cats.values()}

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self.idx_to_id_map = {v: k for k, v in enumerate(self.ids)}

        self._transforms = transforms
        self.street_inds = self._getTypeInds('user')
        self.shop_inds = self._getTypeInds('shop')

        self.match_map_shop = {}
        self.match_map_street = {}

        print("Computing Street Match Descriptors map")
        for i in self.street_inds:
            e = self.coco.imgs[i]
            for x in e['match_desc']:
                if x == '0':
                    continue
                hashable_key = x + '_' + str(e['match_desc'].get(x))
                inds = self.match_map_street.get(hashable_key)
                if inds is None:
                    self.match_map_street.update({hashable_key: [i]})
                else:
                    inds.append(i)
                    self.match_map_street.update({hashable_key: inds})

        print("Computing Shop Match Descriptors map")
        for i in self.shop_inds:
            e = self.coco.imgs[i]
            for x in e['match_desc']:
                if x == '0':
                    continue
                hashable_key = x + '_' + str(e['match_desc'].get(x))
                inds = self.match_map_shop.get(hashable_key)
                if inds is None:
                    self.match_map_shop.update({hashable_key: [i]})
                else:
                    inds.append(i)
                    self.match_map_shop.update({hashable_key: inds})

        print("Filtering  no matches")
        street_match_keys = self.match_map_street.keys()
        shop_match_keys = self.match_map_shop.keys()

        if train:
            to_del = []
            for x in self.match_map_street:
                if x not in shop_match_keys:  # or len(self.match_map_street[x]) < 2:
                    to_del.append(x)

            for x in to_del:
                del self.match_map_street[x]

        else:
            to_del = []
            for x in self.match_map_street:
                if x not in shop_match_keys or len(self.match_map_street[x]) < 2:
                    to_del.append(x)

            for x in to_del:
                del self.match_map_street[x]

        print("Total images after filtering:" + str(len(self.match_map_street)))

    def __getitem__(self, x):
        i, tag, index = x
        if tag == "shop":
            idx = random.choice(self.match_map_shop[i])
        else:
            index2 = int(len(self.match_map_street[i]) * index)
            idx = self.match_map_street[i][index2]

        img, anno = super(DeepFashion2Dataset, self).__getitem__(self.idx_to_id_map[idx])

        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        pair_id = int(i.split('_')[-1])
        style = int(i.split('_')[0])

        box = [obj["bbox"] for obj in anno if obj['area'] != 0 and obj['pair_id'] == pair_id and obj['style'] == style]
        box = torch.as_tensor(box).reshape(-1, 4)  # guard against no boxes
        box = box[0]
        box[2] = box[2] + box[0]
        box[3] = box[3] + box[1]

        img = img.crop((int(box[0]), int(box[1]), int(box[2]), int(box[3])))

        target = torch.LongTensor([pair_id, style])

        if self._transforms is not None:
            img = self._transforms(img)

        return img, target

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data

    def _getTypeInds(self, type_s):
        inds = []
        N = len(self.coco.imgs)
        for i in self.ids:
            if self.coco.imgs[i]['source'] == type_s:
                inds.append(i)

        return inds

    def __len__(self):
        return len(self.match_map_street)
