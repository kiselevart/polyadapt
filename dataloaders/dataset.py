"""VideoDataset — ported from vnn/dataloaders/dataset.py.

Optical flow computation is omitted (single-stream RGB only for polyadapt).
"""

import os
import re
from collections import defaultdict

import cv2
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tqdm import tqdm

from mypath import Path


def _extract_frames(src_path, target_dir, clip_len, resize_height, resize_width):
    """Extract equally-spaced frames from a video file into target_dir."""
    if os.path.isdir(target_dir) and any(f.endswith(".jpg") for f in os.listdir(target_dir)):
        return

    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        cap.release()
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count < clip_len:
        cap.release()
        return

    freq = 4
    while freq > 1 and frame_count // freq < clip_len:
        freq -= 1

    os.makedirs(target_dir, exist_ok=True)
    count = i = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if count % freq == 0:
            h, w = frame.shape[:2]
            if h != resize_height or w != resize_width:
                frame = cv2.resize(frame, (resize_width, resize_height))
            cv2.imwrite(os.path.join(target_dir, f"{i:05d}.jpg"), frame)
            i += 1
        count += 1
    cap.release()

    if i == 0:
        try:
            os.rmdir(target_dir)
        except OSError:
            pass


class VideoDataset(Dataset):
    """Reads pre-extracted JPEG frame directories.

    Expected on-disk layout after preprocessing::

        <output_dir>/{train,val,test}/<class>/<video_dir>/<frame>.jpg

    Args:
        dataset:    Dataset name (ucf101, hmdb51, ssv2, …).
        split:      "train", "val", or "test".
        clip_len:   Number of frames per clip.
        preprocess: Re-extract frames even if they already exist.
        augment:    Random crop + flip + colour jitter when True; centre crop otherwise.
        ucf_split:  Official split number (1–3) for UCF101 / HMDB51.
    """

    def __init__(self, dataset="ucf101", split="train", clip_len=16,
                 preprocess=False, augment=True, ucf_split=1):
        self.root_dir, base_output_dir = Path.db_dir(dataset)
        if dataset.lower() in ("ucf101", "hmdb51"):
            self.output_dir = os.path.join(base_output_dir, f"split{ucf_split}")
        else:
            self.output_dir = base_output_dir
        self.ucf_split = ucf_split
        self.clip_len  = clip_len
        self.split     = split
        self.augment   = augment

        self.resize_height = 128
        self.resize_width  = 171
        self.crop_size     = 112
        self.mean = np.array([90.0, 98.0, 102.0], dtype=np.float32)  # BGR

        self.pre_split = (
            os.path.isdir(os.path.join(self.root_dir, "train")) and
            os.path.isdir(os.path.join(self.root_dir, "test"))
        )

        if not os.path.exists(self.root_dir):
            raise RuntimeError(
                f"Dataset root not found: {self.root_dir}\n"
                f"Set VNN_DATA_ROOT to the directory that contains the '{dataset}' folder."
            )

        if (not self._check_preprocess()) or preprocess:
            print(f"==> Preprocessing {dataset} — extracting frames …")
            self._preprocess(dataset)
            self._invalidate_cache(split)

        folder = os.path.join(self.output_dir, split)
        cache  = os.path.join(self.output_dir, f"filelist_{split}.pkl")

        if os.path.exists(cache):
            import pickle
            with open(cache, "rb") as f:
                self.fnames, labels = pickle.load(f)
            print(f"Number of {split} videos: {len(self.fnames):d} (cache)")
        else:
            import pickle
            self.fnames, labels, skipped = [], [], 0
            for label in sorted(os.listdir(folder)):
                cls_dir = os.path.join(folder, label)
                if not os.path.isdir(cls_dir):
                    continue
                for fname in os.listdir(cls_dir):
                    fpath = os.path.join(cls_dir, fname)
                    if not os.path.isdir(fpath):
                        continue
                    n = sum(1 for f in os.listdir(fpath) if f.endswith(".jpg"))
                    if n < 2:
                        skipped += 1
                        continue
                    self.fnames.append(fpath)
                    labels.append(label)
            if skipped:
                print(f"  Skipped {skipped} videos with < 2 frames.")
            print(f"Number of {split} videos: {len(self.fnames):d}")
            with open(cache, "wb") as f:
                pickle.dump((self.fnames, labels), f)

        self.label2index = {
            lbl: idx for idx, lbl in enumerate(sorted(os.listdir(folder)))
        }
        self.label_array = np.array([self.label2index[l] for l in labels], dtype=int)

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        buffer = self._load_frames(self.fnames[index])
        if self.augment:
            buffer = self._crop(buffer, self.clip_len, self.crop_size)
        else:
            buffer = self._center_crop(buffer, self.clip_len, self.crop_size)
        buffer = self._ensure_clip_len(buffer, self.clip_len)
        if self.augment:
            buffer = self._randomflip(buffer)
            buffer = self._color_jitter(buffer)
        buffer = self._normalize(buffer)
        buffer = buffer.transpose((3, 0, 1, 2))  # [C, T, H, W]
        return torch.from_numpy(buffer), torch.from_numpy(np.array(self.label_array[index]))

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _check_preprocess(self):
        train_dir = os.path.join(self.output_dir, "train")
        if not os.path.isdir(train_dir):
            return False
        for cls in os.listdir(train_dir):
            cls_dir = os.path.join(train_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for vid in os.listdir(cls_dir):
                vid_dir = os.path.join(cls_dir, vid)
                if not os.path.isdir(vid_dir):
                    continue
                frames = sorted(f for f in os.listdir(vid_dir) if f.endswith(".jpg"))
                if not frames:
                    return False
                img = cv2.imread(os.path.join(vid_dir, frames[0]))
                if img is None or img.shape[0] != 128 or img.shape[1] != 171:
                    return False
                return True  # one sample is enough
        return False

    def _invalidate_cache(self, split):
        cache = os.path.join(self.output_dir, f"filelist_{split}.pkl")
        if os.path.exists(cache):
            os.remove(cache)

    def _is_video(self, path):
        return os.path.isfile(path) and path.lower().endswith(
            (".avi", ".mp4", ".mkv", ".mpg", ".mpeg", ".mov", ".webm")
        )

    def _collect_videos(self, class_dir):
        entries = []
        for name in sorted(os.listdir(class_dir)):
            full = os.path.join(class_dir, name)
            if self._is_video(full):
                entries.append(name)
            elif os.path.isdir(full):
                for nested in sorted(os.listdir(full)):
                    if self._is_video(os.path.join(full, nested)):
                        entries.append(os.path.join(name, nested))
        return entries

    def _process_video(self, video, action_name, save_dir):
        src = os.path.join(self.root_dir, action_name, video)
        stem = os.path.splitext(os.path.basename(video))[0]
        _extract_frames(src, os.path.join(save_dir, stem),
                        self.clip_len, self.resize_height, self.resize_width)

    def _resize_frames_to_dir(self, src_dir, dst_dir):
        frames = sorted(f for f in os.listdir(src_dir) if f.endswith(".jpg"))
        if not frames:
            return
        os.makedirs(dst_dir, exist_ok=True)
        for i, fname in enumerate(frames):
            img = cv2.imread(os.path.join(src_dir, fname))
            if img is None:
                continue
            if img.shape[0] != self.resize_height or img.shape[1] != self.resize_width:
                img = cv2.resize(img, (self.resize_width, self.resize_height))
            cv2.imwrite(os.path.join(dst_dir, f"{i:05d}.jpg"), img)

    def _carve_val(self, train_entries, val_ratio=0.2, seed=42):
        rng = np.random.RandomState(seed)
        class_groups = defaultdict(lambda: defaultdict(list))
        for cls, vid in train_entries:
            m = re.match(r"v_\w+_(g\d+)_c\d+\.avi", vid)
            group = m.group(1) if m else "g00"
            class_groups[cls][group].append(vid)
        final_train, final_val = [], []
        for cls in sorted(class_groups):
            groups = sorted(class_groups[cls])
            rng.shuffle(groups)
            n_val = max(1, int(len(groups) * val_ratio))
            val_groups = set(groups[:n_val])
            for g in groups:
                for v in class_groups[cls][g]:
                    (final_val if g in val_groups else final_train).append((cls, v))
        return final_train, final_val

    def _preprocess(self, dataset):
        for s in ("train", "val", "test"):
            os.makedirs(os.path.join(self.output_dir, s), exist_ok=True)

        ucf_train = os.path.join(self.root_dir, "ucfTrainTestlist",
                                 f"trainlist0{self.ucf_split}.txt")
        ucf_test  = os.path.join(self.root_dir, "ucfTrainTestlist",
                                 f"testlist0{self.ucf_split}.txt")
        hmdb_dir  = os.path.join(self.root_dir, "testTrainMulti_7030_splits")

        if os.path.exists(ucf_train) and os.path.exists(ucf_test):
            self._preprocess_ucf101(ucf_train, ucf_test)
        elif os.path.isdir(hmdb_dir):
            self._preprocess_hmdb51(hmdb_dir)
        elif self.pre_split:
            self._preprocess_pre_split()
        elif self._is_preextracted():
            self._preprocess_from_frames()
        else:
            self._preprocess_flat()
        print("Preprocessing finished.")

    def _preprocess_ucf101(self, train_list, test_list):
        def parse(path):
            entries = []
            with open(path) as f:
                for line in f:
                    parts = line.strip().split("/")
                    if len(parts) == 2:
                        cls = parts[0]
                        vid = parts[1].split()[0]
                        entries.append((cls, vid))
            return entries

        train_e = parse(train_list)
        test_e  = parse(test_list)
        final_train, final_val = self._carve_val(train_e)

        for split, entries in [("train", final_train), ("val", final_val), ("test", test_e)]:
            print(f"  {split}: {len(entries)} videos")
            for cls, vid in tqdm(entries, desc=f"Processing {split}"):
                save_dir = os.path.join(self.output_dir, split, cls)
                os.makedirs(save_dir, exist_ok=True)
                self._process_video(vid, cls, save_dir)

    def _preprocess_hmdb51(self, splits_dir):
        train_e, test_e = [], []
        for cls in sorted(os.listdir(self.root_dir)):
            cls_dir = os.path.join(self.root_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            split_file = os.path.join(splits_dir, f"{cls}_test_split{self.ucf_split}.txt")
            if not os.path.exists(split_file):
                continue
            with open(split_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 2:
                        continue
                    vid, marker = parts[0], int(parts[1])
                    if marker == 1:
                        train_e.append((cls, vid))
                    elif marker == 2:
                        test_e.append((cls, vid))

        rng = np.random.RandomState(42)
        by_class = defaultdict(list)
        for cls, vid in train_e:
            by_class[cls].append(vid)
        final_train, final_val = [], []
        for cls in sorted(by_class):
            vids = by_class[cls]
            rng.shuffle(vids)
            n_val = max(1, int(len(vids) * 0.15))
            final_val.extend((cls, v) for v in vids[:n_val])
            final_train.extend((cls, v) for v in vids[n_val:])

        for split, entries in [("train", final_train), ("val", final_val), ("test", test_e)]:
            print(f"  {split}: {len(entries)} videos")
            for cls, vid in tqdm(entries, desc=f"Processing {split}"):
                save_dir = os.path.join(self.output_dir, split, cls)
                os.makedirs(save_dir, exist_ok=True)
                vid_stem = os.path.splitext(vid)[0]
                src_frames = os.path.join(self.root_dir, cls, vid_stem)
                if os.path.isdir(src_frames):
                    dst = os.path.join(save_dir, vid_stem)
                    if not os.path.exists(dst):
                        self._resize_frames_to_dir(src_frames, dst)
                else:
                    self._process_video(vid, cls, save_dir)

    def _preprocess_pre_split(self):
        has_val = os.path.isdir(os.path.join(self.root_dir, "val"))
        if has_val:
            for split in ("train", "val", "test"):
                src = os.path.join(self.root_dir, split)
                if not os.path.isdir(src):
                    continue
                for cls in sorted(os.listdir(src)):
                    cls_dir = os.path.join(src, cls)
                    if not os.path.isdir(cls_dir):
                        continue
                    save_dir = os.path.join(self.output_dir, split, cls)
                    os.makedirs(save_dir, exist_ok=True)
                    for vid in self._collect_videos(cls_dir):
                        self._process_video(vid, os.path.join(split, cls), save_dir)
        else:
            train_src = os.path.join(self.root_dir, "train")
            entries = []
            for cls in sorted(os.listdir(train_src)):
                cls_dir = os.path.join(train_src, cls)
                if not os.path.isdir(cls_dir):
                    continue
                for vid in self._collect_videos(cls_dir):
                    entries.append((cls, vid))
            final_train, final_val = self._carve_val(entries)
            for split, es in [("train", final_train), ("val", final_val)]:
                for cls, vid in tqdm(es, desc=f"Processing {split}"):
                    save_dir = os.path.join(self.output_dir, split, cls)
                    os.makedirs(save_dir, exist_ok=True)
                    self._process_video(vid, os.path.join("train", cls), save_dir)
            for cls in sorted(os.listdir(os.path.join(self.root_dir, "test"))):
                cls_dir = os.path.join(self.root_dir, "test", cls)
                if not os.path.isdir(cls_dir):
                    continue
                save_dir = os.path.join(self.output_dir, "test", cls)
                os.makedirs(save_dir, exist_ok=True)
                for vid in self._collect_videos(cls_dir):
                    self._process_video(vid, os.path.join("test", cls), save_dir)

    def _is_preextracted(self):
        for cls in sorted(os.listdir(self.root_dir)):
            cls_dir = os.path.join(self.root_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for item in sorted(os.listdir(cls_dir)):
                item_path = os.path.join(cls_dir, item)
                if os.path.isdir(item_path):
                    if any(f.endswith(".jpg") for f in os.listdir(item_path)):
                        return True
                elif self._is_video(item_path):
                    return False
            break
        return False

    def _preprocess_from_frames(self):
        rng = np.random.RandomState(42 + self.ucf_split)
        for cls in tqdm(sorted(os.listdir(self.root_dir)), desc="Classes"):
            cls_dir = os.path.join(self.root_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            video_dirs = sorted(
                d for d in os.listdir(cls_dir)
                if os.path.isdir(os.path.join(cls_dir, d))
            )
            if not video_dirs:
                continue
            video_dirs = list(rng.permutation(video_dirs))
            n = len(video_dirs)
            n_test = max(1, int(n * 0.2))
            n_val  = max(1, int(n * 0.1))
            for split, dirs in [
                ("test",  video_dirs[:n_test]),
                ("val",   video_dirs[n_test:n_test + n_val]),
                ("train", video_dirs[n_test + n_val:]),
            ]:
                dst_cls = os.path.join(self.output_dir, split, cls)
                os.makedirs(dst_cls, exist_ok=True)
                for vd in dirs:
                    src = os.path.join(cls_dir, vd)
                    dst = os.path.join(dst_cls, vd)
                    if not os.path.exists(dst):
                        self._resize_frames_to_dir(src, dst)

    def _preprocess_flat(self):
        for cls in os.listdir(self.root_dir):
            cls_dir = os.path.join(self.root_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            vids = self._collect_videos(cls_dir)
            if len(vids) < 3:
                continue
            tv, test = train_test_split(vids, test_size=0.2, random_state=42)
            train, val = train_test_split(tv, test_size=0.2, random_state=42)
            for split, subset in [("train", train), ("val", val), ("test", test)]:
                save_dir = os.path.join(self.output_dir, split, cls)
                os.makedirs(save_dir, exist_ok=True)
                for vid in subset:
                    self._process_video(vid, cls, save_dir)

    # ------------------------------------------------------------------
    # Frame loading + augmentation
    # ------------------------------------------------------------------

    def _load_frames(self, file_dir):
        frames = sorted(
            os.path.join(file_dir, f) for f in os.listdir(file_dir) if f.endswith(".jpg")
        )
        if not frames:
            return np.zeros((0, self.resize_height, self.resize_width, 3), np.float32)
        buf = np.empty((len(frames), self.resize_height, self.resize_width, 3), np.float32)
        for i, path in enumerate(frames):
            img = cv2.imread(path)
            if img is None:
                img = np.zeros((self.resize_height, self.resize_width, 3), np.float32)
            buf[i] = img.astype(np.float32)
        return buf

    def _ensure_clip_len(self, buf, clip_len):
        T = buf.shape[0]
        if T == clip_len:
            return buf
        if T == 0:
            return np.zeros((clip_len, self.crop_size, self.crop_size, 3), np.float32)
        if T > clip_len:
            return buf[:clip_len]
        pad = np.repeat(buf[-1:], clip_len - T, axis=0)
        return np.concatenate([buf, pad], axis=0)

    def _crop(self, buf, clip_len, crop_size):
        max_t = max(0, buf.shape[0] - clip_len)
        max_h = max(1, buf.shape[1] - crop_size)
        max_w = max(1, buf.shape[2] - crop_size)
        t = np.random.randint(max_t + 1)
        h = np.random.randint(max_h)
        w = np.random.randint(max_w)
        return buf[t:t + clip_len, h:h + crop_size, w:w + crop_size]

    def _center_crop(self, buf, clip_len, crop_size):
        t = max(0, (buf.shape[0] - clip_len) // 2)
        h = max(0, (buf.shape[1] - crop_size) // 2)
        w = max(0, (buf.shape[2] - crop_size) // 2)
        return buf[t:t + clip_len, h:h + crop_size, w:w + crop_size]

    def _randomflip(self, buf):
        if np.random.random() < 0.5:
            for i in range(len(buf)):
                buf[i] = cv2.flip(buf[i], flipCode=1)
        return buf

    def _color_jitter(self, buf, brightness=0.3, contrast=0.3):
        alpha = 1.0 + np.random.uniform(-contrast, contrast)
        beta  = np.random.uniform(-brightness, brightness) * 255.0
        return np.clip(alpha * buf + beta, 0.0, 255.0).astype(buf.dtype)

    def _normalize(self, buf):
        for i in range(len(buf)):
            buf[i] -= self.mean
        return buf
