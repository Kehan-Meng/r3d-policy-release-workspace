"""
AffogatoTextHeatmapDataset — 修改版 (change)
任务：读取affogato的数据，返回dataset所需要的sample

与原始 affogato.py 的区别：
1. _build_index: 每个 query 生成一个 item（文本为中心的样本）
2. __getitem__: 每个样本返回单个 text query 和对应 heatmap
   - gt_masks: [1, N] float32 (soft heatmap, 0~1)
   - text: List[str], 长度为 1（[单个文本]）
3. split: 从预先生成的 json 读取点云 folder 名称，再展开 query
4. __len__: = 当前 split json 内所有 folder 的 query 总数
5. 经 DataLoader + collate_fn batch 后：
   - coords:    [B, N, 3]
   - features:  [B, N, 3]
   - gt_masks:  [B, 1, N]
   - text:      List[List[str]]  # 长度 B，每个含 1 个 str
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from torch.utils.data import Dataset

# 旧的逐样本 dprint/tprint 调试已停用，避免 DataLoader 多 worker 并发写 debug.txt。
# from utils.debug_utils import dprint, tprint


class AffogatoTextHeatmapDataset(Dataset):
    """
    Affogato text-to-3D-heatmap dataset.

    Expected directory:

        root/
        ├── sample_id/
        │   ├── queries.json
        │   └── xyzc.npy

    xyzc.npy:
        shape [N, 8]
        xyzc[:, :3] -> xyz
        xyzc[:, 3:] -> Q heatmaps, one per text query

    queries.json example:
        [
            {
                "class_name": "Rock",
                "queries": [
                    "Point to the part you would use to hammer.",
                    ...
                ]
            }
        ]

    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        split_file: str = None,
        transform=None,
        normalize_heatmap: bool = False,
        default_rgb: float = 100,
    ):
        self.root = Path(root)
        self.split = split
        self.split_file = Path(split_file) if split_file else None
        self.transform = transform
        self.normalize_heatmap = normalize_heatmap
        self.default_rgb = default_rgb

        if not self.root.exists():
            raise FileNotFoundError(f"Affogato root does not exist: {self.root}")

        folders = self._load_split_folders() #根据 split json，拿到当前 train/val 应该使用的点云文件夹列表。

        # 旧调试：输出 split 中前几个样本目录，现已停用。
        # for i, folder in enumerate(folders[:6]):
        #     dprint(f"file_id[{i}]", folder, 1)

        if len(folders) == 0:
            raise RuntimeError(
                f"No Affogato samples found for split={split}. "
                "Check split_file and sample folders."
            )

        self.folders = folders  #文件夹相对目录
        self.items = self._build_index(folders)

        if len(self.items) == 0:
            raise RuntimeError(
                f"No valid query-heatmap pairs found under {self.root}, split={split}."
            )

        print(
            f"[AffogatoTextHeatmapDataset] root={self.root}, split={split}, "
            f"split_file={self.split_file}, point_clouds={len(folders)}, "
            f"samples={len(self.items)}"
        )

    def _load_split_folders(self):
        
        with open(self.split_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        folders = []
        for name in data:
            folder = self.root / str(name)  #绝对目录
            if not folder.is_dir():
                raise FileNotFoundError(f"Folder not found: {folder}")
            if not (folder / "queries.json").is_file():
                raise FileNotFoundError(f"Missing queries.json: {folder}")
            if not (folder / "xyzc.npy").is_file():
                raise FileNotFoundError(f"Missing xyzc.npy: {folder}")
            folders.append(folder)

        return folders#绝对目录

    def _load_queries(self, queries_path: Path):
        with open(queries_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            if len(data) == 0:
                return "", []
            data = data[0]

        class_name = data.get("class_name", "")
        queries = data.get("queries", [])

        return class_name, queries

    def _build_index(self, folders: List[Path]):
        """
        每个 query 一个 item — 文本为中心的样本。
        初始化阶段扫描所有合法 folder，把每个 folder 里的每条文本 query 和 xyzc.npy 里的对应 heatmap 通道配成一个独立训练样本，最终得到 self.items
        
        """
        items = []

        for folder in folders:
            xyzc_path = folder / "xyzc.npy"
            queries_path = folder / "queries.json"

            class_name, queries = self._load_queries(queries_path)


            # 旧调试：输出类别名和全部文本查询，现已停用。
            # dprint("class_name", class_name)
            # dprint("queries", queries)

            if len(queries) == 0:
                    continue

            try:
                xyzc_shape = np.load(xyzc_path, mmap_mode="r").shape
            except Exception as e:
                print(f"[skip] failed to read {xyzc_path}: {e}")
                continue

            if len(xyzc_shape) != 2 or xyzc_shape[1] <= 3:
                print(f"[skip] bad xyzc shape: {xyzc_path}, shape={xyzc_shape}")
                continue
        

            # Match text queries with heatmap channels. Do not assume every
            # sample always has exactly 5 valid query/heatmap pairs.
            num_heatmaps = xyzc_shape[1] - 3
            num_pairs = min(len(queries), num_heatmaps)

            for q in range(num_pairs):
                items.append(
                    {
                        "folder": folder,
                        "query_index": q,
                        "text": queries[q],
                        "class_name": class_name,
                    }
                )

        return items

    def _normalize_heatmap(self, heatmap: np.ndarray):
        heatmap = np.asarray(heatmap, dtype=np.float32)

        if not self.normalize_heatmap:
            return heatmap.astype(np.float32)

        h_min = float(heatmap.min())
        h_max = float(heatmap.max())

        if h_max > h_min:
            heatmap = (heatmap - h_min) / (h_max - h_min)
        else:
            heatmap = np.zeros_like(heatmap, dtype=np.float32)

        return heatmap.astype(np.float32)

    def __len__(self):
        return len(self.items)

    def _apply_transform(self, sample: Dict[str, Any]):
        """
        Point-SAM transforms 配合 HuggingFace Dataset.set_transform 使用，
        输入形态通常是 batch dict: key -> list[value]。

        这里把单样本包成 batch，再解出来。
        """
        if self.transform is None:
            return sample

        batch = {k: [v] for k, v in sample.items()}
        batch = self.transform(batch)

        out = {}
        for k, v in batch.items():
            if isinstance(v, list):
                out[k] = v[0]
            else:
                out[k] = v[0] if hasattr(v, "__getitem__") else v

        return out

    def __getitem__(self, index: int):
        """
        Returns:
            coords:    [N, 3] float32
            features:  [N, 3] float32
            gt_masks:  [1, N] float32 — single soft heatmap in [0, 1]
            text:      List[str], length 1（["单个文本描述"]）
            class_name: str
            sample_id: str, unique per point-cloud/query pair
            point_cloud_id: str, folder id shared by all queries from one point cloud
            query_index: int
        """
        item = self.items[index]
        folder = item["folder"]
        q_idx = item["query_index"]

        xyzc = np.load(folder / "xyzc.npy").astype(np.float32)

        coords = xyzc[:, :3].astype(np.float32)

        features = np.full(
            (coords.shape[0], 3),
            self.default_rgb,
            dtype=np.float32,
        )

        # 加载单个 heatmap channel
        heatmap_raw = xyzc[:, 3 + q_idx]  # [N]

        # 直接使用 _build_index 时已经存好的 text
        text = [item["text"]]  # List[str], 长度 1

        # Normalize
        gt_masks = self._normalize_heatmap(heatmap_raw)  # [N]
        gt_masks = gt_masks[np.newaxis, :]  # [1, N] float32

        sample = {
            "coords": coords,              # [N, 3]
            "features": features,          # [N, 3]
            "gt_masks": gt_masks,          # [1, N], soft heatmap
            "text": text,                  # List[str], length 1
            "class_name": item["class_name"],
            "sample_id": f"{folder.name}::q{q_idx}",
            "point_cloud_id": folder.name,
            "query_index": q_idx,
        }

        # 旧的逐样本调试已停用。这些调用在 DataLoader worker 中执行时会并发追加
        # debug.txt，既产生大量重复内容，也可能拖慢数据加载。
        # dprint("sample.keys", list(sample.keys()))
        # dprint("sample.sample_id", sample["sample_id"])
        # dprint("sample.class_name", sample["class_name"])
        # dprint("sample.text", sample["text"])
        # dprint("sample.coords", sample["coords"].shape)
        # dprint("sample.features", sample["features"].shape)
        # dprint("sample.gt_masks", sample["gt_masks"].shape)
        # dprint("sample.coords[:5]", sample["coords"][:5])
        # dprint("sample.features[:5]", sample["features"][:5])
        # dprint("sample.gt_masks[:, :10]", sample["gt_masks"][:, :10])


        return self._apply_transform(sample)
